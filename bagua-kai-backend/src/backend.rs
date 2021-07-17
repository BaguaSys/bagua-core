use std::{sync::Arc, sync::Mutex, time};
use tokio::runtime::Runtime;

use bagua_core_internal::{
    communicators::{BaguaCommOpConfig, BaguaSingleCommunicator},
    cuda_utils::cuda_memcpy_D2D_async,
    datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype},
    resource_pool::{CudaMemory, CUDA_DEVICE_MEMORY_POOL},
    telemetry::{BaguaCommCoreTelemetry, RegisterTensorsRequest, TensorDeclaration},
    BaguaCommBackend, BaguaCommOpChannels,
};
use bagua_store::{BaguaKvStore, BaguaKvStoreServer, KvStoreService};

use sized_object_pool::DynamicPoolItem;

fn init_process_group(
    rank: usize,
    nranks: usize,
    device_id: usize,
    master_addr: String,
    master_port: i32,
    cuda_stream_ptr: u64,
) -> BaguaSingleCommunicator {
    let mut kv = loop {
        let kv = BaguaKvStore::open(format!("http://{}:{}", master_addr, master_port));
        match kv {
            Err(err) => {
                println!("BaguaKvStore::open failed, err={:?}", err);
                std::thread::sleep(time::Duration::from_secs(1));
            }
            Ok(kv) => break kv,
        }
    };

    let nccl_unique_id = if rank == 0 {
        let nccl_unique_id = BaguaSingleCommunicator::generate_nccl_unique_id_str();
        kv.set("nccl_unique_id".into(), nccl_unique_id.clone().as_bytes())
            .unwrap();

        nccl_unique_id.as_bytes().to_vec()
    } else {
        let nccl_unique_id = loop {
            let nccl_unique_id = kv.get("nccl_unique_id".into());
            match nccl_unique_id {
                Err(err) => {
                    println!("kv.get nccl_unique_id failed, err={:?}", err);
                    std::thread::sleep(time::Duration::from_secs(1));
                }
                Ok(nccl_unique_id) => break nccl_unique_id,
            }
        };
        println!("nccl_unique_id={:?}", nccl_unique_id);

        nccl_unique_id
    };

    BaguaSingleCommunicator::new(
        device_id,
        nranks,
        device_id,
        cuda_stream_ptr,
        std::str::from_utf8(&nccl_unique_id).unwrap(),
    )
}

type callback_func = Arc<dyn Fn() + Send + Sync + 'static>;

pub struct BaguaSingleBackendForKAI {
    pub rank: usize,
    pub nranks: usize,
    pub device_id: usize,
    pub backend: BaguaCommBackend,
    pub comm: BaguaSingleCommunicator,
    pub kv_store: Option<(
        std::thread::JoinHandle<()>,
        tokio::sync::oneshot::Sender<()>,
    )>,
    pub bucket_callback: Vec<Arc<Mutex<Vec<callback_func>>>>,
    // pub bucket_callback: Vec<Arc<dyn Fn() + Send + Sync + 'static>>,
    pub tensor_name_to_bucket_id: std::collections::HashMap<String, usize>,
    pub tmpbuff: DynamicPoolItem<CudaMemory>,
    pub inner_tensors: std::collections::HashMap<String, BaguaTensor>,
}

impl BaguaSingleBackendForKAI {
    const BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP: usize = 100;

    pub fn new(
        rank: usize,
        nranks: usize,
        device_id: usize,
        master_addr: String,
        master_port: i32,
        cuda_stream_ptr: u64,
    ) -> BaguaSingleBackendForKAI {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let service_addr = format!("{}:{}", master_addr.clone(), master_port);
        let kv_store = if device_id == 0 {
            Some((
                std::thread::spawn(move || {
                    let rt = Runtime::new().unwrap();
                    let kv_store = KvStoreService::new();
                    println!(
                        "{} listen on service_addr={:?}",
                        std::process::id(),
                        service_addr
                    );
                    let service_fut = tonic::transport::Server::builder()
                        .add_service(BaguaKvStoreServer::new(kv_store))
                        .serve_with_shutdown(service_addr.parse().unwrap(), async {
                            rx.await.ok();
                        });
                    rt.block_on(service_fut)
                        .expect("failed to successfully run the future on RunTime");
                }),
                tx,
            ))
        } else {
            None
        };

        Self {
            rank: rank,
            nranks: nranks,
            device_id: device_id,
            backend: BaguaCommBackend::new(
                BaguaSingleBackendForKAI::BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP,
                device_id,
            ),
            comm: init_process_group(
                rank,
                nranks,
                device_id,
                master_addr,
                master_port,
                cuda_stream_ptr,
            ),
            kv_store: kv_store,
            bucket_callback: vec![],
            tensor_name_to_bucket_id: Default::default(),
            tmpbuff: CUDA_DEVICE_MEMORY_POOL[device_id]
                .try_pull(1)
                .expect("cannot allocate gpu memory"),
            inner_tensors: Default::default(),
        }
    }

    pub fn register_ordered_buckets(&mut self, buckets: Vec<BaguaBucket>) {
        let mut buckets_ref = Vec::new();
        for bucket in &buckets {
            buckets_ref.push(bucket);
        }

        self.backend.register_ordered_buckets(&buckets_ref).unwrap();
        self.bucket_callback = Vec::with_capacity(buckets.len());
        for (i, _) in buckets.iter_mut().enumerate() {
            self.bucket_callback.push(Arc::new(Mutex::new(vec![])));
        }
        for (i, bucket) in buckets.clone().iter_mut().enumerate() {
            for tensor in &bucket.inner.lock().tensors {
                self.tensor_name_to_bucket_id.insert(tensor.name(), i);
            }

            bucket.append_centralized_synchronous_op(
                Some(&self.comm),
                Some(&self.comm),
                false,
                true,
                false,
                None,
            );
            let callback_list = self.bucket_callback[i].clone();
            bucket.append_custom_op(Arc::new(
                move |_x: Arc<BaguaBucket>, _y: &BaguaCommOpChannels| {
                    for callback in callback_list.lock().unwrap().iter() {
                        callback();
                    }
                },
            ));
        }
    }

    pub fn register_tensors(
        &mut self,
        model_name: String,
        tensors: Vec<BaguaTensor>,
        autotune_service_addr: String,
        autotune_service_port: i32,
    ) {
        let total_bytes = (&tensors).iter().map(|b| b.bytes()).sum();
        self.tmpbuff = CUDA_DEVICE_MEMORY_POOL[self.device_id]
            .try_pull(total_bytes)
            .expect("cannot allocate gpu memory");
        let mut tmpbuff_ptr = self.tmpbuff.ptr;

        let telemetry = BaguaCommCoreTelemetry::new(&*format!(
            "{}:{}",
            autotune_service_addr, autotune_service_port
        ));
        let req = RegisterTensorsRequest {
            model_name: model_name,
            whether_to_bucket: true,
            tensor_list: tensors
                .clone()
                .iter()
                .map(|t| TensorDeclaration {
                    name: t.name(),
                    num_elements: t.num_elements(),
                    dtype: t.dtype().to_lowercase(),
                })
                .collect(),
        };
        println!("req={:?}", req);
        let rsp = telemetry.register_tensors(req).unwrap();
        let mut buckets = Vec::new();
        println!("buckets={:?}", rsp.recommended_hyperparameters.buckets);
        self.inner_tensors.clear();
        for (i, td_bucket) in rsp.recommended_hyperparameters.buckets.iter().enumerate() {
            let mut inner_tensor_holder = Vec::new();
            for td_tensor in td_bucket.iter() {
                let filter_list: Vec<&BaguaTensor> = tensors
                    .iter()
                    .filter(|t| t.name() == td_tensor.name)
                    .collect();
                assert_eq!(
                    filter_list.len(),
                    1,
                    "Invalid filter_list={:?}",
                    filter_list
                );
                let input_tensor = filter_list[0];
                let inner_tensor = BaguaTensor::new(
                    input_tensor.name(),
                    self.device_id,
                    tmpbuff_ptr,
                    input_tensor.num_elements(),
                    input_tensor.inner.read().raw.dtype(),
                    0,
                );
                self.inner_tensors
                    .insert(input_tensor.name(), inner_tensor.clone());
                inner_tensor_holder.push(inner_tensor);

                tmpbuff_ptr += input_tensor.bytes() as u64;
            }

            let mut tensors_ref = Vec::<&BaguaTensor>::new();
            for inner_tensor in &inner_tensor_holder {
                tensors_ref.push(inner_tensor);
            }

            let bucket =
                BaguaBucket::new(tensors_ref.as_slice(), &*format!("bucket-{}", i)).unwrap();
            for t in tensors_ref {
                self.tensor_name_to_bucket_id.insert(t.name(), i);
            }
            buckets.push(bucket);
        }
        self.register_ordered_buckets(buckets);
    }

    pub fn allreduce_inplace(
        &mut self,
        tensor: &BaguaTensor,
        ready_cuda_event_ptr: u64,
        callback: Arc<dyn Fn() + Send + Sync + 'static>,
    ) {
        let bucket_id = *self.tensor_name_to_bucket_id.get(&tensor.name()).unwrap();
        let callback_list = self.bucket_callback[bucket_id];
        callback_list.lock().unwrap().push(callback);

        self.backend
            .mark_communication_ready(&tensor, ready_cuda_event_ptr)
            .unwrap();
    }

    pub fn allreduce(
        &mut self,
        input_tensor: &BaguaTensor,
        output_tensor: &BaguaTensor,
        ready_cuda_event_ptr: u64,
        callback: Arc<dyn Fn() + Send + Sync + 'static>,
    ) {
        let comm_stream_ptr = self.comm.inner.stream_ptr;

        let inner_tensor = self
            .inner_tensors
            .get(&input_tensor.name())
            .unwrap()
            .clone();
        let ready_cuda_event_ptr = unsafe {
            cuda_memcpy_D2D_async(
                inner_tensor.data_ptr(),
                input_tensor.data_ptr(),
                input_tensor.bytes(),
                comm_stream_ptr,
                ready_cuda_event_ptr,
            )
        };

        let bucket_id = *self
            .tensor_name_to_bucket_id
            .get(&input_tensor.name())
            .unwrap();
        let callback_list = self.bucket_callback[bucket_id].clone();
        let output_tensor_clone = output_tensor.clone();
        let inner_tensor_clone = inner_tensor.clone();
        let new_callback = Arc::new(move || {
            unsafe {
                cuda_memcpy_D2D_async(
                    output_tensor_clone.data_ptr(),
                    inner_tensor_clone.data_ptr(),
                    inner_tensor_clone.bytes(),
                    comm_stream_ptr,
                    0,
                );
            }

            callback();
        });
        callback_list.lock().unwrap().push(new_callback);

        self.backend
            .mark_communication_ready(&inner_tensor, ready_cuda_event_ptr)
            .unwrap();
    }
}

impl Drop for BaguaSingleBackendForKAI {
    fn drop(&mut self) {
        if let Some((server_thread, tx)) = self.kv_store.take() {
            tx.send(()).unwrap();
            server_thread.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nix::{
        sys::wait::waitpid,
        unistd::{fork, ForkResult},
    };

    use bagua_core_internal::{
        cuda_utils::{
            cuda_memcpy_device_to_host_sync, cuda_memcpy_host_to_device_sync, cuda_set_device,
        },
        resource_pool::CudaMemory,
    };

    #[test]
    fn test_bagua_single_backend_for_kai() {
        let nranks = 8;
        let mut master_addr = "127.0.0.1".to_string();
        let mut master_port = 8123;

        let mut child_id_list = Vec::new();
        let processes_gpu_setting = vec![vec![0], vec![1, 2], vec![3, 4, 5, 6, 7]];
        for gpu_setting in processes_gpu_setting {
            let gpu_setting: Vec<usize> = gpu_setting.iter().map(|&x| x as usize).collect();
            match fork().expect("Failed to fork process") {
                ForkResult::Parent { child } => {
                    child_id_list.push(child);
                }
                ForkResult::Child => {
                    println!("gpu_setting={:?}", gpu_setting);
                    let mut memory_holder = Vec::new();
                    let mut io_tensors = Vec::new();
                    let tensor_name = "tensor-1".to_string();
                    for device_id in gpu_setting.clone() {
                        let input_ptr = unsafe {
                            cuda_set_device(device_id as u64);

                            let bytes = 4;
                            let device_x = Arc::new(CudaMemory::new(bytes));
                            memory_holder.push(device_x.clone());
                            let host_x = device_id as f32;
                            let host_x_ptr: *const f32 = &host_x;
                            cuda_memcpy_host_to_device_sync(
                                device_x.ptr,
                                host_x_ptr as u64,
                                bytes as i32,
                            );

                            device_x.ptr
                        };
                        let output_ptr = unsafe {
                            cuda_set_device(device_id as u64);

                            let bytes = 4;
                            let device_x = Arc::new(CudaMemory::new(bytes));
                            memory_holder.push(device_x.clone());

                            device_x.ptr
                        };

                        io_tensors.push((
                            BaguaTensor::new(
                                tensor_name.clone(),
                                device_id,
                                input_ptr,
                                1,
                                BaguaTensorDtype::F32,
                                0,
                            ),
                            BaguaTensor::new(
                                tensor_name.clone(),
                                device_id,
                                output_ptr,
                                1,
                                BaguaTensorDtype::F32,
                                0,
                            ),
                        ));
                    }
                    let mut workers = Vec::new();
                    for (i, device_id) in gpu_setting.iter().enumerate() {
                        let device_id_clone = device_id.clone();
                        let master_addr_clone = master_addr.clone();
                        let in_and_out = io_tensors[i].clone();
                        workers.push(std::thread::spawn(move || {
                            let mut backend4kai = BaguaSingleBackendForKAI::new(
                                device_id_clone,
                                nranks,
                                device_id_clone,
                                master_addr_clone,
                                master_port,
                                0,
                            );
                            let io_list = vec![in_and_out.clone()];
                            let mut tensors_ref = Vec::new();
                            for in_and_out in &io_list {
                                tensors_ref.push(&(in_and_out.0));
                            }
                            let bucket =
                                BaguaBucket::new(tensors_ref.as_slice(), "bucket-1").unwrap();
                            backend4kai.register_ordered_buckets(vec![bucket]);

                            for in_and_out in io_list {
                                let ptr = in_and_out.1.inner.read().raw.data_ptr();
                                backend4kai.allreduce(
                                    &(in_and_out.0),
                                    &(in_and_out.1),
                                    0,
                                    Arc::new(move || {
                                        let result = unsafe {
                                            cuda_set_device(device_id_clone as u64);
                                            let host_x: f32 = 0.;
                                            let host_x_ptr: *const f32 = &host_x;
                                            cuda_memcpy_device_to_host_sync(
                                                host_x_ptr as u64,
                                                ptr,
                                                4,
                                            );

                                            host_x
                                        };

                                        assert_eq!(result, 3.5);
                                    }),
                                );
                            }

                            return backend4kai;
                        }));
                    }

                    for worker in workers {
                        worker.join().unwrap();
                    }
                    std::thread::sleep(time::Duration::from_secs(5));
                    std::process::exit(0);
                }
            }
        }

        for child_id in child_id_list {
            waitpid(child_id, None).unwrap();
        }
    }
}
