extern crate argparse;

use argparse::{ArgumentParser, Store, StoreTrue};
use cpp::cpp;
use nix::{
    sys::wait::waitpid,
    unistd::{fork, ForkResult},
};
use std::{
    sync::Arc,
    collections::HashMap,
    thread, time,
};
use tokio::runtime::Runtime;
use tracing;
use tracing::{info, Level};
use tracing_subscriber;

use bagua_core_internal::communicators::{BaguaCommOpConfig, BaguaSingleCommunicator};
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype};
use bagua_core_internal::resource_pool::CudaMemory;
use bagua_core_internal::telemetry::{
    BaguaCommCoreTelemetry, RegisterTensorsRequest, TensorDeclaration,
};
use bagua_core_internal::{BaguaCommBackend, BaguaCommOpChannels};
use bagua_store::{BaguaKvStore, BaguaKvStoreServer, KvStoreService};

cpp! {{
#include <nccl.h>
#include <stdio.h>
#include <iostream>

#define CUDACHECK(cmd) do { cudaError_t e = cmd; if( e != cudaSuccess ) { printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)
}}

fn init_process_group(
    rank: usize,
    nranks: usize,
    device_id: usize,
    master_addr: String,
    master_port: i32,
) -> BaguaSingleCommunicator {
    let mut kv = loop {
        let kv = BaguaKvStore::open(format!("http://{}:{}", master_addr, master_port));
        match kv {
            Err(err) => {
                println!("BaguaKvStore::open failed, err={:?}", err);
                thread::sleep(time::Duration::from_secs(1));
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
                    thread::sleep(time::Duration::from_secs(1));
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
        0,
        std::str::from_utf8(&nccl_unique_id).unwrap(),
    )
}

pub struct BaguaSingleBackendForKAI {
    pub rank: usize,
    pub nranks: usize,
    pub device_id: usize,
    pub backend: BaguaCommBackend,
    pub comm: BaguaSingleCommunicator,
    pub registered_tensors: Vec<BaguaTensor>,
    pub registered_buckets: Vec<BaguaBucket>,
    pub kv_store: Option<(
        std::thread::JoinHandle<()>,
        tokio::sync::oneshot::Sender<()>,
    )>,
    pub bucket_callback: Vec<Arc<Fn()>>,
    pub tensor_name_to_bucket_id: std::collections::HashMap<String, i32>,
}

impl BaguaSingleBackendForKAI {
    const BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP: usize = 100;

    pub fn new(
        rank: usize,
        nranks: usize,
        device_id: usize,
        master_addr: String,
        master_port: i32,
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
        let mut backend = BaguaCommBackend::new(
            BaguaSingleBackendForKAI::BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP,
            device_id,
        );

        Self {
            rank: rank,
            nranks: nranks,
            device_id: device_id,
            backend: backend,
            comm: init_process_group(rank, nranks, device_id, master_addr, master_port),
            registered_tensors: vec![],
            registered_buckets: vec![],
            kv_store: kv_store,
        }
    }

    pub fn register_tensors(
        &mut self,
        model_name: String,
        tensors: Vec<BaguaTensor>,
        autotune_service_addr: String,
        autotune_service_port: i32,
    ) {
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
        for (i, td_bucket) in rsp.recommended_hyperparameters.buckets.iter().enumerate() {
            let mut tensors_ref = Vec::<&BaguaTensor>::new();
            for td_tensor in td_bucket.mut_iter() {
                let t: Vec<&BaguaTensor> = tensors
                    .iter()
                    .filter(|t| t.name() == td_tensor.name)
                    .collect();
                tensors_ref.extend(t);
            }
            bucket = BaguaBucket::new(tensors_ref.as_slice(), &*format!("bucket-{}", i)).unwrap();
            for t in tensors_ref {
                self.tensor_name_to_bucket_id.insert(t.name(), i);
            }
            buckets.push(bucket);
        }
        let mut buckets_ref = Vec::new();
        for bucket in &buckets {
            buckets_ref.push(bucket);
        }
        self.backend
            .register_ordered_buckets(buckets_ref.as_slice())
            .unwrap();
        for bucket in buckets.iter_mut() {
            bucket.append_centralized_synchronous_op(
                Some(&self.comm),
                Some(&self.comm),
                false,
                true,
                false,
                None,
            );
            let rank_clone = self.rank;
            bucket.append_custom_op(Arc::new(
                move |_x: Arc<BaguaBucket>, _y: &BaguaCommOpChannels| {
                    println!("rank_clone={}", rank_clone);
                },
            ));
        }

        self.bucket_callback = Vec::with_capacity(buckets.len())
        self.registered_tensors = tensors;
        self.registered_buckets = buckets;
    }

    pub fn mark_tensor_ready(&mut self, tensor: &BaguaTensor, ready_cuda_event_ptr: u64) {
        self.backend
            .mark_communication_ready(tensor, ready_cuda_event_ptr)
            .unwrap();
    }

    pub fn wait_pending_comm_ops(&self) {
        self.backend.wait_pending_comm_ops();
    }

    pub fn allreduce(
        &mut self,
        tensor: &BaguaTensor,
        ready_cuda_event_ptr: u64,
        callback: Arc<dyn Fn()>,
    ) {
        let bucket_id = self.tensor_name_to_bucket_id.get(tensor.name()).unwrap();
        let raw_callback = self.bucket_callback[bucket_id].clone();
        let new_callback = Arc::new(move || {
            raw_callback();
            callback();
        });
        self.bucket_callback[bucket_id] = new_callback;
    }
}

impl Drop for BaguaSingleBackendForKAI {
    fn drop(&mut self) {
        if let Some((server_thread, tx)) = self.kv_store.take() {
            tx.send(()).unwrap();
            server_thread.join();
        }
    }
}

fn main() {
    let nranks = 8;
    let mut master_addr = "127.0.0.1".to_string();
    let mut master_port = 8123;
    let mut autotune_service_addr = "127.0.0.1".to_string();
    let mut autotune_service_port = 8124;

    {
        let mut ap = ArgumentParser::new();
        ap.refer(&mut master_addr)
            .add_option(&["--master_addr"], Store, "");
        ap.refer(&mut master_port)
            .add_option(&["--master_port"], Store, "");
        ap.refer(&mut autotune_service_addr)
            .add_option(&["--autotune_service_addr"], Store, "");
        ap.refer(&mut autotune_service_port)
            .add_option(&["--autotune_service_port"], Store, "");
    }

    let mut child_id_list = Vec::new();
    let processes_gpu_setting = vec![vec![0], vec![1, 2], vec![3, 4, 5, 6, 7]];
    for gpu_setting in processes_gpu_setting {
        let gpu_setting: Vec<usize> = gpu_setting.iter().map(|&x| x as usize).collect();
        match fork().expect("Failed to fork process") {
            ForkResult::Parent { child } => {
                // println!("Try to kill me to check if the target process will be killed");
                child_id_list.push(child);
                // // Do not forget to wait for the fork in order to prevent it from becoming a zombie!!!
                // waitpid(Some(child), None).unwrap();
                // // You have 120 seconds to kill the process :)
                // sleep(Duration::from_secs(120));
            }
            ForkResult::Child => {
                println!("gpu_setting={:?}", gpu_setting);
                let mut tensors = Vec::new();
                for device_id in gpu_setting.clone() {
                    let ptr = unsafe {
                        cpp::cpp!([device_id as "size_t"] -> u64 as "void*"
                        {
                            size_t bytes = 4;
                            CUDACHECK(cudaSetDevice(device_id));
                            void* ptr = 0;
                            CUDACHECK(cudaMalloc(&ptr, bytes));
                            float x = device_id;
                            CUDACHECK(cudaMemcpy(ptr, (void*)&x, bytes, cudaMemcpyHostToDevice));

                            return ptr;
                        })
                    };
                    tensors.push(BaguaTensor::new(
                        "tensor-1".to_string(),
                        device_id,
                        ptr,
                        1,
                        BaguaTensorDtype::F32,
                        0,
                    ));
                }
                let mut workers = Vec::new();
                for (i, device_id) in gpu_setting.iter().enumerate() {
                    let device_id_clone = device_id.clone();
                    let master_addr_clone = master_addr.clone();
                    let autotune_service_addr_clone = autotune_service_addr.clone();
                    let t = tensors[i].clone();
                    workers.push(std::thread::spawn(move || {
                        let mut backend4kai = BaguaSingleBackendForKAI::new(
                            device_id_clone,
                            nranks,
                            device_id_clone,
                            master_addr_clone,
                            master_port,
                        );
                        backend4kai.register_tensors(
                            "default_model".to_string(),
                            vec![t.clone()],
                            autotune_service_addr_clone,
                            autotune_service_port,
                        );
                        backend4kai.mark_tensor_ready(&t, 0);
                        backend4kai.wait_pending_comm_ops();
                        let ptr = t.inner.read().raw.data_ptr();
                        unsafe {
                            cpp::cpp!([device_id_clone as "size_t", ptr as "void*"]
                            {
                                size_t bytes = 4;
                                CUDACHECK(cudaSetDevice(device_id_clone));
                                float x = 0.;
                                CUDACHECK(cudaMemcpy((void*)&x, ptr, bytes, cudaMemcpyDeviceToHost));
                                printf("avg=%f\n", x);
                            });
                        };
                        return backend4kai;
                    }));
                }

                for worker in workers {
                    worker.join();
                }
                thread::sleep(time::Duration::from_secs(5));
                std::process::exit(0);
            }
        }
    }

    for child_id in child_id_list {
        waitpid(child_id, None).unwrap();
    }
}
