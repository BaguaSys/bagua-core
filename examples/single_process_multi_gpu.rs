extern crate argparse;

use argparse::{ArgumentParser, Store, StoreTrue};
use cpp::cpp;
use nix::{
    sys::wait::waitpid,
    unistd::{fork, ForkResult},
};
use std::{thread, time};
use tokio::runtime::Runtime;

use bagua_core_internal::communicators::{BaguaCommOpConfig, BaguaSingleCommunicator};
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype};
use bagua_core_internal::resource_pool::CudaMemory;
use bagua_core_internal::telemetry::{
    BaguaCommCoreTelemetry, RegisterTensorsRequest, TensorDeclaration,
};
use bagua_core_internal::BaguaCommBackend;
use bagua_store::{BaguaKvStore, BaguaKvStoreServer, KvStoreService};

cpp! {{
#include <nccl.h>
#include <stdio.h>
#include <iostream>

#define CUDACHECK(cmd) do { cudaError_t e = cmd; if( e != cudaSuccess ) { printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)
}}

// fn init_process_group(
//     ranks: Vec<usize>,
//     nranks: usize,
//     gpu_setting: Vec<usize>,
//     master_addr: String,
//     master_port: i32,
// ) -> Vec<BaguaSingleCommunicator> {
//     let mut kv = loop {
//         let kv = BaguaKvStore::open(format!("http://{}:{}", master_addr, master_port));
//         match kv {
//             Err(err) => {
//                 println!("BaguaKvStore::open failed, err={:?}", err);
//                 thread::sleep(time::Duration::from_secs(1));
//             }
//             Ok(kv) => break kv,
//         }
//     };

//     let nccl_unique_id = if ranks.iter().any(|&i| i == 0) {
//         let nccl_unique_id = BaguaSingleCommunicator::generate_nccl_unique_id_str();
//         kv.set("nccl_unique_id".into(), nccl_unique_id.clone().as_bytes())
//             .unwrap();

//         nccl_unique_id.as_bytes().to_vec()
//     } else {
//         let nccl_unique_id = loop {
//             let nccl_unique_id = kv.get("nccl_unique_id".into());
//             match nccl_unique_id {
//                 Err(err) => {
//                     println!("kv.get nccl_unique_id failed, err={:?}", err);
//                     thread::sleep(time::Duration::from_secs(1));
//                 }
//                 Ok(nccl_unique_id) => break nccl_unique_id,
//             }
//         };
//         println!("nccl_unique_id={:?}", nccl_unique_id);

//         nccl_unique_id
//     };

//     let mut comm_init_threads = Vec::new();
//     for gpu_id in gpu_setting {
//         let nranks_clone = nranks.clone();
//         let nccl_unique_id_clone = nccl_unique_id.clone();
//         let t = std::thread::spawn(move || {
//             BaguaSingleCommunicator::new(
//                 gpu_id as usize,
//                 nranks_clone,
//                 gpu_id as usize,
//                 0,
//                 std::str::from_utf8(&nccl_unique_id_clone).unwrap(),
//             )
//         });
//         comm_init_threads.push(t);
//     }

//     let mut comm_list = Vec::new();
//     for t in comm_init_threads {
//         comm_list.push(t.join().unwrap());
//     }

//     for communicator in &comm_list {
//         println!("rank={} ready!", communicator.rank());
//     }

//     comm_list
// }

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

    let mut comm_init_threads = Vec::new();
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
            // TODO @shjwudp: split new to init_process_group and register_tensors
            model_name: model_name,
            whether_to_bucket: true,
            tensor_list: tensors
                .clone()
                .iter()
                .map(|t| TensorDeclaration {
                    name: t.name(),
                    num_elements: t.num_elements(),
                    dtype: t.dtype(),
                })
                .collect(),
        };
        let rsp = telemetry.register_tensors(req).unwrap();
        let mut buckets = Vec::new();
        for (i, td_bucket) in rsp.recommended_hyperparameters.buckets.iter().enumerate() {
            let mut tensors_ref = Vec::<&BaguaTensor>::new();
            for td_tensor in td_bucket.iter() {
                let t: Vec<&BaguaTensor> = tensors
                    .iter()
                    .filter(|t| t.name() == td_tensor.name)
                    .collect();
                tensors_ref.extend(t);
            }
            buckets
                .push(BaguaBucket::new(tensors_ref.as_slice(), &*format!("bucket-{}", i)).unwrap());
        }
        let mut buckets_ref = Vec::new();
        for bucket in &buckets {
            buckets_ref.push(bucket);
        }
        self.backend.register_ordered_buckets(buckets_ref.as_slice());
        for bucket in buckets.iter_mut() {
            bucket.append_centralized_synchronous_op(
                Some(&self.comm), Some(&self.comm), false, true, false, None);
        }

        self.registered_tensors = tensors;
        self.registered_buckets = buckets;
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

// pub struct BaguaBackendForKAI {
//     pub kv_store: Option<(
//         std::thread::JoinHandle<()>,
//         tokio::sync::oneshot::Sender<()>,
//     )>,
//     pub ranks: Vec<usize>,
//     pub nranks: usize,
//     pub gpu_setting: Vec<usize>,
//     pub bagua_backends: Vec<BaguaCommBackend>,
//     pub communicators: Vec<BaguaSingleCommunicator>,
//     pub registered_tensors: Vec<BaguaTensor>,
//     pub registered_buckets: Vec<BaguaBucket>,
//     pub task_schedule_worker: std::thread::JoinHandle<()>,
// }

// impl BaguaBackendForKAI {
//     const BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP: usize = 100;

//     pub fn new(
//         ranks: Vec<usize>,
//         nranks: usize,
//         gpu_setting: Vec<usize>,
//         master_addr: String,
//         master_port: i32,
//         autotune_service_addr: String,
//         autotune_service_port: i32,
//         tensors: Vec<BaguaTensor>,
//     ) -> BaguaBackendForKAI {
//         let (tx, rx) = tokio::sync::oneshot::channel::<()>();
//         let service_addr = format!("{}:{}", master_addr.clone(), master_port);
//         let kv_store = if gpu_setting.iter().any(|&i| i == 0) {
//             Some((
//                 std::thread::spawn(move || {
//                     let rt = Runtime::new().unwrap();
//                     let kv_store = KvStoreService::new();
//                     println!(
//                         "{} listen on service_addr={:?}",
//                         std::process::id(),
//                         service_addr
//                     );
//                     let service_fut = tonic::transport::Server::builder()
//                         .add_service(BaguaKvStoreServer::new(kv_store))
//                         .serve_with_shutdown(service_addr.parse().unwrap(), async {
//                             rx.await.ok();
//                         });
//                     rt.block_on(service_fut)
//                         .expect("failed to successfully run the future on RunTime");
//                 }),
//                 tx,
//             ))
//         } else {
//             None
//         };
//         let mut backends: Vec<BaguaCommBackend> = gpu_setting
//             .clone()
//             .iter()
//             .map(|&device_id| {
//                 BaguaCommBackend::new(
//                     BaguaBackendForKAI::BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP,
//                     device_id,
//                 )
//             })
//             .collect();

//         let telemetry = BaguaCommCoreTelemetry::new(&*format!(
//             "{}:{}",
//             autotune_service_addr, autotune_service_port
//         ));
//         let req = RegisterTensorsRequest {
//             // TODO @shjwudp: split new to init_process_group and register_tensors
//             model_name: "default_model".to_string(),
//             whether_to_bucket: true,
//             tensor_list: tensors
//                 .clone()
//                 .iter()
//                 .map(|t| TensorDeclaration {
//                     name: t.name(),
//                     num_elements: t.num_elements(),
//                     dtype: t.dtype(),
//                 })
//                 .collect(),
//         };
//         let rsp = telemetry.register_tensors(req).unwrap();
//         let mut buckets = Vec::new();
//         for (i, td_bucket) in rsp.recommended_hyperparameters.buckets.iter().enumerate() {
//             let mut tensors_ref = Vec::<&BaguaTensor>::new();
//             for td_tensor in td_bucket.iter() {
//                 let t: Vec<&BaguaTensor> = tensors
//                     .iter()
//                     .filter(|t| t.name() == td_tensor.name)
//                     .collect();
//                 tensors_ref.extend(t);
//             }
//             buckets
//                 .push(BaguaBucket::new(tensors_ref.as_slice(), &*format!("bucket-{}", i)).unwrap());
//         }
//         let mut buckets_ref = Vec::new();
//         for bucket in &buckets {
//             buckets_ref.push(bucket);
//         }
//         for backend in &mut backends {
//             backend.register_ordered_buckets(buckets_ref.as_slice());
//         }
//         for bucket in buckets.iter_mut().enumerate() {
//             bucket.append_centralized_synchronous_op()
//         }

//         Self {
//             kv_store: kv_store,
//             ranks: ranks.clone(),
//             nranks: nranks,
//             gpu_setting: gpu_setting.clone(),
//             bagua_backends: backends,
//             communicators: init_process_group(ranks, nranks, gpu_setting, master_addr, master_port),
//             registered_tensors: tensors,
//             registered_buckets: buckets,
//         }
//     }

//     pub fn allreduce(
//         t: BaguaTensor,
//         ready_cuda_event_ptr: u64,
//         Box<dyn Fn() -> ()>,
//     ) {

//     }
// }

// impl Drop for BaguaBackendForKAI {
//     fn drop(&mut self) {
//         if let Some((server_thread, tx)) = self.kv_store.take() {
//             tx.send(()).unwrap();
//             server_thread.join();
//         }
//     }
// }

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
                            CUDACHECK(cudaMemcpy(ptr, (void*)&x, 4, cudaMemcpyHostToDevice));

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
                let workers = Vec::new();
                for (i, device_id) in gpu_setting.iter().enumerate() {
                    workers.push(std::thread::spawn(move || {
                        let backend4kai = BaguaSingleBackendForKAI::new(
                            *device_id,
                            nranks,
                            *device_id,
                            master_addr.clone(),
                            master_port,
                        );
                        backend4kai.register_tensors("default_model".to_string(), vec![tensors[i].clone()], autotune_service_addr.clone(), autotune_service_port);
                        return backend4kai;
                    }));
                }

                for worker in &workers {
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
