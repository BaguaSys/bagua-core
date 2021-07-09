use bagua_core_internal::communicators::{BaguaCommOpConfig, BaguaSingleCommunicator};
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor};
use bagua_core_internal::telemetry::{
    BaguaCommCoreTelemetry, RegisterModelsRequest, TensorDeclaration,
};
use bagua_store::{BaguaKvStore, BaguaKvStoreServer, KvStoreService};
use nix::{
    sys::wait::waitpid,
    unistd::{fork, ForkResult},
};
use std::{
    process::{exit, Command},
    thread::sleep,
    thread,
    time,
    time::Duration,
};
use tokio::runtime::{Builder, Runtime};
use tonic::{Request, Response, Status};

fn init_process_group(gpu_setting: Vec<i32>, nranks: usize, master_addr: String, master_port: i32) {
    let mut kv = loop {
        let kv = BaguaKvStore::open(format!("http://{}:{}", master_addr, master_port));
        match kv {
            Err(err) => {
                println!("BaguaKvStore::open failed, err={:?}", err);
                thread::sleep(time::Duration::from_secs(1));
            },
            Ok(kv) => break kv,
        }
    };

    let nccl_unique_id = if gpu_setting.iter().any(|&i| i == 0) {
        let nccl_unique_id = BaguaSingleCommunicator::generate_nccl_unique_id_str().as_bytes();
        kv.set("nccl_unique_id".into(), nccl_unique_id.clone()).unwrap();

        nccl_unique_id.to_vec()
    } else {
        let nccl_unique_id = loop {
            let nccl_unique_id = kv.get("nccl_unique_id".into());
            match nccl_unique_id {
                Err(err) => {
                    println!("kv.get nccl_unique_id failed, err={:?}", err);
                    thread::sleep(time::Duration::from_secs(1));
                },
                Ok(nccl_unique_id) => break nccl_unique_id,
            }
        };
        println!("nccl_unique_id={:?}", nccl_unique_id);
        assert_eq!(nccl_unique_id, "123".as_bytes());

        nccl_unique_id
    };

    let mut comm_init_threads = Vec::new();
    for gpu_id in gpu_setting {
        let mut t = std::thread::spawn(move || {
            BaguaSingleCommunicator::new(
                gpu_id as usize,
                nranks,
                gpu_id as usize,
                0,
                std::str::from_utf8(&nccl_unique_id).unwrap(),
            )
        });
        comm_init_threads.push(t);
    }

    let mut comm_list = Vec::new();
    for t in comm_init_threads {
        comm_list.push(t.join().unwrap());
    }

    for communicator in comm_list {
        println!("rank={} ready!", communicator.rank());
    }
}

fn main() {
    let nranks = 8;
    let master_addr = "127.0.0.1";
    let master_port = 8123;

    let mut child_id_list = Vec::new();
    let processes_gpu_setting = vec![vec![0], vec![1, 2], vec![3, 4, 5, 6, 7]];
    for gpu_setting in processes_gpu_setting {
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
                // let (sender, receiver) = std::sync::mpsc::channel();
                let kv_store = if gpu_setting.iter().any(|&i| i == 0) {
                    Some(std::thread::spawn(move || {
                        // match sender.send(std::net::TcpStream::connect(("127.0.0.1", 12333))) {
                        //     Ok(()) => {
                        //         let rt = Runtime::new().unwrap();
                        //         let kv_store = KvStoreService::new();
                        //         let service_addr = format!("{}:{}", master_addr, master_port);
                        //         println!("{} listen on service_addr={:?}", std::process::id(), service_addr);
                        //         let service_fut = tonic::transport::Server::builder()
                        //             .add_service(BaguaKvStoreServer::new(kv_store))
                        //             .serve(service_addr.parse().unwrap());
                        //         rt.block_on(service_fut)
                        //             .expect("failed to successfully run the future on RunTime");
                        //     }, // everything good
                        //     Err(_) => {}, // we have been released, don't panic
                        // }
                        let rt = Runtime::new().unwrap();
                        let kv_store = KvStoreService::new();
                        let service_addr = format!("{}:{}", master_addr, master_port);
                        println!("{} listen on service_addr={:?}", std::process::id(), service_addr);
                        let service_fut = tonic::transport::Server::builder()
                            .add_service(BaguaKvStoreServer::new(kv_store))
                            .serve(service_addr.parse().unwrap());
                        rt.block_on(service_fut)
                            .expect("failed to successfully run the future on RunTime");
                    }))
                } else {
                    None
                };

                // let (sender, receiver) = mpsc::channel();
                // let t = thread::spawn(move || {
                //     match sender.send(std::net::TcpStream::connect(("127.0.0.1", 12333))) {
                //         Ok(()) => {
                //             let rt = Runtime::new().unwrap();
                //             let kv_store = KvStoreService::new();
                //             let service_addr = format!("{}:{}", master_addr, master_port);
                //             println!("{} listen on service_addr={:?}", std::process::id(), service_addr);
                //             let service_fut = tonic::transport::Server::builder()
                //                 .add_service(BaguaKvStoreServer::new(kv_store))
                //                 .serve(service_addr.parse().unwrap());
                //             rt.block_on(service_fut)
                //                 .expect("failed to successfully run the future on RunTime");
                //         }, // everything good
                //         Err(_) => {}, // we have been released, don't panic
                //     }
                // });
                // return receiver.recv_timeout(Duration::from_millis(5000));

                init_process_group(gpu_setting, nranks, master_addr.into(), master_port);

                if let Some(server_thread) = kv_store {
                    thread::sleep(time::Duration::from_secs(5));
                    // server_thread.join();
                }
                exit(0);
            }
        }
    }

    for child_id in child_id_list {
        waitpid(child_id, None).unwrap();
    }

    // // 1, 2, 5
    // if let Ok(Fork::Child) = daemon(false, false) {
    //     println!("aabb");
    // }

    println!("ccdd");
}
