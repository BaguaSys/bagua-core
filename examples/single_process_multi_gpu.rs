use bagua_core_internal::communicators::{BaguaCommOpConfig, BaguaSingleCommunicator};
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor};
use bagua_core_internal::telemetry::{
    BaguaCommCoreTelemetry, RegisterModelsRequest, TensorDeclaration,
};
use bagua_core_internal::BaguaCommBackend;
use bagua_store::{BaguaKvStore, BaguaKvStoreServer, KvStoreService};
use nix::{
    sys::wait::waitpid,
    unistd::{fork, ForkResult},
};
use std::{process::exit, thread, time};
use tokio::runtime::Runtime;

fn init_process_group(
    ranks: Vec<usize>,
    nranks: usize,
    gpu_setting: Vec<usize>,
    master_addr: String,
    master_port: i32,
) -> Vec<BaguaSingleCommunicator> {
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

    let nccl_unique_id = if ranks.iter().any(|&i| i == 0) {
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
    for gpu_id in gpu_setting {
        let nranks_clone = nranks.clone();
        let nccl_unique_id_clone = nccl_unique_id.clone();
        let t = std::thread::spawn(move || {
            BaguaSingleCommunicator::new(
                gpu_id as usize,
                nranks_clone,
                gpu_id as usize,
                0,
                std::str::from_utf8(&nccl_unique_id_clone).unwrap(),
            )
        });
        comm_init_threads.push(t);
    }

    let mut comm_list = Vec::new();
    for t in comm_init_threads {
        comm_list.push(t.join().unwrap());
    }

    for communicator in &comm_list {
        println!("rank={} ready!", communicator.rank());
    }

    comm_list
}

pub struct BaguaBackendForKai {
    pub kv_store: Option<(
        std::thread::JoinHandle<()>,
        tokio::sync::oneshot::Sender<()>,
    )>,
    pub ranks: Vec<usize>,
    pub nranks: usize,
    pub gpu_setting: Vec<usize>,
    pub bagua_backends: Vec<BaguaCommBackend>,
    pub communicators: Vec<BaguaSingleCommunicator>,
}

impl BaguaBackendForKai {
    const BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP: usize = 100;

    pub fn new(
        ranks: Vec<usize>,
        nranks: usize,
        gpu_setting: Vec<usize>,
        master_addr: String,
        master_port: i32,
        autotune_service_addr: String,
        autotune_service_port: i32,
        tensors: &[&BaguaTensor],
    ) -> BaguaBackendForKai {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let service_addr = format!("{}:{}", master_addr.clone(), master_port);
        let kv_store = if gpu_setting.iter().any(|&i| i == 0) {
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
            kv_store: kv_store,
            ranks: ranks.clone(),
            nranks: nranks,
            gpu_setting: gpu_setting.clone(),
            bagua_backends: gpu_setting
                .clone()
                .iter()
                .map(|&device_id| {
                    BaguaCommBackend::new(
                        BaguaBackendForKai::BAGUA_BACKEND_SCHEDULE_CHANNEL_CAP,
                        device_id,
                    )
                })
                .collect(),
            communicators: init_process_group(ranks, nranks, gpu_setting, master_addr, master_port),
        }
    }
}

impl Drop for BaguaBackendForKai {
    fn drop(&mut self) {
        let _ = self.kv_store;
        // if let Some((server_thread, tx)) = self.kv_store {
        //     tx.send(()).unwrap();
        //     server_thread.join();
        // }
    }
}

fn main() {
    let nranks = 8;
    let master_addr = "127.0.0.1";
    let master_port = 8123;

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
                let backend4kai = BaguaBackendForKai::new(
                    gpu_setting.clone(),
                    nranks,
                    gpu_setting.clone(),
                    master_addr.clone().into(),
                    master_port,
                    master_addr.clone().into(),
                    123,
                    &[] as &[&BaguaTensor],
                );
                thread::sleep(time::Duration::from_secs(5));
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
