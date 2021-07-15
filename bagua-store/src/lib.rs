pub mod memdb;

use crate::memdb::Memdb;
use tonic::{Request, Response, Status};

pub mod bagua {
    pub mod kv_store {
        tonic::include_proto!("bagua.kv_store"); // The string specified here must match the proto package name
    }
}

use bagua::kv_store::kv_store_server::{KvStore, KvStoreServer};
use bagua::kv_store::{
    DeleteRequest, DeleteResponse, GetRequest, GetResponse, SetRequest, SetResponse,
};

pub type BaguaKvStoreServer<T> = bagua::kv_store::kv_store_server::KvStoreServer<T>;

#[derive(Debug)]
pub struct KvStoreService {
    memdb: Memdb,
}

impl KvStoreService {
    pub fn new() -> Self {
        Self {
            memdb: Memdb::open(),
        }
    }
}

#[tonic::async_trait]
impl KvStore for KvStoreService {
    async fn set(&self, request: Request<SetRequest>) -> Result<Response<SetResponse>, Status> {
        let request = request.into_inner();
        self.memdb.set(request.key, request.value);

        Ok(Response::new(SetResponse {}))
    }

    async fn get(&self, request: Request<GetRequest>) -> Result<Response<GetResponse>, Status> {
        let request = request.into_inner();

        let value = self.memdb.get(request.key);
        if value.is_none() {
            return Err(Status::invalid_argument(format!("{:?}", value)));
        }

        Ok(Response::new(GetResponse {
            value: value.unwrap(),
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let request = request.into_inner();

        self.memdb.del(request.key);

        Ok(Response::new(DeleteResponse {}))
    }
}

use bagua::kv_store::kv_store_client::KvStoreClient;
use std::io;
use tokio::runtime::Builder;

type StdError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub struct BaguaKvStore {
    client: KvStoreClient<tonic::transport::Channel>,
    rt: tokio::runtime::Runtime,
}

impl BaguaKvStore {
    pub fn open<D>(dst: D) -> Result<Self, io::Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<StdError>,
    {
        let rt = Builder::new_multi_thread().enable_all().build().unwrap();
        let client = rt.block_on(KvStoreClient::connect(dst));
        match client {
            Ok(client) => Ok(Self { rt, client }),
            Err(err) => Err(io::Error::new(io::ErrorKind::Other, format!("{:?}", err))),
        }
    }

    /// Set a value in the database.
    pub fn set(&mut self, key: String, value: impl AsRef<[u8]>) -> Result<(), io::Error> {
        let request = tonic::Request::new(SetRequest {
            key: key,
            value: value.as_ref().to_owned(),
        });
        let rsp = self.rt.block_on(self.client.set(request));
        match rsp {
            Ok(_) => Ok(()),
            Err(err) => Err(io::Error::new(io::ErrorKind::Other, format!("{:?}", err))),
        }
    }

    pub fn get(&mut self, key: String) -> Result<Vec<u8>, io::Error> {
        let request = tonic::Request::new(GetRequest { key: key });
        let rsp = self.rt.block_on(self.client.get(request));
        match rsp {
            Ok(rsp) => Ok(rsp.into_inner().value),
            Err(err) => Err(io::Error::new(io::ErrorKind::Other, format!("{:?}", err))),
        }
    }

    pub fn del(&mut self, key: String) -> Result<(), io::Error> {
        let request = tonic::Request::new(DeleteRequest { key: key });
        let rsp = self.rt.block_on(self.client.delete(request));
        match rsp {
            Ok(_) => Ok(()),
            Err(err) => Err(io::Error::new(io::ErrorKind::Other, format!("{:?}", err))),
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use bagua::kv_store::kv_store_server::KvStoreServer;
    use tokio::runtime::Runtime;

    #[test]
    fn test_kvstore_service() {
        let service = KvStoreService::new();

        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let _ = service
                .set(tonic::Request::new(SetRequest {
                    key: "key1".into(),
                    value: "123".as_bytes().to_vec(),
                }))
                .await;
            let rsp = service
                .get(tonic::Request::new(GetRequest { key: "key1".into() }))
                .await;
            assert_eq!(rsp.unwrap().into_inner().value, "123".as_bytes().to_vec());
            let _ = service
                .delete(tonic::Request::new(DeleteRequest { key: "key1".into() }))
                .await;
            let rsp = service
                .get(tonic::Request::new(GetRequest { key: "key1".into() }))
                .await;
            assert_eq!(rsp.unwrap_err().code(), tonic::Code::InvalidArgument);
        });
    }

    // #[test]
    // fn test_bagua_kv_store() {
    //     let addr = "[::1]:50051";
    //     let service_addr = addr.clone().parse().unwrap();
    //     let kv_store = KvStoreService::new();

    //     let rt = Runtime::new().unwrap();
    //     let service_fut = tonic::transport::Server::builder()
    //         .add_service(KvStoreServer::new(kv_store))
    //         .serve(service_addr);

    //     std::thread::spawn(move || {
    //         let one_sec = std::time::Duration::from_secs(1);
    //         std::thread::sleep(one_sec);

    //         let mut kv = BaguaKvStore::open("http://[::1]:50051").unwrap();
    //         kv.set("key1".into(), "123".as_bytes()).unwrap();
    //         assert_eq!(kv.get("key1".into()).unwrap(), "123".as_bytes().to_vec());
    //         kv.del("key1".into()).unwrap();
    //         kv.del("key1".into()).unwrap_err();
    //         kv.get("key1".into()).unwrap_err();
    //     });

    //     rt.block_on(service_fut)
    //         .expect("failed to successfully run the future on RunTime");
    // }
}
