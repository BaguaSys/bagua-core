fn main() {
    tonic_build::compile_protos("proto/bagua/kv_store.proto").unwrap();
}