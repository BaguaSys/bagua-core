pub unsafe fn cuda_memcpy_device_to_host_sync(host_ptr: u64, device_ptr: u64, num_bytes: i32) {
    cpp::cpp!([host_ptr as "void*", device_ptr as "void*", num_bytes as "int"]
    {
        CUDACHECK(cudaMemcpy(host_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost));
    });
}

pub unsafe fn cuda_memcpy_host_to_device_sync(device_ptr: u64, host_ptr: u64, num_bytes: i32) {
    cpp::cpp!([host_ptr as "void*", device_ptr as "void*", num_bytes as "int"]
    {
        CUDACHECK(cudaMemcpy(device_ptr, host_ptr, num_bytes, cudaMemcpyHostToDevice));
    });
}

pub unsafe fn cuda_set_device(device_id: u64) {
    cpp::cpp!([device_id as "size_t"]
    {
        CUDACHECK(cudaSetDevice(device_id));
    });
}
