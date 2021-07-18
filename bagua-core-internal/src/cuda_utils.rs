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

pub unsafe fn cuda_memcpy_D2D_async(
    dst_ptr: u64,
    src_ptr: u64,
    bytes: usize,
    cuda_stream_ptr: u64,
    ready_cuda_event_ptr: u64,
) -> u64 {
    cpp::cpp!([dst_ptr as "void*", src_ptr as "void*", bytes as "size_t",
        cuda_stream_ptr as "cudaStream_t", ready_cuda_event_ptr as "cudaEvent_t"] -> u64 as "cudaEvent_t"
    {
        if (ready_cuda_event_ptr != 0) {
            CUDACHECK(cudaStreamWaitEvent(cuda_stream_ptr, ready_cuda_event_ptr , 0));
        }
        CUDACHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, cuda_stream_ptr));
        cudaEvent_t cuda_event = NULL;
        CUDACHECK(cudaEventCreateWithFlags(
            &cuda_event, cudaEventBlockingSync | cudaEventDisableTiming));
        CUDACHECK(cudaEventRecord(cuda_event, cuda_stream_ptr));

        return cuda_event;
    })
}

pub unsafe fn cuda_set_device(device_id: u64) {
    cpp::cpp!([device_id as "size_t"]
    {
        CUDACHECK(cudaSetDevice(device_id));
    });
}
