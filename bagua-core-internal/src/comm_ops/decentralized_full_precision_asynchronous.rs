use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator; 
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::{BaguaCommOpChannels, BaguaCoreError};
use crate::events::BaguaEventChannel;
use std::sync::Arc;
use std::time::Duration;
use cuda_runtime_sys::{cudaStream_t, cudaEvent_t};


#[derive(Debug)]
pub struct DecentralizedFullPrecisionAsynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub torch_stream: u64,
    pub src_ready_event: u64,
    pub dst_ready_event: u64,
}

impl CommOpTrait for DecentralizedFullPrecisionAsynchronous {

    fn execute_background_communication(
            &self,
            bucket: Arc<BaguaBucket>,
            comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket_guard = bucket.inner.lock();
        let comm_stream = self.communicator.stream_ptr();

        let mut communication_tensor = match &self.communicator {
            BaguaCommunicator::SingleCommunicator(_) => {
                bucket_guard.get_communication_tensor(comm_stream, false, false)
            }
            BaguaCommunicator::HierarchicalCommunicator(x) => {
                panic!("asynchronous op only accepts non-hierarchical communicator");
            },
        };

        let peer_mode = &self.peer_selection_mode;

        let torch_stream = self.torch_stream;
        let src_ready_event = self.src_ready_event;
        let dst_ready_event = self.dst_ready_event;

        self.communicator.execute_communication(
                &mut communication_tensor,
                false,
                false,
                false,
                &mut |c, t| {

             let start_time = std::time::Instant::now();

             let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
             .try_pull(t.raw.num_elements_allocated() * t.raw.dtype().bytes())
             .expect("cannot allocate cuda memory");

             let mut temp_tensor = BaguaTensorRaw {
                 ptr: temp_buf.ptr,
                 num_elem_allocated: t.raw.num_elements_allocated(),
                 dtype: t.raw.dtype().clone(),
                 num_elem: t.raw.num_elements(),
                 device_id: t.raw.device_id(),
                 pool_allocations: vec![Arc::new(temp_buf)],
             };

             let reduced_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
             .try_pull(t.raw.num_elements_allocated() * t.raw.dtype().bytes())
             .expect("cannot allocate cuda memory");

             let mut reduced_tensor = BaguaTensorRaw {
                 ptr: reduced_buf.ptr,
                 num_elem_allocated: t.raw.num_elements_allocated(),
                 dtype: t.raw.dtype().clone(),
                 num_elem: t.raw.num_elements(),
                 device_id: t.raw.device_id(),
                 pool_allocations: vec![Arc::new(reduced_buf)],
             };

             // wait the completion of last loop
             unsafe {
                 cpp::cpp!([
                     dst_ready_event as "cudaEvent_t",
                     comm_stream as "cudaStream_t",
                     torch_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaEventRecord(dst_ready_event, comm_stream));
                     CUDACHECK(cudaStreamWaitEvent(torch_stream, dst_ready_event , 0));
                 });
             }

             // use default stream to copy weights
             temp_tensor.clone_from(&t.raw, torch_stream as u64);

             unsafe {
                 cpp::cpp!([
                     src_ready_event as "cudaEvent_t",
                     comm_stream as "cudaStream_t",
                     torch_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaEventRecord(src_ready_event, torch_stream));
                     CUDACHECK(cudaStreamWaitEvent(comm_stream, src_ready_event , 0));
                 });
             }

             c.allreduce(&temp_tensor, &mut reduced_tensor, BaguaReductionOp::SUM);

             // do we need to wait default stream?
             unsafe {
                 cpp::cpp!([
                     src_ready_event as "cudaEvent_t",
                     comm_stream as "cudaStream_t",
                     torch_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaEventRecord(src_ready_event, torch_stream));
                     CUDACHECK(cudaStreamWaitEvent(comm_stream, src_ready_event , 0));
                 });
             }

             t.raw.async_model_average(&reduced_tensor, &temp_tensor, c.nranks as f32, comm_stream);

             unsafe {
                 cpp::cpp!([dst_ready_event as "cudaEvent_t", comm_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaEventRecord(dst_ready_event, comm_stream));
                     CUDACHECK(cudaStreamSynchronize(comm_stream));
                 });
             }

             tracing::debug!("async model average update cost: {:?}", start_time.elapsed());

        });
    }
}
