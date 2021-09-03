use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor, BaguaTensor};
use crate::events::BaguaEventChannel;
use crate::resource_pool::{CUDA_DEVICE_MEMORY_POOL, CUDA_EVENT_POOL};
use crate::{BaguaCommOpChannels, BaguaCoreError};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub struct DecentralizedFullPrecisionAsynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub torch_stream: u64,
    pub weight: BaguaTensor,
    pub diff_tensor: BaguaTensor,
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
            }
        };

        let peer_mode = &self.peer_selection_mode;

        let torch_stream = self.torch_stream;

        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            false,
            false,
            &mut |c, t| {
                let start_time = std::time::Instant::now();
                tracing::debug!("#{} async model average start", c.rank);

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

                let src_ready_event = CUDA_EVENT_POOL.take().event;
                let dst_ready_event = CUDA_EVENT_POOL.take().event;
                
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

      /*          if c.check_abort() {
                    return;
                }
*/
                match peer_mode {
                    PeerSelectionMode::All => {
                        c.allreduce(&temp_tensor, &mut reduced_tensor, BaguaReductionOp::SUM);
                    }
                    PeerSelectionMode::Ring => {
                        unimplemented!()
                    }
                    PeerSelectionMode::ShiftOne => {
                        unimplemented!()
                    }
                };

                { 
                    let mut tensor_guard = self.diff_tensor.inner.write();
                    tensor_guard.raw.async_model_average(
                        &reduced_tensor,
                        &temp_tensor,
                        c.nranks as f32,
                        comm_stream,
                    );
                }

                tracing::debug!(
                    "#{} async model average update cost: {:?}",
                    c.rank, start_time.elapsed()
                );
            },
        );
    }
    
    fn execute_post_step(
        &self,
        bucket: Arc<BaguaBucket>
    ) {
        
        tracing::debug!("async update weight start");
        let torch_stream = self.torch_stream;
        
        let mut tensor_guard = self.diff_tensor.inner.write();
        let mut guard = self.weight.inner.write();
         
        guard.raw.async_model_update(tensor_guard.raw.as_ref(), torch_stream);
        
        tracing::debug!("async update weight end");
        
    }

}
