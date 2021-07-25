use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::comm_ops::AsyncCommOpTrait;
use crate::communicators::BaguaCommunicator; 
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor, BaguaExecutionHandle};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::{BaguaCommOpChannels, BaguaCoreError};
use crate::events::BaguaEventChannel;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use cuda_runtime_sys::{cudaStream_t, cudaEvent_t};


#[derive(Debug)]
pub struct DecentralizedFullPrecisionAsynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub sync_interval_ms: u64,
}

impl AsyncCommOpTrait for DecentralizedFullPrecisionAsynchronous {

    fn execute_background_communication_async(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
        _event_channel: &BaguaEventChannel,
    ) -> BaguaExecutionHandle {
        let bucket = bucket.inner.clone();

        let c = match &self.communicator {
            BaguaCommunicator::SingleCommunicator(communicator) => {
                communicator.inner.clone()
            },
            BaguaCommunicator::HierarchicalCommunicator(communicator) => {
                panic!("asynchronous op only accepts non-hierarchical communicator");
            },
        };

        let peer_selection_mode = self.peer_selection_mode.clone();
        let sync_interval_ms = self.sync_interval_ms;
        let device_id = self.communicator.device_id();
        let stream_ptr = self.communicator.stream_ptr();

        BaguaExecutionHandle {
            inner: thread::spawn(move || {

                let mut dst_ready_event = std::ptr::null_mut() as cudaEvent_t;
                let mut src_ready_event = std::ptr::null_mut() as cudaEvent_t;

                let dst_ready_event_ptr = &mut dst_ready_event;
                let src_ready_event_ptr = &mut src_ready_event;

                unsafe {
                    cpp::cpp!([device_id as "size_t"]
                    { CUDACHECK(cudaSetDevice(device_id)); });

                    cpp::cpp!([dst_ready_event_ptr as "cudaEvent_t *"]
                    { CUDACHECK(cudaEventCreate(dst_ready_event_ptr)); });

                    cpp::cpp!([src_ready_event_ptr as "cudaEvent_t *"]
                    { CUDACHECK(cudaEventCreate(src_ready_event_ptr)); });
                }

                let torch_stream = std::ptr::null_mut() as cudaStream_t;
                loop {
                    {
                        tracing::debug!("async model average start");
                        let bucket_guard = bucket.lock();
                        let mut communication_tensor = bucket_guard.get_communication_tensor(stream_ptr, false, false);
                        let t = &communication_tensor.raw;

                        let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.device_id]
                        .try_pull(t.num_elements_allocated() * t.dtype().bytes())
                        .expect("cannot allocate cuda memory");

                        let mut temp_tensor = BaguaTensorRaw {
                            ptr: temp_buf.ptr,
                            num_elem_allocated: t.num_elements_allocated(),
                            dtype: t.dtype().clone(),
                            num_elem: t.num_elements(),
                            device_id: t.device_id(),
                            pool_allocations: vec![Arc::new(temp_buf)],
                        };

                        unsafe {
                            cpp::cpp!([dst_ready_event as "cudaEvent_t", stream_ptr as "cudaStream_t", torch_stream as "cudaStream_t"]
                            {
                                CUDACHECK(cudaEventRecord(dst_ready_event, stream_ptr));
                                CUDACHECK(cudaStreamWaitEvent(torch_stream, dst_ready_event , 0));
                            });
                        }

                        temp_tensor.clone_from(t, torch_stream as u64);

                        unsafe {
                            cpp::cpp!([src_ready_event as "cudaEvent_t", stream_ptr as "cudaStream_t", torch_stream as "cudaStream_t"]
                            {
                                CUDACHECK(cudaEventRecord(src_ready_event, torch_stream));
                                CUDACHECK(cudaStreamWaitEvent(stream_ptr, src_ready_event , 0));
                            });
                        }

                        c.allreduce_inplace(&mut temp_tensor, BaguaReductionOp::SUM);
                        temp_tensor.divide_inplace(stream_ptr, c.nranks as f32);

                        unsafe {
                            cpp::cpp!([dst_ready_event as "cudaEvent_t", stream_ptr as "cudaStream_t", torch_stream as "cudaStream_t"]
                            {
                                CUDACHECK(cudaEventRecord(dst_ready_event, stream_ptr));
                                CUDACHECK(cudaStreamWaitEvent(torch_stream, dst_ready_event , 0));
                            });
                        }

                        tracing::debug!("async model average end");
                    }

                    thread::sleep(Duration::from_millis(sync_interval_ms));
                }
            }),
        }

    }
}
