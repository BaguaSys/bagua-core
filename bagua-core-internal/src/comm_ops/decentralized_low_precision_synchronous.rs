use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{BaguaBucket, BaguaTensorRaw, RawBaguaTensor, TensorCompressionMethod};
use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::events::BaguaEventChannel;
use crate::{BaguaCommOpChannels, BaguaScheduledCommOp};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronousPostStep {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub compression_method: TensorCompressionMethod,
    pub step: usize,
}

impl CommOpTrait for DecentralizedLowPrecisionSynchronousPostStep {

    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket_guard = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();

        let mut communication_tensor = bucket_guard.get_communication_tensor(stream_ptr, false, false);

        let weight = bucket_guard.get_state_tensor("weight");
        let mut left_peer_weight = bucket_guard.get_state_tensor("left_peer_weight");
        let mut right_peer_weight = bucket_guard.get_state_tensor("right_peer_weight");
       
        let t = &communication_tensor;
        let peer_mode = &self.peer_selection_mode;

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {
                tracing::debug!("start compress diff");
                t.raw.substract_inplace(&weight, c.stream_ptr);
                let compressed_tensor = t
                    .raw
                    .compress(&self.compression_method, 1, c.stream_ptr, -1)
                    .expect("cannot compress tensor");

                tracing::debug!("start communicate with peers");
                let lrecv_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
                    .try_pull(
                        compressed_tensor.num_elements_allocated()
                            * compressed_tensor.dtype().bytes(),
                    )
                    .expect("cannot allocate cuda memory");
                let mut lrecv_tensor = BaguaTensorRaw {
                    ptr: lrecv_buf.ptr,
                    num_elem_allocated: compressed_tensor.num_elements_allocated(),
                    dtype: compressed_tensor.dtype().clone(),
                    num_elem: compressed_tensor.num_elements(),
                    device_id: compressed_tensor.device_id(),
                    pool_allocations: vec![Arc::new(lrecv_buf)],
                };
                
                let rrecv_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
                    .try_pull(
                        compressed_tensor.num_elements_allocated()
                            * compressed_tensor.dtype().bytes(),
                    )
                    .expect("cannot allocate cuda memory");
                let mut rrecv_tensor = BaguaTensorRaw {
                    ptr: rrecv_buf.ptr,
                    num_elem_allocated: compressed_tensor.num_elements_allocated(),
                    dtype: compressed_tensor.dtype().clone(),
                    num_elem: compressed_tensor.num_elements(),
                    device_id: compressed_tensor.device_id(),
                    pool_allocations: vec![Arc::new(rrecv_buf)],
                };

                match peer_mode {
                    PeerSelectionMode::Ring => {
                        {
                            let left_peer_rank = ((c.rank + c.nranks - 1) % c.nranks) as i32;
                            let right_peer_rank = ((c.rank + 1)  % c.nranks) as i32;

                            {
                                let _guard = NCCLGroupGuard::new();

                                tracing::debug!(
                                    "rank: {} left peer: {} right peer: {}", 
                                    c.rank, left_peer_rank, right_peer_rank)
                                ;
                                c.send(compressed_tensor.as_ref(), left_peer_rank);
                                c.send(compressed_tensor.as_ref(), right_peer_rank);
                                c.recv(&mut lrecv_tensor, left_peer_rank);
                                c.recv(&mut rrecv_tensor, right_peer_rank);
                            }
                        }
                    },
                    PeerSelectionMode::All => {
                        unimplemented!()
                    },
                    PeerSelectionMode::ShiftOne => {
                        unimplemented!()
                    }
                };

                tracing::debug!("start decompress diff and update weights");
                t.raw.decompress_from(
                    &self.compression_method,
                    1,
                    &lrecv_tensor,
                    c.stream_ptr,
                ); 
                left_peer_weight.add_inplace(&t.raw, c.stream_ptr);

                t.raw.decompress_from(
                    &self.compression_method,
                    1,
                    &rrecv_tensor,
                    c.stream_ptr,
                );
                right_peer_weight.add_inplace(&t.raw, c.stream_ptr);

                t.raw.decompress_from(
                     &self.compression_method,
                     1,
                     compressed_tensor.as_ref(),
                     c.stream_ptr,
                );

                t.raw.add_inplace(&weight, c.stream_ptr);
            },
        );
    }
}

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub step: Mutex<usize>,
    pub communication_interval: usize,
    pub compression_method: TensorCompressionMethod,
}

impl CommOpTrait for DecentralizedLowPrecisionSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket_guard = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = match &self.communicator {
            BaguaCommunicator::SingleCommunicator(_) => {
                bucket_guard.get_communication_tensor(stream_ptr, false, false)
            }
            BaguaCommunicator::HierarchicalCommunicator(x) => match x {
                BaguaHierarchicalCommunicator::Leader(_) => {
                    bucket_guard.get_communication_tensor(stream_ptr, true, true)
                }
                BaguaHierarchicalCommunicator::Worker(_) => {
                    bucket_guard.get_communication_tensor(stream_ptr, false, false)
                }
            },
        };

        let mut weight = bucket_guard.get_state_tensor("weight");
        let left_peer_weight = bucket_guard.get_state_tensor("left_peer_weight");
        let right_peer_weight = bucket_guard.get_state_tensor("right_peer_weight");

        let peer_mode = &self.peer_selection_mode;
        let comm_interval = &self.communication_interval;
        let step = { *self.step.lock() };

        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            false,
            true,
            &mut |c, t| {
                if step % comm_interval == 0 {
                    tracing::debug!("start compute averaged weight");
                    t.raw.add_inplace(&left_peer_weight, c.stream_ptr);
                    t.raw.add_inplace(&right_peer_weight, c.stream_ptr);
                    t.raw.divide_inplace(c.stream_ptr, 3 as f32);

                    weight.clone_from(&t.raw, c.stream_ptr);

                }
            },
        );

        if step % comm_interval == 0 {
            let post_optimizer_step_comm_op = BaguaScheduledCommOp {
                name: format!("post optimizer step comm op for bucket {}", bucket.name),
                bucket: bucket.clone(),
                ops: vec![Arc::new(DecentralizedLowPrecisionSynchronousPostStep {
                    communicator: self.communicator.clone(),
                    peer_selection_mode: self.peer_selection_mode.clone(),
                    compression_method: self.compression_method.clone(),
                    step,
                })],
                event_channel: BaguaEventChannel::new("low_precision_decentralized_post_optimizer_step"),
            };

            comm_op_channels
                .not_waited_post_optimizer_step_events_sender
                .send(post_optimizer_step_comm_op.event_channel.clone())
                .expect("cannot send post optimizer step event");
            comm_op_channels
                .post_optimizer_step_channel_sender
                .send(post_optimizer_step_comm_op)
                .expect("cannot send post optimizer step op");
        }
        *self.step.lock() += 1;
    }
}
