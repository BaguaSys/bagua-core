use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{BaguaBucket, BaguaTensorRaw};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::{BaguaCommOpChannels, BaguaScheduledCommOp};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub step: Mutex<usize>,
    pub communication_interval: usize,
    pub my_weight: BaguaTensorRaw,
    pub peer_weight: BaguaTensorRaw,
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

        let step = { *self.step.lock() };

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {

                t.raw.substract_inplace(&self.my_weight, c.stream_ptr);
                tracing::debug!("start compress");
                let compressed_tensor = t
                    .raw
                    .compress(&self.compression_method, c.nranks, c.stream_ptr, -1)
                    .expect("cannot compress tensor");

                let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
                    .try_pull(
                        compressed_tensor.num_elem_allocated * compressed_tensor.dtype.bytes(),
                    )
                    .expect("cannot allocate cuda memory");
                let mut temp_tensor = BaguaTensorRaw {
                    ptr: temp_buf.ptr,
                    num_elem_allocated: compressed_tensor.num_elem_allocated,
                    dtype: compressed_tensor.dtype.clone(),
                    num_elem: compressed_tensor.num_elem,
                    device_id: compressed_tensor.device_id,
                    pool_allocation: Some(temp_buf),
                };

                tracing::debug!("start communication with peers");
                match peer_mode {
                    PeerSelectionMode::ShiftOne => {
                        if step % comm_interval == 0 {
                            assert_eq!(
                                c.nranks % 2,
                                0,
                                "you cannot use decentralized algorithm with average_all off when there are odd number of ranks, current n_ranks {}",
                                c.nranks
                            );
                            let comm_step = step / comm_interval;
                            let peer_rank = if c.rank < c.nranks / 2 {
                                ((comm_step + c.rank) % ((c.nranks + 1) / 2)) + (c.nranks / 2)
                            } else {
                                (c.rank - (c.nranks / 2) - comm_step).rem_euclid(c.nranks / 2)
                            } as i32;
                            tracing::debug!("rank {} peer_rank {}", c.rank, peer_rank);
                            {
                                let _guard = NCCLGroupGuard::new();
                                c.send(&compressed_tensor, peer_rank);
                                c.recv(&mut temp_tensor, peer_rank);
                            }
                        }
                    }
                }

                tracing::debug!("start decompress");
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &compressed_tensor,
                    c.stream_ptr,
                );
                self.my_weight.add_inplace(&t.raw, c.stream_ptr);
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &temp_tensor,
                    c.stream_ptr,
                );
                self.peer_weight.add_inplace(&t.raw, c.stream_ptr);
            },
        );

        if step % comm_interval == 0 {
            // TODO: move this to .then() python API instead of hard code this in op
            let post_backward_comm_op = BaguaScheduledCommOp {
                bucket: bucket.clone(),
                ops: vec![Arc::new(DecentralizedLowPrecisionSynchronousPostStep {
                    communicator: self.communicator.clone(),
                    my_weight: self.my_weight,
                    peer_weight: self.peer_weight,
                })],
                event_channel: Default::default(),
            };

            comm_op_channels
                .not_waited_post_backward_events_sender
                .send(post_backward_comm_op.event_channel.clone())
                .expect("cannot send post backward event");
            comm_op_channels
                .post_backward_channel_sender
                .send(post_backward_comm_op)
                .expect("cannot send post backward op");
        }

        *self.step.lock() += 1;
    }
}

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronousPostStep {
    pub communicator: BaguaCommunicator,
    pub my_weight: BaguaTensorRaw,
    pub peer_weight: BaguaTensorRaw,
}

impl CommOpTrait for DecentralizedLowPrecisionSynchronousPostStep {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = bucket.get_communication_tensor(stream_ptr, false, false);
        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            false,
            true,
            &mut |c, t| {
                // calculate averaged weight
                t.raw.clone_from(&self.my_weight, c.stream_ptr);
                t.raw.average_inplace(&self.peer_weight, c.stream_ptr);
            },
        );
    }
}
