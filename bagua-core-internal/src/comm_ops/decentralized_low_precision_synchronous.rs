use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{BaguaBucket, BaguaTensorRaw, RawBaguaTensor, TensorCompressionMethod};
use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
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
    pub my_tensor: BaguaTensorRaw,
    pub left_peer_tensor: BaguaTensorRaw,
    pub right_peer_tensor: BaguaTensorRaw,
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

        let peer_mode = &self.peer_selection_mode;

        let left_peer_tensor = &self.left_peer_tensor;
        let right_peer_tensor = &self.left_peer_tensor;

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {
                tracing::debug!("start compress diff");
                t.raw.substract_inplace(&self.my_tensor, c.stream_ptr);
                let compressed_tensor = t
                    .raw
                    .compress(&self.compression_method, c.nranks, c.stream_ptr, -1)
                    .expect("cannot compress tensor");

                tracing::debug!("start communication with peers");
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
                            let left_peer_rank = ((c.rank - 1 + c.nranks) % c.nranks) as i32;
                            let right_peer_rank = ((c.rank + 1)  % c.nranks) as i32;

                            {
                                let _guard = NCCLGroupGuard::new();
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

                tracing::debug!("start decompress");
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &lrecv_tensor,
                    c.stream_ptr,
                );
                
                // FIXME
                /* left_peer_tensor.add_inplace(&t.raw, c.stream_ptr);
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &rrecv_tensor,
                    c.stream_ptr,
                );
                right_peer_tensor.add_inplace(&t.raw, c.stream_ptr);

                t.raw.decompress_from(
                     &self.compression_method,
                     c.nranks,
                     &compressed_tensor,
                     c.stream_ptr,
                );
                t.raw.add_inplace(&self.my_tensor, c.stream_ptr);
                */
            },
        );
    }
}

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronousAverageStep {
    pub communicator: BaguaCommunicator,
    pub left_peer_weight: BaguaTensorRaw,
    pub right_peer_weight: BaguaTensorRaw,
}

impl CommOpTrait for DecentralizedLowPrecisionSynchronousAverageStep {
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
                t.raw.add_inplace(&self.left_peer_weight, c.stream_ptr);
                t.raw.add_inplace(&self.right_peer_weight, c.stream_ptr);
                t.raw.divide_inplace(c.stream_ptr, 3 as f32);
            },
        );
    }
}
