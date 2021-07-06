use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{BaguaBucket, BaguaTensor, BaguaTensorRaw, RawBaguaTensor, TensorCompressionMethod};
use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::events::BaguaEventChannel;
use crate::{BaguaCommOpChannels, BaguaScheduledCommOp};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub struct DecentralizedLowPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub step: Mutex<usize>,
    pub communication_interval: usize,
    pub compression_method: TensorCompressionMethod,
    pub weight: BaguaTensor,
    pub left_peer_weight: BaguaTensor,
    pub right_peer_weight: BaguaTensor,
}

impl CommOpTrait for DecentralizedLowPrecisionSynchronous {

    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
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

        let mut weight = BaguaTensorRaw {
            ptr: self.weight.data_ptr(),
            num_elem: self.weight.num_elements(),
            dtype: self.weight.inner.read().raw.dtype(),
            num_elem_allocated: self.weight.num_elements_allocated(),
            device_id: self.weight.device_id(),
            pool_allocations: vec![],
        };
        let mut left_peer_weight = BaguaTensorRaw {
            ptr: self.left_peer_weight.data_ptr(),
            num_elem: self.left_peer_weight.num_elements(),
            dtype: self.left_peer_weight.inner.read().raw.dtype(),
            num_elem_allocated: self.left_peer_weight.num_elements_allocated(),
            device_id: self.left_peer_weight.device_id(),
            pool_allocations: vec![],
        };
        let mut right_peer_weight = BaguaTensorRaw {
            ptr: self.right_peer_weight.data_ptr(),
            num_elem: self.right_peer_weight.num_elements(),
            dtype: self.right_peer_weight.inner.read().raw.dtype(),
            num_elem_allocated: self.right_peer_weight.num_elements_allocated(),
            device_id: self.right_peer_weight.device_id(),
            pool_allocations: vec![],
        };

        let peer_mode = &self.peer_selection_mode;
        let comm_interval = &self.communication_interval;
        let step = { *self.step.lock() };

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {
                if step % comm_interval == 0 {
                    tracing::debug!("start compress diff");

                    t.raw.addmul_inplace(&left_peer_weight, 1.0 / 3.0, c.stream_ptr);
                    t.raw.addmul_inplace(&right_peer_weight, 1.0 / 3.0, c.stream_ptr);
                    t.raw.addmul_inplace(&weight, -2.0 / 3.0, c.stream_ptr);

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

                    weight.clone_from(&t.raw, c.stream_ptr);
                }
            },
        );
        
        *self.step.lock() += 1;
    }
}
