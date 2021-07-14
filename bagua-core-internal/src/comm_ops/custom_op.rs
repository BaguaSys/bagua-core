use crate::comm_ops::CommOpTrait;
use crate::datatypes::BaguaBucket;
use crate::BaguaCommOpChannels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CustomOp {
    pub callable: Arc<dyn Fn(Arc<BaguaBucket>, &BaguaCommOpChannels) -> ()>,
}

impl CommOpTrait for CustomOp {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_op_channels: &BaguaCommOpChannels,
    ) {
        self.callable(bucket, comm_op_channels);
    }
}
