use crate::comm_ops::CommOpTrait;
use crate::datatypes::BaguaBucket;
use crate::BaguaCommOpChannels;
use std::sync::Arc;
use std::fmt;

#[derive(Debug)]
pub struct CustomOp {
    pub callable: Arc<dyn Fn(Arc<BaguaBucket>, &BaguaCommOpChannels) -> ()>,
}

impl fmt::Debug for CustomOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomOp")
            .field("callable", format!("{:?}", self.callable))
            .finish()
    }
}

impl CommOpTrait for CustomOp {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_op_channels: &BaguaCommOpChannels,
    ) {
        (self.callable)(bucket, comm_op_channels);
    }
}
