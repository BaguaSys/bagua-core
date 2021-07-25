pub mod centralized_full_precision_synchronous;
pub mod centralized_low_precision_synchronous;
pub mod decentralized_full_precision_synchronous;
pub mod decentralized_low_precision_synchronous;
pub mod decentralized_full_precision_asynchronous;
pub mod python_ffi_op;

use crate::datatypes::{BaguaBucket, BaguaExecutionHandle};
use crate::{BaguaCommOpChannels, BaguaCoreError};
use crate::events::BaguaEventChannel;
use std::fmt::Debug;
use std::sync::Arc;

pub trait CommOpTrait: Debug {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_channels: &BaguaCommOpChannels,
    );
}

pub trait AsyncCommOpTrait: Debug {
    fn execute_background_communication_async(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_channels: &BaguaCommOpChannels,
        event_channel: &BaguaEventChannel,
    ) -> BaguaExecutionHandle;
}
