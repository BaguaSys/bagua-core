use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaTensorRaw};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use std::sync::Arc;

#[derive(Debug)]
pub struct PythonFFIOp {
    pub py_callable: Arc<pyo3::Py<pyo3::PyAny>>,
}

impl CommOpTrait for PythonFFIOp {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        pyo3::Python::with_gil(|gil| {
            let result = self.py_callable.call1(gil, (bucket.name.as_str(),));
            if let Err(e) = result {
                tracing::error!("python ffi op error: {:?}", e);
            }
        });
    }
}
