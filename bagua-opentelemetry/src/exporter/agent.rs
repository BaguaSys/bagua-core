use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
pub struct BaguaSpan {
    pub trace_id: u128,
    pub action: String,
    pub tensor_name: String,
    pub start_time: u128,
    pub end_time: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
pub struct BaguaBatch {
    pub spans: Vec<BaguaSpan>,
}

#[derive(Serialize, Deserialize)]
pub struct EmitBatchResponse {}

#[derive(Debug)]
pub struct AgentAsyncClientHTTP {
    server_addr: String,
}

impl AgentAsyncClientHTTP {
    pub fn new(server_addr: String) -> AgentAsyncClientHTTP {
        Self {
            server_addr: server_addr,
        }
    }

    pub async fn emit_batch(
        &mut self,
        batch: BaguaBatch,
    ) -> Result<EmitBatchResponse, reqwest::Error> {
        let uri = format!("http://{}/report_tensor_execution_order", self.server_addr);

        let new_post = reqwest::Client::new()
            .post(uri)
            .json(&batch)
            .send()
            .await?
            .json()
            .await?;

        Ok(new_post)
    }
}
