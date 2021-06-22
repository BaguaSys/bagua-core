extern crate serde;
extern crate serde_json;

use crate::BaguaCoreError;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use reqwest;
use scheduled_thread_pool::ScheduledThreadPool;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[allow(dead_code)]
pub static SCHEDULED_THREAD_POOL: Lazy<ScheduledThreadPool> =
    Lazy::new(|| ScheduledThreadPool::with_name("bagua_scheduled_thread_pool", 1));

pub static TELEMETRY: Lazy<Option<Mutex<BaguaCommCoreTelemetry>>> =
    Lazy::new(|| match std::env::var("AUTO_TUNE_SERVER_ADDR") {
        Ok(x) => {
            tracing::info!("detected auto tuning server, connecting");
            Some(Mutex::new(BaguaCommCoreTelemetry::new(x.as_str())))
        }
        Err(_) => {
            tracing::warn!("auto tuning server not detected, may experience degraded performance");
            None
        }
    });

pub struct BaguaCommCoreTelemetry {
    client: reqwest::blocking::Client,
    server_addr: String,
    current_payload: TelemetryPayload,
    pub recent_speed: RecentMeanMetric,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TelemetryPayload {
    tensor_ready_order: Vec<u64>,
    communication_time_ms: u64,
}

impl TelemetryPayload {
    pub fn clear(&mut self) {
        self.tensor_ready_order.clear();
        self.communication_time_ms = 0;
    }
}

impl Default for TelemetryPayload {
    fn default() -> Self {
        Self {
            tensor_ready_order: vec![],
            communication_time_ms: 0,
        }
    }
}

impl BaguaCommCoreTelemetry {
    pub fn new(server_addr: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap();

        Self {
            client: client,
            server_addr: server_addr.to_string(),
            current_payload: TelemetryPayload::default(),
            recent_speed: RecentMeanMetric::new(),
        }
    }

    pub fn new_tensor_ready(&mut self, tensor_id: u64) {
        self.current_payload.tensor_ready_order.push(tensor_id);
    }

    pub fn push_payload_and_clear(&mut self) -> Result<(), BaguaCoreError> {
        if self
            .client
            .post(format!("http://{}/api/v1/bagua_backend_metrics", self.server_addr).as_str())
            .json(&self.current_payload)
            .send()
            .unwrap()
            .status()
            != 200
        {
            return Err(BaguaCoreError::TelemetryError(
                "post TELEMETRY payload failed".into(),
            ));
        }
        self.clear();
        Ok(())
    }

    pub fn register_models(
        &self,
        req: RegisterModelsRequest,
    ) -> Result<RegisterModelsResponse, BaguaCoreError> {
        let rsp: RegisterModelsResponse = self
            .client
            .post(format!("http://{}/api/v1/register_models", self.server_addr).as_str())
            .json(&req)
            .send()
            .unwrap()
            .json()
            .unwrap();

        Ok(rsp)
    }

    pub fn report_metrics(&self, req: ReportMetricsRequest) -> Result<(), BaguaCoreError> {
        let rsp = self
            .client
            .post(format!("{}/api/v1/report_metrics", self.server_addr).as_str())
            .json(&req)
            .send()
            .unwrap();
        if rsp.status() != 200 {
            return Err(BaguaCoreError::TelemetryError(format!(
                "register_models failed, rsp={:?}",
                rsp
            )));
        }

        Ok(())
    }

    pub fn ask_hyperparameters(
        &self,
        req: AskHyperparametersRequest,
    ) -> Result<AskHyperparametersResponse, BaguaCoreError> {
        let rsp: AskHyperparametersResponse = self
            .client
            .post(format!("{}/api/v1/ask_hyperparameters", self.server_addr).as_str())
            .json(&req)
            .send()
            .unwrap()
            .json()
            .unwrap();

        Ok(rsp)
    }

    pub fn clear(&mut self) {
        self.current_payload.clear();
    }
}

#[derive(Debug)]
pub struct RecentMeanMetric {
    history_base_on: std::time::Instant,
    records: Vec<f64>,
    tail: Option<(f64, f64)>,
}

impl RecentMeanMetric {
    pub fn new() -> RecentMeanMetric {
        RecentMeanMetric {
            history_base_on: Instant::now(),
            records: Default::default(),
            tail: None,
        }
    }

    fn get_records_mean(&self, last_x_seconds: f64) -> f64 {
        if approx_eq!(f64, last_x_seconds, 0., ulps = 2) {
            return 0.;
        }

        let records_seconds: f64 = if self.records.len() != 0 {
            (2 as f64).powf((self.records.len() - 1) as f64)
        } else {
            0.
        };

        let tail_seconds: f64 = match self.tail {
            Some((seconds, _)) => seconds,
            _ => 0.,
        };

        let tail_mean: f64 = match self.tail {
            Some((_, val)) => val,
            _ => 0.,
        };

        // NO records part
        if self.records.len() == 0 {
            return tail_mean;
        }

        if last_x_seconds < 1.0 {
            return self.records[0];
        }

        return if last_x_seconds <= records_seconds {
            let floor_id = std::cmp::max(last_x_seconds.log(2.0).floor() as usize, 0);
            let floor_time = (2 as f64).powf(floor_id as f64);
            let mean: f64 = if floor_id + 1 < self.records.len() {
                let (a, b) = (self.records[floor_id], self.records[floor_id + 1]);
                let (a_l, b_l) = (floor_time, 2. * floor_time);
                a + (b - a) * (last_x_seconds - a_l) / (b_l - a_l)
            } else {
                self.records[floor_id]
            };

            mean
        } else if last_x_seconds <= records_seconds + tail_seconds {
            let (a, b) = (self.records[self.records.len() - 1], tail_mean);
            let (a_l, b_l) = (records_seconds, records_seconds + tail_seconds);
            a + (b - a) * (last_x_seconds - a_l) / (b_l - a_l)
        } else {
            tail_mean
        };
    }

    pub fn total_recording_time(&self) -> f64 {
        let records_seconds: f64 = if self.records.len() != 0 {
            (2 as f64).powf((self.records.len() - 1) as f64)
        } else {
            0.
        };

        let tail_seconds = match self.tail {
            Some((seconds, _)) => seconds,
            _ => 0.,
        };

        records_seconds + tail_seconds
    }

    pub fn record(&mut self, val: f64) {
        let now = Instant::now();
        let time_dist = now.duration_since(self.history_base_on).as_secs_f64();
        let mut new_records: Vec<f64> = Default::default();
        let mut new_tail: Option<(f64, f64)> = None;

        for i in 0..64 {
            let coverage_period = (2 as i64).pow(i as u32) as f64;

            if coverage_period <= time_dist {
                new_records.push(val);
            } else if coverage_period <= time_dist + self.total_recording_time() {
                let record_contribution_percentage = time_dist / coverage_period;
                let new_val = val * record_contribution_percentage
                    + self.get_records_mean(coverage_period - time_dist)
                        * (1. - record_contribution_percentage);

                new_records.push(new_val);

                if approx_eq!(
                    f64,
                    coverage_period,
                    time_dist + self.total_recording_time(),
                    ulps = 2
                ) {
                    break;
                }
            } else {
                let new_total_time = time_dist + self.total_recording_time();
                let report_contribution_percentage = time_dist / new_total_time;

                let tail_len = (time_dist + self.total_recording_time()) - coverage_period / 2.;
                let tail_val = val * report_contribution_percentage
                    + self.get_records_mean(self.total_recording_time())
                        * (1. - report_contribution_percentage);
                new_tail = Some((tail_len, tail_val));
                break;
            }
        }

        self.history_base_on = now;
        self.records = new_records;
        self.tail = new_tail;
    }

    pub fn get(&self, x_seconds: f64) -> f64 {
        let time_dist = Instant::now()
            .duration_since(self.history_base_on)
            .as_secs_f64();

        if x_seconds <= time_dist {
            return self.records[0];
        }

        self.get_records_mean(x_seconds - time_dist)
    }

    pub fn debug(&self) {
        println!("{:?}", self);
        let report_list = vec![1, 2, 10, 60, 120, 60 * 60];
        for x in report_list.iter() {
            println!("In recent {} seconds, mean val={}", x, self.get(*x as f64));
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_end_to_end() {
        let mut m = RecentMeanMetric {
            history_base_on: Instant::now(),
            records: vec![5.0, 4.5005175, 3.5034241850204078],
            tail: Some((2.0166309999999985, 2.499061185570463)),
        };
        std::thread::sleep(Duration::from_secs(1));

        m.record(6.);

        assert!(approx_eq!(f64, m.get_records_mean(1.), 6., ulps = 2));
        assert!(approx_eq!(
            f64,
            m.get_records_mean(2.),
            (6. + 5.) / 2.,
            ulps = 2,
            epsilon = 0.1
        ));
        assert!(approx_eq!(
            f64,
            m.get_records_mean(3.),
            (6. + 4.5005175 * 2.) / 3.,
            (0.1, 2)
        ));
        assert!(approx_eq!(
            f64,
            m.get_records_mean(5.),
            (6. + 3.5034241850204078 * 4.) / 5.,
            (0.1, 2)
        ));
        assert!(approx_eq!(
            f64,
            m.get_records_mean(7.),
            (6. + 2.499061185570463 * 6.) / 7.,
            (0.1, 2)
        ));
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorDeclaration {
    pub name: String,
    pub num_elements: usize,
    pub dtype: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterModelsRequest {
    pub tensor_list: Vec<TensorDeclaration>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BaguaHyperparameters {
    pub buckets: Vec<Vec<TensorDeclaration>>,
    pub is_hierarchical_reduce: bool,
    pub distributed_algorithm: String,
}

#[derive(Deserialize, Debug)]
pub struct RegisterModelsResponse {
    pub recommended_hyperparameters: BaguaHyperparameters,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReportMetricsRequest {
    pub rank: i32,
    pub train_iter: i32,
    pub hyperparameters: BaguaHyperparameters,
    pub speed: f64,
    pub distributed_algorithm: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AskHyperparametersRequest {
    pub rank: i32,
    pub train_iter: i32,
}

#[derive(Deserialize, Debug)]
pub struct AskHyperparametersResponse {
    pub recommended_hyperparameters: BaguaHyperparameters,
    pub recommended_from_iter: i32,
    pub is_autotune_processing: bool,
}
