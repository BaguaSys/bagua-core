pub mod exporter;
pub mod runtime;

use crate::exporter::agent::AgentAsyncClientHTTP;
use crate::exporter::Exporter;
use crate::runtime::BaguaTraceRuntime;
use opentelemetry;
use opentelemetry::{
    global, sdk,
    sdk::trace::{TraceRuntime, TrySend},
    trace::Tracer,
    trace::TracerProvider,
};

pub fn init_tracer<R: TraceRuntime>(autotune_server_addr: &str) -> impl Tracer {
    let exporter = Exporter {
        uploader: AgentAsyncClientHTTP::new(autotune_server_addr.to_string()),
    };

    let builder =
        sdk::trace::TracerProvider::builder().with_batch_exporter(exporter, BaguaTraceRuntime);

    let tracer_provider = builder.build();
    let tracer = tracer_provider.get_tracer("bagua-opentelemetry", Some(env!("CARGO_PKG_VERSION")));
    let _ = global::set_tracer_provider(tracer_provider);

    tracer
}
