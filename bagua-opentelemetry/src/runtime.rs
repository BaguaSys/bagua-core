use futures::{future::BoxFuture, Stream};
use opentelemetry::{
    global, sdk,
    sdk::trace::{TraceRuntime, TrySend},
    trace::Tracer,
    trace::TracerProvider,
};
use std::time::Duration;

use opentelemetry;
use opentelemetry::runtime::Runtime;
use opentelemetry::sdk::trace::BatchMessage;
use opentelemetry::trace::TraceError;

#[derive(Debug, Clone)]
pub struct BaguaTraceRuntime;

// /// TrySend is an abstraction of sender that is capable to send BatchMessage with reference.
// pub trait TrySend: Sync + Send {
//     /// Try to send one batch message to worker thread.
//     ///
//     /// It can fail because either the receiver has closed or the buffer is full.
//     fn try_send(&self, item: BatchMessage) -> Result<(), TraceError>;
// }

const CHANNEL_FULL_ERROR: &str =
    "cannot send span to the batch span processor because the channel is full";
const CHANNEL_CLOSED_ERROR: &str =
    "cannot send span to the batch span processor because the channel is closed";

#[derive(Debug)]
pub struct MySender {
    inner: tokio::sync::mpsc::Sender<BatchMessage>,
}

impl TrySend for MySender {
    fn try_send(&self, item: BatchMessage) -> Result<(), TraceError> {
        self.inner.try_send(item).map_err(|err| match err {
            tokio::sync::mpsc::error::TrySendError::Full(_) => TraceError::from(CHANNEL_FULL_ERROR),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => {
                TraceError::from(CHANNEL_CLOSED_ERROR)
            }
        })
    }
}

pub fn tokio_interval_stream(
    period: std::time::Duration,
) -> tokio_stream::wrappers::IntervalStream {
    tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(period))
}

impl Runtime for BaguaTraceRuntime {
    type Interval = tokio_stream::wrappers::IntervalStream;
    type Delay = tokio::time::Sleep;

    fn interval(&self, duration: Duration) -> Self::Interval {
        tokio_interval_stream(duration)
    }

    fn spawn(&self, future: BoxFuture<'static, ()>) {
        let _ = tokio::spawn(future);
    }

    fn delay(&self, duration: Duration) -> Self::Delay {
        tokio::time::sleep(duration)
    }
}

impl TraceRuntime for BaguaTraceRuntime {
    type Receiver = tokio_stream::wrappers::ReceiverStream<BatchMessage>;
    type Sender = MySender;

    fn batch_message_channel(&self, capacity: usize) -> (Self::Sender, Self::Receiver) {
        let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
        (
            MySender { inner: sender },
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )
    }
}
