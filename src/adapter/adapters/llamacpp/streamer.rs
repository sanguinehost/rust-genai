//! Streaming support for LlamaCpp adapter that provides token-by-token streaming
//! of generated responses.

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::ChatStreamResponse;
use crate::{ModelIden, Result};

/// Represents a chunk of streamed content from llama.cpp generation
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Delta text content to append
    Delta(String),
    /// Generation completed
    Done,
    /// Error occurred during generation
    Error(String),
}

/// Streamer for LlamaCpp that implements Stream for InterStreamEvent
pub struct LlamaCppStreamer {
    receiver: mpsc::Receiver<StreamChunk>,
    buffer: String,
    finished: bool,
}

impl LlamaCppStreamer {
    /// Create a new LlamaCppStreamer with the given receiver
    pub fn new(receiver: mpsc::Receiver<StreamChunk>) -> Self {
        Self {
            receiver,
            buffer: String::new(),
            finished: false,
        }
    }
}

impl Stream for LlamaCppStreamer {
    type Item = Result<InterStreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        
        if this.finished {
            return Poll::Ready(None);
        }

        match this.receiver.poll_recv(cx) {
            Poll::Ready(Some(chunk)) => match chunk {
                StreamChunk::Delta(text) => {
                    this.buffer.push_str(&text);
                    Poll::Ready(Some(Ok(InterStreamEvent::Chunk(text))))
                }
                StreamChunk::Done => {
                    this.finished = true;
                    let end_event = InterStreamEnd {
                        content: if this.buffer.is_empty() { None } else { Some(this.buffer.clone()) },
                        ..Default::default()
                    };
                    Poll::Ready(Some(Ok(InterStreamEvent::End(end_event))))
                }
                StreamChunk::Error(err) => {
                    this.finished = true;
                    Poll::Ready(Some(Err(crate::Error::AdapterError(err))))
                }
            },
            Poll::Ready(None) => {
                // Channel closed
                this.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Create a streaming channel pair for llama.cpp generation
pub fn create_streaming_channel(model_iden: ModelIden) -> (mpsc::Sender<StreamChunk>, ChatStreamResponse) {
    let (tx, rx) = mpsc::channel(32);
    
    let streamer = LlamaCppStreamer::new(rx);
    let chat_stream = crate::chat::ChatStream::from_inter_stream(streamer);
    
    let response = ChatStreamResponse {
        model_iden,
        stream: chat_stream,
    };
    
    (tx, response)
}