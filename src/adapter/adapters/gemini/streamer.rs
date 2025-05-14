use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::gemini::{GeminiAdapter, GeminiChatResponse};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::ChatOptionsSet;
use crate::{Error, ModelIden, Result};
use reqwest_eventsource::{Event, EventSource};
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::GeminiChatContent;

pub struct GeminiStreamer {
	inner: EventSource,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to not poll the EventSource after a MessageStop event.
	done: bool,
	captured_data: StreamerCapturedData,
	/// If a single Gemini event contains both reasoning and main content,
	/// the main content is stored here to be emitted after the reasoning chunk.
	pending_main_content: Option<String>,
}

impl GeminiStreamer {
	pub fn new(inner: EventSource, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			pending_main_content: None,
		}
	}
}

// Implement futures::Stream for InterStream<GeminiStream>
impl futures::Stream for GeminiStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}

		// -- Emit pending main content first if it exists
		if let Some(main_text) = self.pending_main_content.take() {
			if self.options.capture_content {
				match self.captured_data.content {
					Some(ref mut c) => c.push_str(&main_text),
					None => self.captured_data.content = Some(main_text.clone()),
				}
			}
			return Poll::Ready(Some(Ok(InterStreamEvent::Chunk(main_text))));
		}

		// Loop to process multiple events if they are ready in the EventSource
		loop {
			match Pin::new(&mut self.inner).poll_next(cx) {
				Poll::Ready(Some(Ok(event))) => {
					match event {
						Event::Open => {
							// First event in the stream
							return Poll::Ready(Some(Ok(InterStreamEvent::Start)));
						}
						Event::Message(message) => {
							tracing::debug!("GeminiStreamer received SSE message.data: >>>{}<<<", message.data);
							let json_block = match serde_json::from_str::<Value>(&message.data) {
								Ok(json_block) => json_block,
								Err(serde_error) => {
									let err = Error::StreamParse {
										model_iden: self.options.model_iden.clone(),
										serde_error,
									};
									tracing::error!("GeminiStreamer JSON parsing error: {}", err);
									return Poll::Ready(Some(Err(err)));
								}
							};

							match GeminiAdapter::body_to_gemini_chat_response(&self.options.model_iden, json_block) {
								Ok(gemini_response) => {
									let GeminiChatResponse { content, reasoning_content, usage } = gemini_response;

									if self.options.capture_usage {
										self.captured_data.usage = Some(usage);
									}

									let mut main_text_from_event: Option<String> = None;
									if let Some(GeminiChatContent::Text(text)) = content {
										main_text_from_event = Some(text);
									} else if let Some(GeminiChatContent::ToolCall(tool_call)) = content {
										// Emit the ToolCall event directly
										return Poll::Ready(Some(Ok(InterStreamEvent::ToolCall(tool_call))));
									}

									if let Some(GeminiChatContent::Text(reasoning_text)) = reasoning_content {
										if self.options.capture_reasoning_content {
											match self.captured_data.reasoning_content {
												Some(ref mut c) => c.push_str(&reasoning_text),
												None => self.captured_data.reasoning_content = Some(reasoning_text.clone()),
											}
										}
										if let Some(main_text) = main_text_from_event {
											// If there's also main text, store it to be emitted next.
											self.pending_main_content = Some(main_text);
										}
										return Poll::Ready(Some(Ok(InterStreamEvent::ReasoningChunk(reasoning_text))));
									} else if let Some(main_text) = main_text_from_event {
										// No reasoning text, but there is main text.
										if self.options.capture_content {
											match self.captured_data.content {
												Some(ref mut c) => c.push_str(&main_text),
												None => self.captured_data.content = Some(main_text.clone()),
											}
										}
										return Poll::Ready(Some(Ok(InterStreamEvent::Chunk(main_text))));
									} else {
										// Neither reasoning_text nor main_text. Could be just usage update or unhandled tool_call in reasoning_content.
										// Or a ToolCall was the main content and was logged above.
										// Continue polling for the next event.
										continue;
									}
								}
								Err(err) => {
									tracing::error!(
										"GeminiStreamer error processing Gemini response body: {}",
										err
									);
									return Poll::Ready(Some(Err(err)));
								}
							}
						}
					}
				}
				Poll::Ready(Some(Err(event_source_err))) => {
					if let reqwest_eventsource::Error::StreamEnded = event_source_err {
						tracing::debug!("GeminiStreamer: EventSource reported StreamEnded (graceful close).");
						self.done = true;
						let inter_stream_end = InterStreamEnd {
							captured_usage: self.captured_data.usage.take(),
							captured_content: self.captured_data.content.take(),
							captured_reasoning_content: self.captured_data.reasoning_content.take(),
						};
						return Poll::Ready(Some(Ok(InterStreamEvent::End(inter_stream_end))));
					} else {
						tracing::error!("GeminiStreamer EventSource error: {}", event_source_err);
						self.done = true;
						return Poll::Ready(Some(Err(Error::ReqwestEventSource(event_source_err.into()))));
					}
				}
				Poll::Ready(None) => {
					tracing::debug!("GeminiStreamer: EventSource returned None (stream closed).");
					self.done = true;
					let inter_stream_end = InterStreamEnd {
						captured_usage: self.captured_data.usage.take(),
						captured_content: self.captured_data.content.take(),
						captured_reasoning_content: self.captured_data.reasoning_content.take(),
					};
					return Poll::Ready(Some(Ok(InterStreamEvent::End(inter_stream_end))));
				}
				Poll::Pending => {
					return Poll::Pending;
				}
			}
		}
	}
}
