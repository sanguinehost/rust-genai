use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::ChatOptionsSet;
use crate::webc::WebStream;
use crate::{Error, ModelIden, Result};
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};

pub struct GeminiStreamer {
	inner: WebStream,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to not poll the `EventSource` after a `MessageStop` event.
	done: bool,
	captured_data: StreamerCapturedData,
}

impl GeminiStreamer {
	pub fn new(inner: WebStream, model_iden: ModelIden, options_set: &ChatOptionsSet) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: StreamerCapturedData::default(),
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

		while let Poll::Ready(item) = Pin::new(&mut self.inner).poll_next(cx) {
			match item {
				Some(Ok(raw_message)) => {
					// This is the message sent by the WebStream in PrettyJsonArray mode.
					// - `[` document start
					// - `{...}` block
					// - `]` document end
					let inter_event = match raw_message.as_str() {
						"[" => InterStreamEvent::Start,
						"]" => {
							let inter_stream_end = InterStreamEnd {
								usage: self.captured_data.usage.take(),
								content: self.captured_data.content.take(),
								reasoning_content: self.captured_data.reasoning_content.take(),
							};

							InterStreamEvent::End(inter_stream_end)
						}
						block_string => {
							// -- Parse the block to JSON
							let json_block = match serde_json::from_str::<Value>(block_string).map_err(|serde_error| {
								Error::StreamParse {
									model_iden: self.options.model_iden.clone(),
									serde_error,
								}
							}) {
								Ok(json_block) => json_block,
								Err(err) => {
									tracing::error!("Gemini Adapter Stream Error: {}", err);
									return Poll::Ready(Some(Err(err)));
								}
							};

							// -- Extract the Gemini Response
							let gemini_response = match super::GeminiAdapter::body_to_gemini_chat_response(
								&self.options.model_iden,
								json_block,
								&ChatOptionsSet::default(), // Pass default options_set in streaming
							) {
								Ok(gemini_response) => gemini_response,
								Err(err) => {
									tracing::error!("Gemini Adapter Stream Error: {}", err);
									return Poll::Ready(Some(Err(err)));
								}
							};

							let super::GeminiChatResponse { contents, usage } = gemini_response;

							// For streaming with InterStreamEvent::Chunk(String), we process only the text from the first candidate.
							let first_content = contents.into_iter().next();

							// Capture usage regardless of content type for this chunk, as Gemini sends it cumulatively.
							if self.options.capture_usage {
								self.captured_data.usage = Some(usage);
							}

							match first_content {
								Some(super::GeminiChatContent::Parts(parts)) => {
									// Extract text content from parts for streaming
									let text_content: String = parts
										.into_iter()
										.filter_map(|part| match part {
											crate::chat::ContentPart::Text(text) => Some(text),
											crate::chat::ContentPart::Image { .. }
											| crate::chat::ContentPart::Document { .. } => None, // Ignore image and document parts for streaming chunks
										})
										.collect();

									if text_content.is_empty() {
										// No text content from the first candidate, usage already updated. Skip emitting an empty chunk.
										continue; // Go to next item from WebStream
									}
									// Capture content text if option is set
									if self.options.capture_content {
										match self.captured_data.content {
											Some(ref mut c) => c.push_str(&text_content),
											None => self.captured_data.content = Some(text_content.clone()),
										}
									}
									// This is the event to return for this branch of the outer match
									InterStreamEvent::Chunk(text_content)
								}
								Some(super::GeminiChatContent::ToolCall(tool_call)) => {
									tracing::warn!(
										"GeminiStreamer received a ToolCall in a stream chunk: {:?}. This will not be emitted as InterStreamEvent::Chunk.",
										tool_call
									);
									continue; // Go to next item from WebStream, effectively skipping this block for event emission
								}
								None => {
									// No content from the first candidate, usage already updated. Skip emitting an empty chunk.
									continue; // Go to next item from WebStream
								}
							}
						} // End of block_string processing, result is the InterStreamEvent for this iteration
					};

					return Poll::Ready(Some(Ok(inter_event)));
				}
				Some(Err(err)) => {
					tracing::error!("Gemini Adapter Stream Error: {}", err);
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
					})));
				}
				None => {
					self.done = true;
					return Poll::Ready(None);
				}
			}
		}
		Poll::Pending
	}
}
