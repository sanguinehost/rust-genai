use bytes::Bytes;
use futures::stream::TryStreamExt;
use futures::{Future, Stream};
use reqwest::{RequestBuilder, Response};
use std::collections::VecDeque;
use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};

/// `WebStream` is a simple web stream implementation that splits the stream messages by a given delimiter.
/// - It is intended to be a pragmatic solution for services that do not adhere to the `text/event-stream` format and content type.
/// - For providers that support the standard `text/event-stream`, `genai` uses the `reqwest-eventsource`/`eventsource-stream` crates.
/// - This stream item is just a `String` and has different stream modes that define the message delimiter strategy (without any event typing).
/// - Each "Event" is just string-based and has only one event type, which is a string.
/// - It is the responsibility of the user of this stream to wrap it into a semantically correct stream of events depending on the domain.
#[allow(clippy::type_complexity)]
pub struct WebStream {
	stream_mode: StreamMode,
	reqwest_builder: Option<RequestBuilder>,
	response_future: Option<Pin<Box<dyn Future<Output = Result<Response, Box<dyn Error>>> + Send>>>,
	bytes_stream: Option<Pin<Box<dyn Stream<Item = Result<Bytes, Box<dyn Error>>> + Send>>>,
	// If a poll was a partial message, then we keep the previous part
	partial_message: Option<String>,
	// If a poll retrieved multiple messages, we keep them to be sent in the next poll
	remaining_messages: Option<VecDeque<String>>,
}

pub enum StreamMode {
	// This is used for Cohere with a single `\n`
	Delimiter(&'static str),
	// This is for Gemini (standard JSON array, pretty formatted)
	PrettyJsonArray,
}

impl WebStream {
	pub fn new_with_delimiter(reqwest_builder: RequestBuilder, message_delimiter: &'static str) -> Self {
		Self {
			stream_mode: StreamMode::Delimiter(message_delimiter),
			reqwest_builder: Some(reqwest_builder),
			response_future: None,
			bytes_stream: None,
			partial_message: None,
			remaining_messages: None,
		}
	}

	pub fn new_with_pretty_json_array(reqwest_builder: RequestBuilder) -> Self {
		Self {
			stream_mode: StreamMode::PrettyJsonArray,
			reqwest_builder: Some(reqwest_builder),
			response_future: None,
			bytes_stream: None,
			partial_message: None,
			remaining_messages: None,
		}
	}
}

impl Stream for WebStream {
	type Item = Result<String, Box<dyn Error>>;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let this = self.get_mut();

		// -- First, we check if we have any remaining messages to send.
		if let Some(ref mut remaining_messages) = this.remaining_messages {
			if let Some(msg) = remaining_messages.pop_front() {
				return Poll::Ready(Some(Ok(msg)));
			}
		}

		// -- Then execute the web poll and processing loop
		loop {
			if let Some(ref mut fut) = this.response_future {
				match Pin::new(fut).poll(cx) {
					Poll::Ready(Ok(response)) => {
						let bytes_stream = response.bytes_stream().map_err(|e| Box::new(e) as Box<dyn Error>);
						this.bytes_stream = Some(Box::pin(bytes_stream));
						this.response_future = None;
					}
					Poll::Ready(Err(e)) => {
						this.response_future = None;
						return Poll::Ready(Some(Err(e)));
					}
					Poll::Pending => return Poll::Pending,
				}
			}

			if let Some(ref mut stream) = this.bytes_stream {
				match stream.as_mut().poll_next(cx) {
					Poll::Ready(Some(Ok(bytes))) => {
						let buff_string = match String::from_utf8(bytes.to_vec()) {
							Ok(s) => s,
							Err(e) => return Poll::Ready(Some(Err(Box::new(e) as Box<dyn Error>))),
						};

						// -- Iterate through the parts
						let buff_response = match this.stream_mode {
							StreamMode::Delimiter(delimiter) => {
								process_buff_string_delimited(&buff_string, &mut this.partial_message, delimiter)
							}
							StreamMode::PrettyJsonArray => {
								new_with_pretty_json_array(&buff_string, &mut this.partial_message)
							}
						};

						let BuffResponse {
							mut first_message,
							next_messages,
							candidate_message,
						} = buff_response;

						// -- Add next_messages as remaining messages if present
						if let Some(next_messages) = next_messages {
							this.remaining_messages.get_or_insert(VecDeque::new()).extend(next_messages);
						}

						// -- If we still have a candidate, it's the partial for the next one
						if let Some(candidate_message) = candidate_message {
							// For now, we will just log this
							if this.partial_message.is_some() {
								tracing::warn!("GENAI - WARNING - partial_message is not none");
							}
							this.partial_message = Some(candidate_message);
						}

						// -- If we have a first message, we have to send it.
						if let Some(first_message) = first_message.take() {
							return Poll::Ready(Some(Ok(first_message)));
						}
						continue;
					}
					Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
					Poll::Ready(None) => {
						if let Some(partial) = this.partial_message.take() {
							if !partial.is_empty() {
								// Only emit partial if it's not for JSON mode or if it's valid JSON
								match this.stream_mode {
									StreamMode::PrettyJsonArray => {
										// For JSON mode, validate before emitting
										if partial.trim() == "]" || serde_json::from_str::<serde_json::Value>(&partial).is_ok() {
											tracing::debug!("WebStream: Emitting final partial: {}", partial.chars().take(50).collect::<String>());
											return Poll::Ready(Some(Ok(partial)));
										} else {
											tracing::warn!("WebStream: Discarding invalid partial JSON at stream end: {}", partial.chars().take(50).collect::<String>());
										}
									}
									StreamMode::Delimiter(_) => {
										// For delimiter mode, emit as-is
										return Poll::Ready(Some(Ok(partial)));
									}
								}
							}
						}
						this.bytes_stream = None;
					}
					Poll::Pending => return Poll::Pending,
				}
			}

			if let Some(reqwest_builder) = this.reqwest_builder.take() {
				let fut = async move { reqwest_builder.send().await.map_err(|e| Box::new(e) as Box<dyn Error>) };
				this.response_future = Some(Box::pin(fut));
				continue;
			}

			return Poll::Ready(None);
		}
	}
}

struct BuffResponse {
	first_message: Option<String>,
	next_messages: Option<Vec<String>>,
	candidate_message: Option<String>,
}

/// Process a string buffer for the `pretty_json_array` (for Gemini)
/// It will split the messages as follows:
/// - If it starts with `[`, then the message will be `[`
/// - Then, each main JSON object (from the first `{` to the last `}`) will become a message
/// - Main JSON object `,` delimiter will be skipped
/// - The ending `]` will be sent as a `]` message as well.
///
/// IMPORTANT: Properly handles partial JSON objects that are split across network packets
///            by maintaining state in the partial_message parameter and only emitting complete objects.
fn new_with_pretty_json_array(buff_string: &str, partial_message: &mut Option<String>) -> BuffResponse {
	// Combine with any existing partial message
	let combined_str = if let Some(existing_partial) = partial_message.take() {
		tracing::debug!("WebStream: Combining partial message ({}chars) with new data ({}chars)", existing_partial.len(), buff_string.len());
		format!("{}{}", existing_partial, buff_string)
	} else {
		buff_string.to_string()
	};
	
	let buff_str = combined_str.trim();
	let mut messages: Vec<String> = Vec::new();

	// -- Handle array start
	let (array_start, rest_str) = buff_str
		.strip_prefix('[')
		.map_or((None, buff_str), |rest| (Some("["), rest.trim()));

	if let Some(array_start) = array_start {
		messages.push(array_start.to_string());
	}

	// -- Try to extract complete JSON objects from the remaining content
	let mut remaining_content = rest_str;
	
	// Remove any leading comma
	remaining_content = remaining_content.trim_start_matches(',').trim();
	
	// Check for array end first
	if remaining_content.starts_with(']') {
		messages.push("]".to_string());
		remaining_content = &remaining_content[1..];
	}
	
	// Try to extract JSON objects
	while !remaining_content.is_empty() && remaining_content.starts_with('{') {
		// Find the end of the JSON object by trying to parse incrementally
		let mut found_complete_object = false;
		
		// Look for potential end positions by finding closing braces
		for (i, _) in remaining_content.match_indices('}') {
			let candidate = &remaining_content[..=i];
			
			// Try to parse this candidate as valid JSON
			if let Ok(_) = serde_json::from_str::<serde_json::Value>(candidate) {
				// Found a complete JSON object
				tracing::debug!("WebStream: Found complete JSON object ({}chars)", candidate.len());
				messages.push(candidate.to_string());
				remaining_content = remaining_content[i+1..].trim();
				
				// Skip any comma after the object
				remaining_content = remaining_content.strip_prefix(',').unwrap_or(remaining_content).trim();
				
				// Check for array end
				if remaining_content.starts_with(']') {
					messages.push("]".to_string());
					remaining_content = &remaining_content[1..];
				}
				
				found_complete_object = true;
				break;
			}
		}
		
		if !found_complete_object {
			// No complete JSON object found - store remainder as partial
			if !remaining_content.is_empty() {
				tracing::debug!("WebStream: Storing incomplete JSON as partial ({}chars)", remaining_content.len());
				*partial_message = Some(remaining_content.to_string());
			}
			break;
		}
	}

	// -- Return the buff response
	let first_message = if messages.is_empty() {
		None
	} else {
		Some(messages[0].to_string())
	};

	let next_messages = if messages.len() > 1 {
		Some(messages[1..].to_vec())
	} else {
		None
	};

	BuffResponse {
		first_message,
		next_messages,
		candidate_message: None,
	}
}

/// Process a string buffer for the delimited mode (e.g., Cohere)
fn process_buff_string_delimited(
	buff_string: &str,
	partial_message: &mut Option<String>,
	delimiter: &str,
) -> BuffResponse {
	let mut first_message: Option<String> = None;
	let mut candidate_message: Option<String> = None;
	let mut next_messages: Option<Vec<String>> = None;

	let parts = buff_string.split(delimiter);

	for part in parts {
		// If we already have a candidate, the candidate becomes the message
		if let Some(candidate_message) = candidate_message.take() {
			// If candidate is empty, we skip
			if candidate_message.is_empty() {
				continue;
			}
			let message = candidate_message.to_string();
			if first_message.is_none() {
				first_message = Some(message);
			} else {
				next_messages.get_or_insert_with(Vec::new).push(message);
			}
		} else {
			// And then, this part becomes the candidate
			if let Some(partial) = partial_message.take() {
				candidate_message = Some(format!("{partial}{part}"));
			} else {
				candidate_message = Some(part.to_string());
			}
		}
	}

	BuffResponse {
		first_message,
		next_messages,
		candidate_message,
	}
}
