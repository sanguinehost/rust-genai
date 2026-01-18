use bytes::Bytes;
use futures::stream::TryStreamExt;
use futures::{Future, Stream};
use reqwest::{RequestBuilder, Response};
use std::collections::VecDeque;
use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};

/// WebStream is a simple web stream implementation that splits the stream messages by a given delimiter.
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
		if let Some(ref mut remaining_messages) = this.remaining_messages
			&& let Some(msg) = remaining_messages.pop_front()
		{
			return Poll::Ready(Some(Ok(msg)));
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
								process_buff_string_delimited(buff_string, &mut this.partial_message, delimiter)
							}
							StreamMode::PrettyJsonArray => {
								new_with_pretty_json_array(buff_string, &mut this.partial_message)
							}
						};

						let BuffResponse {
							mut first_message,
							next_messages,
							candidate_message,
						} = buff_response?;

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
						} else {
							continue;
						}
					}
					Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
					Poll::Ready(None) => {
						if let Some(partial) = this.partial_message.take()
							&& !partial.is_empty()
						{
							return Poll::Ready(Some(Ok(partial)));
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

/// Process a string buffer for the pretty_json_array (for Gemini)
/// It will split the messages as follows:
/// - If it starts with `[`, then the message will be `[`
/// - Then, each main JSON object (from the first `{` to the last `}`) will become a message
/// - Main JSON object `,` delimiter will be skipped
/// - The ending `]` will be sent as a `]` message as well.
fn new_with_pretty_json_array(
	buff_string: String,
	partial_message: &mut Option<String>,
) -> Result<BuffResponse, crate::webc::Error> {
	let buff_str = buff_string.trim();

	let mut messages: Vec<String> = Vec::new();

	// -- Capture the array start
	let (array_start, rest_str) = match buff_str.strip_prefix('[') {
		Some(rest) => (Some("["), rest.trim()),
		None => (None, buff_str),
	};

	if let Some(array_start) = array_start {
		messages.push(array_start.to_string());
	}

	let full_str = if let Some(partial) = partial_message.take() {
		format!("{partial}{rest_str}")
	} else {
		rest_str.to_string()
	};

	if !full_str.is_empty() {
		let mut cursor = full_str.as_str();
		loop {
			cursor = cursor.trim_start_matches(|c| c == ',' || c == '\n' || c == ' ' || c == '\t');
			if cursor.is_empty() {
				break;
			}

			let mut stream = serde_json::Deserializer::from_str(cursor).into_iter::<serde_json::Value>();
			match stream.next() {
				Some(Ok(val)) => {
					messages.push(val.to_string());
					let offset = stream.byte_offset();
					cursor = &cursor[offset..];
				}
				Some(Err(_)) => {
					// If it failed, check if it's the array end delimiter ']'
					if cursor.starts_with(']') {
						messages.push("]".to_string());
						cursor = &cursor[1..].trim_start();
						// Continue the loop to handle any trailing commas or other objects (though unlikely after ']')
						continue;
					}

					*partial_message = Some(cursor.to_string());
					break;
				}
				None => {
					break;
				}
			}
		}
	}

	// -- Return the buff response
	let first_message = if !messages.is_empty() {
		Some(messages[0].to_string())
	} else {
		None
	};

	let next_messages = if messages.len() > 1 {
		Some(messages[1..].to_vec())
	} else {
		None
	};

	Ok(BuffResponse {
		first_message,
		next_messages,
		candidate_message: partial_message.take(),
	})
}

/// Process a string buffer for the delimited mode (e.g., Cohere)
fn process_buff_string_delimited(
	buff_string: String,
	partial_message: &mut Option<String>,
	delimiter: &str,
) -> Result<BuffResponse, crate::webc::Error> {
	let mut first_message: Option<String> = None;
	let mut candidate_message: Option<String> = None;
	let mut next_messages: Option<Vec<String>> = None;

	let parts = buff_string.split(delimiter);

	for part in parts {
		// If we already have a candidate, the candidate becomes the message
		if let Some(candidate_message) = candidate_message.take() {
			// If candidate is empty, we skip
			if !candidate_message.is_empty() {
				let message = candidate_message.to_string();
				if first_message.is_none() {
					first_message = Some(message);
				} else {
					next_messages.get_or_insert_with(Vec::new).push(message);
				}
			} else {
				continue;
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

	Ok(BuffResponse {
		first_message,
		next_messages,
		candidate_message,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_new_with_pretty_json_array_nested_ok() {
		let mut partial = None;
		// Chunk ends with a nested array bracket.
		let chunk = r#"{ "a": [1] }"#.to_string();
		let res = new_with_pretty_json_array(chunk, &mut partial).unwrap();

		// Should parse successfully as one message.
		assert_eq!(res.first_message.as_deref(), Some(r#"{"a":[1]}"#));
		assert_eq!(partial, None);
	}

	#[test]
	fn test_new_with_pretty_json_array_partial_nested() {
		let mut partial = None;
		// Chunk ends with a nested bracket, but it's partial.
		let chunk = r#"{ "a": [1"#.to_string();
		let res = new_with_pretty_json_array(chunk, &mut partial).unwrap();

		// Should be partial, no messages.
		assert_eq!(res.first_message, None);
		assert_eq!(res.candidate_message.as_deref(), Some(r#"{ "a": [1"#));
	}

	#[test]
	fn test_new_with_pretty_json_array_full_stream() {
		let mut partial = None;
		// Full stream: [ { "a": 1 }, { "b": 2 } ]
		let chunk = r#"[ { "a": 1 }, { "b": 2 } ]"#.to_string();
		let res = new_with_pretty_json_array(chunk, &mut partial).unwrap();

		assert_eq!(res.first_message.as_deref(), Some("["));
		let next = res.next_messages.unwrap();
		assert_eq!(next.len(), 3);
		assert_eq!(next[0], r#"{"a":1}"#);
		assert_eq!(next[1], r#"{"b":2}"#);
		assert_eq!(next[2], "]");
	}
}
