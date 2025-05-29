//! Internal stream event types that serve as intermediaries between the provider event and the `GenAI` stream event.
//!
//! This allows for flexibility if we want to capture events across providers that do not need to
//! be reflected in the public `ChatStream` event.
//!
//! NOTE: This might be removed at some point as it may not be needed, and we could go directly to the `GenAI` stream.

use crate::chat::Usage;

#[derive(Debug, Default)]
pub struct InterStreamEnd {
	// When `ChatOptions..capture_usage == true`
	pub usage: Option<Usage>,

	// When `ChatOptions..capture_content == true`
	pub content: Option<String>,

	// When `ChatOptions..capture_reasoning_content == true`
	pub reasoning_content: Option<String>,
}

/// Intermediary `StreamEvent`
#[derive(Debug)]
pub enum InterStreamEvent {
	Start,
	Chunk(String),
	ReasoningChunk(String),
	End(InterStreamEnd),
}
