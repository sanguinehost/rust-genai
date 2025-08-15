//! `ChatOptions` allows customization of a chat request.
//! - It can be provided at the `client::exec_chat(..)` level as an argument,
//! - or set in the client config `client_config.with_chat_options(..)` to be used as the default for all requests
//!
//! Note 1: In the future, we will probably allow setting the client
//! Note 2: Extracting it from the `ChatRequest` object allows for better reusability of each component.

use crate::chat::chat_req_response_format::ChatResponseFormat;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

// region:    --- Safety Settings

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmCategory {
	#[serde(rename = "HARM_CATEGORY_HARASSMENT")]
	Harassment,
	#[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
	HateSpeech,
	#[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
	SexuallyExplicit,
	#[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
	DangerousContent,
	#[serde(rename = "HARM_CATEGORY_CIVIC_INTEGRITY")]
	CivicIntegrity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmBlockThreshold {
	#[serde(rename = "BLOCK_NONE")]
	BlockNone,
	#[serde(rename = "OFF")]
	Off,
	#[serde(rename = "BLOCK_ONLY_HIGH")]
	BlockOnlyHigh,
	#[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
	BlockMediumAndAbove,
	#[serde(rename = "BLOCK_LOW_AND_ABOVE")]
	BlockLowAndAbove,
	#[serde(rename = "HARM_BLOCK_THRESHOLD_UNSPECIFIED")]
	Unspecified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
	pub category: HarmCategory,
	pub threshold: HarmBlockThreshold,
}

impl SafetySetting {
	#[must_use]
	pub const fn new(category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
		Self { category, threshold }
	}
}

// endregion: --- Safety Settings

/// Chat Options that are considered for any `Client::exec...` calls.
///
/// A fallback `ChatOptions` can also be set at the `Client` during the client builder phase.
///
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
	/// Will be used for this request if the Adapter/provider supports it.
	pub temperature: Option<f64>,

	/// Will be used for this request if the Adapter/provider supports it.
	pub max_tokens: Option<u32>,

	/// Will be used for this request if the Adapter/provider supports it.
	pub top_p: Option<f64>,

	/// Specifies sequences used as end markers when generating text
	pub stop_sequences: Vec<String>,

	// -- Stream Options
	/// (for streaming only) Capture the meta usage when in stream mode
	/// `StreamEnd` event payload will contain `captured_usage`
	/// > Note: Will capture the `MetaUsage`
	pub capture_usage: Option<bool>,

	/// (for streaming only) Capture/concatenate the full message content from all content chunks
	/// `StreamEnd` from `StreamEvent::End(StreamEnd)` will contain `StreamEnd.captured_content`
	pub capture_content: Option<bool>,

	/// (for streaming only) Capture/concatenate the full message content from all content chunks
	/// `StreamEnd` from `StreamEvent::End(StreamEnd)` will contain `StreamEnd.captured_reasoning_content`
	pub capture_reasoning_content: Option<bool>,

	/// Specifies the response format for a chat request.
	/// - `ChatResponseFormat::JsonMode` is for OpenAI-like API usage, where the user must specify in the prompt that they want a JSON format response.
	///
	/// NOTE: More response formats are coming soon.
	pub response_format: Option<ChatResponseFormat>,

	// -- Reasoning options
	/// Denote if the content should be parsed to extract eventual `<think>...</think>` content
	/// into `ChatResponse.reasoning_content`
	pub normalize_reasoning_content: Option<bool>,

	pub reasoning_effort: Option<ReasoningEffort>,

	// -- Gemini Specific Options (or other future provider specific advanced options)
	/// Corresponds to `topK` in Gemini.
	pub top_k: Option<i32>,
	/// Corresponds to `seed` in Gemini.
	pub seed: Option<i32>,
	/// Corresponds to `presencePenalty` in Gemini.
	pub presence_penalty: Option<f32>,
	/// Corresponds to `frequencyPenalty` in Gemini.
	pub frequency_penalty: Option<f32>,
	/// Corresponds to `candidateCount` in Gemini.
	pub candidate_count: Option<i32>,
	/// Corresponds to `cachedContent` in Gemini (expects the cached content ID string).
	pub cached_content_id: Option<String>,
	/// Corresponds to `toolConfig.functionCallingConfig.mode` in Gemini (e.g., "AUTO", "ANY", "NONE").
	pub function_calling_mode: Option<String>,
	/// Corresponds to `toolConfig.functionCallingConfig.allowedFunctionNames` in Gemini.
	pub allowed_function_names: Option<Vec<String>>,

	/// Corresponds to `generationConfig.responseModalities` in Gemini.
	/// E.g., `vec!["TEXT".to_string(), "IMAGE".to_string()]`
	pub response_modalities: Option<Vec<String>>,

	/// Corresponds to `safetySettings` in Gemini.
	/// Configures safety filtering for harmful content.
	pub safety_settings: Option<Vec<SafetySetting>>,

	// -- Gemini Specific --
	/// (Gemini specific) If true, will include the thoughts in the response.
	pub include_thoughts: Option<bool>,
}

/// Chainable Setters
impl ChatOptions {
	/// Set the `temperature` for this request.
	#[must_use]
	pub const fn with_temperature(mut self, value: f64) -> Self {
		self.temperature = Some(value);
		self
	}

	/// Set the `max_tokens` for this request.
	#[must_use]
	pub const fn with_max_tokens(mut self, value: u32) -> Self {
		self.max_tokens = Some(value);
		self
	}

	/// Set the `top_p` for this request.
	#[must_use]
	pub const fn with_top_p(mut self, value: f64) -> Self {
		self.top_p = Some(value);
		self
	}

	/// Set the `capture_usage` for this request.
	#[must_use]
	pub const fn with_capture_usage(mut self, value: bool) -> Self {
		self.capture_usage = Some(value);
		self
	}

	/// Set the `capture_content` for this request.
	#[must_use]
	pub const fn with_capture_content(mut self, value: bool) -> Self {
		self.capture_content = Some(value);
		self
	}

	/// Set the `capture_reasoning_content` for this request.
	#[must_use]
	pub const fn with_capture_reasoning_content(mut self, value: bool) -> Self {
		self.capture_reasoning_content = Some(value);
		self
	}

	#[must_use]
	pub fn with_stop_sequences(mut self, values: Vec<String>) -> Self {
		self.stop_sequences = values;
		self
	}

	#[must_use]
	pub const fn with_normalize_reasoning_content(mut self, value: bool) -> Self {
		self.normalize_reasoning_content = Some(value);
		self
	}

	/// Set the `response_format` for this request.
	#[must_use]
	pub fn with_response_format(mut self, res_format: impl Into<ChatResponseFormat>) -> Self {
		self.response_format = Some(res_format.into());
		self
	}

	#[must_use]
	pub const fn with_reasoning_effort(mut self, value: ReasoningEffort) -> Self {
		self.reasoning_effort = Some(value);
		self
	}

	// -- Gemini Specific Option Setters (or other future provider specific advanced options)
	#[must_use]
	pub const fn with_top_k(mut self, value: i32) -> Self {
		self.top_k = Some(value);
		self
	}

	#[must_use]
	pub const fn with_seed(mut self, value: i32) -> Self {
		self.seed = Some(value);
		self
	}

	#[must_use]
	pub const fn with_presence_penalty(mut self, value: f32) -> Self {
		self.presence_penalty = Some(value);
		self
	}

	#[must_use]
	pub const fn with_frequency_penalty(mut self, value: f32) -> Self {
		self.frequency_penalty = Some(value);
		self
	}

	#[must_use]
	pub const fn with_candidate_count(mut self, value: i32) -> Self {
		self.candidate_count = Some(value);
		self
	}

	#[must_use]
	pub fn with_cached_content_id(mut self, value: String) -> Self {
		self.cached_content_id = Some(value);
		self
	}

	#[must_use]
	pub fn with_function_calling_mode(mut self, value: String) -> Self {
		self.function_calling_mode = Some(value);
		self
	}

	#[must_use]
	pub fn with_allowed_function_names(mut self, values: Vec<String>) -> Self {
		self.allowed_function_names = Some(values);
		self
	}

	#[must_use]
	pub fn with_response_modalities(mut self, values: Vec<String>) -> Self {
		self.response_modalities = Some(values);
		self
	}

	#[must_use]
	pub fn with_safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
		self.safety_settings = Some(settings);
		self
	}

	#[must_use]
	pub const fn with_include_thoughts(mut self, value: bool) -> Self {
		self.include_thoughts = Some(value);
		self
	}

	// -- Deprecated

	/// Set the `json_mode` for this request.
	///
	/// IMPORTANT: This is deprecated now; use `with_response_format(ChatResponseFormat::JsonMode)`
	///
	/// IMPORTANT: When this is `JsonMode`, it's important to instruct the model to produce JSON yourself
	///            for many models/providers to work correctly. This can be approximately done
	///            by checking if any System and potentially User messages contain `"json"`
	///            (make sure to check the `.system` property as well).
	#[deprecated(note = "Use with_response_format(ChatResponseFormat::JsonMode)")]
	#[must_use]
	pub fn with_json_mode(mut self, value: bool) -> Self {
		if value {
			self.response_format = Some(ChatResponseFormat::JsonMode);
		}
		self
	}
}

// region:    --- ReasoningEffort

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningEffort {
	Low,
	Medium,
	High,
	Budget(u32),
}

impl ReasoningEffort {
	/// Returns the lowercase, static variant name.
	/// Budget always returns "budget" regardless of the number.
	#[must_use]
	pub const fn variant_name(&self) -> &'static str {
		match self {
			Self::Low => "low",
			Self::Medium => "medium",
			Self::High => "high",
			Self::Budget(_) => "budget",
		}
	}

	/// Keywords are just the "high", "medium", "low",
	/// Budget will be None
	#[must_use]
	pub const fn as_keyword(&self) -> Option<&'static str> {
		match self {
			Self::Low => Some("low"),
			Self::Medium => Some("medium"),
			Self::High => Some("high"),
			Self::Budget(_) => None,
		}
	}

	/// Keywords are just the "high", "medium", "low",
	/// This function will not create budget variant (no)
	#[must_use]
	pub fn from_keyword(name: &str) -> Option<Self> {
		match name {
			"low" => Some(Self::Low),
			"medium" => Some(Self::Medium),
			"high" => Some(Self::High),
			_ => None,
		}
	}

	/// If the `model_name` ends with the lowercase string of a `ReasoningEffort` variant,
	/// return the `ReasoningEffort` and the trimmed `model_name`.
	///
	/// Otherwise, return the `model_name` as is.
	///
	/// This will not create budget variant, only the keyword one
	/// Returns (`reasoning_effort`, `model_name`)
	#[must_use]
	pub fn from_model_name(model_name: &str) -> (Option<Self>, &str) {
		if let Some((prefix, last)) = model_name.rsplit_once('-') {
			if let Some(effort) = Self::from_keyword(last) {
				return (Some(effort), prefix);
			}
		}
		(None, model_name)
	}
}

impl std::fmt::Display for ReasoningEffort {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Low => write!(f, "low"),
			Self::Medium => write!(f, "medium"),
			Self::High => write!(f, "high"),
			Self::Budget(n) => write!(f, "{n}"),
		}
	}
}

impl std::str::FromStr for ReasoningEffort {
	type Err = Error;

	fn from_str(s: &str) -> Result<Self> {
		Self::from_keyword(s)
			.or_else(|| s.parse::<u32>().ok().map(Self::Budget))
			.ok_or(Error::ReasoningParsingError { actual: s.to_string() })
	}
}

// endregion: --- ReasoningEffort

// region:    --- ChatOptionsSet

/// This is an internal crate struct to resolve the `ChatOptions` value in a cascading manner.
///
/// First, it attempts to get the value at the chat level (`ChatOptions` from the `exec_chat`...(...) argument).
/// If a value for the property is not found, it looks at the client default one.
#[derive(Default, Clone, Debug)]
pub struct ChatOptionsSet<'a, 'b> {
	client: Option<&'a ChatOptions>,
	chat: Option<&'b ChatOptions>,
}

impl<'a, 'b> ChatOptionsSet<'a, 'b> {
	#[must_use]
	pub const fn with_client_options(mut self, options: Option<&'a ChatOptions>) -> Self {
		self.client = options;
		self
	}
	#[must_use]
	pub const fn with_chat_options(mut self, options: Option<&'b ChatOptions>) -> Self {
		self.chat = options;
		self
	}
}

impl ChatOptionsSet<'_, '_> {
	#[must_use]
	pub fn temperature(&self) -> Option<f64> {
		self.chat
			.and_then(|chat| chat.temperature)
			.or_else(|| self.client.and_then(|client| client.temperature))
	}

	#[must_use]
	pub fn max_tokens(&self) -> Option<u32> {
		self.chat
			.and_then(|chat| chat.max_tokens)
			.or_else(|| self.client.and_then(|client| client.max_tokens))
	}

	#[must_use]
	pub fn top_p(&self) -> Option<f64> {
		self.chat
			.and_then(|chat| chat.top_p)
			.or_else(|| self.client.and_then(|client| client.top_p))
	}

	#[must_use]
	pub fn stop_sequences(&self) -> &[String] {
		self.chat
			.map(|chat| &*chat.stop_sequences)
			.or_else(|| self.client.map(|client| &*client.stop_sequences))
			.unwrap_or(&[])
	}

	#[must_use]
	pub fn capture_usage(&self) -> Option<bool> {
		self.chat
			.and_then(|chat| chat.capture_usage)
			.or_else(|| self.client.and_then(|client| client.capture_usage))
	}

	#[must_use]
	pub fn capture_content(&self) -> Option<bool> {
		self.chat
			.and_then(|chat| chat.capture_content)
			.or_else(|| self.client.and_then(|client| client.capture_content))
	}

	#[must_use]
	pub fn capture_reasoning_content(&self) -> Option<bool> {
		self.chat
			.and_then(|chat| chat.capture_reasoning_content)
			.or_else(|| self.client.and_then(|client| client.capture_reasoning_content))
	}

	#[must_use]
	pub fn response_format(&self) -> Option<&ChatResponseFormat> {
		self.chat
			.and_then(|chat| chat.response_format.as_ref())
			.or_else(|| self.client.and_then(|client| client.response_format.as_ref()))
	}

	#[must_use]
	pub fn normalize_reasoning_content(&self) -> Option<bool> {
		self.chat
			.and_then(|chat| chat.normalize_reasoning_content)
			.or_else(|| self.client.and_then(|client| client.normalize_reasoning_content))
	}

	#[must_use]
	pub fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
		self.chat
			.and_then(|chat| chat.reasoning_effort.as_ref())
			.or_else(|| self.client.and_then(|client| client.reasoning_effort.as_ref()))
	}

	/// Returns true only if there is a `ChatResponseFormat::JsonMode`
	#[deprecated(note = "Use .response_format()")]
	#[allow(unused)]
	#[must_use]
	pub fn json_mode(&self) -> Option<bool> {
		match self.response_format() {
			Some(ChatResponseFormat::JsonMode) => Some(true),
			None => None,
			_ => Some(false),
		}
	}

	// -- Gemini Specific Option Accessors (or other future provider specific advanced options)
	#[must_use]
	pub fn top_k(&self) -> Option<i32> {
		self.chat
			.and_then(|chat| chat.top_k)
			.or_else(|| self.client.and_then(|client| client.top_k))
	}

	#[must_use]
	pub fn seed(&self) -> Option<i32> {
		self.chat
			.and_then(|chat| chat.seed)
			.or_else(|| self.client.and_then(|client| client.seed))
	}

	#[must_use]
	pub fn presence_penalty(&self) -> Option<f32> {
		self.chat
			.and_then(|chat| chat.presence_penalty)
			.or_else(|| self.client.and_then(|client| client.presence_penalty))
	}

	#[must_use]
	pub fn frequency_penalty(&self) -> Option<f32> {
		self.chat
			.and_then(|chat| chat.frequency_penalty)
			.or_else(|| self.client.and_then(|client| client.frequency_penalty))
	}

	#[must_use]
	pub fn candidate_count(&self) -> Option<i32> {
		self.chat
			.and_then(|chat| chat.candidate_count)
			.or_else(|| self.client.and_then(|client| client.candidate_count))
	}

	#[must_use]
	pub fn cached_content_id(&self) -> Option<&String> {
		self.chat
			.and_then(|chat| chat.cached_content_id.as_ref())
			.or_else(|| self.client.and_then(|client| client.cached_content_id.as_ref()))
	}

	#[must_use]
	pub fn function_calling_mode(&self) -> Option<&String> {
		self.chat
			.and_then(|chat| chat.function_calling_mode.as_ref())
			.or_else(|| self.client.and_then(|client| client.function_calling_mode.as_ref()))
	}

	#[must_use]
	pub fn allowed_function_names(&self) -> Option<&Vec<String>> {
		self.chat
			.and_then(|chat| chat.allowed_function_names.as_ref())
			.or_else(|| self.client.and_then(|client| client.allowed_function_names.as_ref()))
	}

	#[must_use]
	pub fn response_modalities(&self) -> Option<&Vec<String>> {
		self.chat
			.and_then(|chat| chat.response_modalities.as_ref())
			.or_else(|| self.client.and_then(|client| client.response_modalities.as_ref()))
	}

	#[must_use]
	pub fn safety_settings(&self) -> Option<&Vec<SafetySetting>> {
		self.chat
			.and_then(|chat| chat.safety_settings.as_ref())
			.or_else(|| self.client.and_then(|client| client.safety_settings.as_ref()))
	}

	#[must_use]
	pub fn include_thoughts(&self) -> Option<bool> {
		self.chat
			.and_then(|chat| chat.include_thoughts)
			.or_else(|| self.client.and_then(|client| client.include_thoughts))
	}
}

// endregion: --- ChatOptionsSet
