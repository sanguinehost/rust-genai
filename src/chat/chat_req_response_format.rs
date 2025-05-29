use derive_more::From;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The chat response format for the `ChatRequest` for structured output.
/// This will be taken into consideration only if the provider supports it.
///
/// > Note: Currently, the AI Providers do not report an error if not supported; it will just be ignored.
/// >       This may change in the future.
#[derive(Debug, Clone, From, Serialize, Deserialize)]
pub enum ChatResponseFormat {
	/// Request to return a well-formatted JSON. Mostly for `OpenAI`.
	/// Note: Make sure to add "Reply in JSON format." to the prompt or system to ensure it works correctly.
	JsonMode,

	/// Request to return a structured output.
	#[from]
	JsonSpec(JsonSpec),

	/// Request to return an enum value.
	#[from]
	EnumSpec(EnumSpec),

	/// Request to return a structured output using the new JSON Schema (Gemini 2.5+).
	#[from]
	JsonSchemaSpec(JsonSchemaSpec),
}

/// The JSON specification for the structured output format.
#[derive(Debug, Clone, From, Serialize, Deserialize)]
pub struct JsonSpec {
	/// The name of the specification. Mostly used by `OpenAI`.
	/// IMPORTANT: With `OpenAI`, this cannot contain any spaces or special characters besides `-` and `_`.
	pub name: String,
	/// The description of the JSON specification. Mostly used by `OpenAI` adapters (future).
	/// NOTE: Currently ignored in the `OpenAI` adapter.
	pub description: Option<String>,

	/// The simplified JSON schema that will be used by the AI provider as JSON schema.
	pub schema: Value,
}

/// The Enum specification for the structured output format.
#[derive(Debug, Clone, From, Serialize, Deserialize)]
pub struct EnumSpec {
	pub mime_type: String,
	pub schema: Value,
}

/// The JSON Schema specification for the structured output format (Gemini 2.5+).
#[derive(Debug, Clone, From, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
	pub schema: Value,
}

/// Constructors
impl JsonSpec {
	/// Create a new `JsonSpec` from name and schema.
	pub fn new(name: impl Into<String>, schema: impl Into<Value>) -> Self {
		Self {
			name: name.into(),
			description: None,
			schema: schema.into(),
		}
	}
}

impl EnumSpec {
	/// Create a new `EnumSpec` from schema.
	pub fn new(schema: impl Into<Value>) -> Self {
		Self {
			mime_type: "text/x.enum".to_string(),
			schema: schema.into(),
		}
	}
}

impl JsonSchemaSpec {
	/// Create a new `JsonSchemaSpec` from schema.
	pub fn new(schema: impl Into<Value>) -> Self {
		Self { schema: schema.into() }
	}
}

/// Setters
impl JsonSpec {
	/// Chainable setter to set the description in a `JsonSpec` construct.
	#[must_use]
	pub fn with_description(mut self, description: impl Into<String>) -> Self {
		self.description = Some(description.into());
		self
	}
}
