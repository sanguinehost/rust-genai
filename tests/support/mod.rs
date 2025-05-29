//! Some support utilities for the tests
//! Note: Must be imported in each test file

#![allow(unused)] // For test support

// region:    --- Modules

mod asserts;
mod data;
mod helpers;
mod seeders;

pub use asserts::*;
pub use helpers::*;
pub use seeders::*;

pub mod common_tests;

pub type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

// endregion: --- Modules

// region:    --- Common GenAI Constants

// NOTE: To run `anthropic` tests, an `ANTHROPIC_API_KEY` environment variable must be set.
//       It can be a temp one, as it should only do one request, which should be well within the free tier.
pub const MODEL_ANTHROPIC_CLAUDE_3_OPUS: &str = "anthropic/claude-3-opus-20240229";
pub const MODEL_ANTHROPIC_CLAUDE_3_SONNET: &str = "anthropic/claude-3-sonnet-20240229";
pub const MODEL_ANTHROPIC_CLAUDE_3_HAIKU: &str = "anthropic/claude-3-haiku-20240307";

// endregion: --- Common GenAI Constants

// region:    --- Common Clients
use genai::Client;
use genai::Result as GenaiResult;
use genai::adapter::AdapterKind; // genai's Result, aliased to avoid conflict

pub fn common_client_gemini() -> Client {
	// Client::builder()
	//     .with_adapter_kind(AdapterKind::Gemini) // This method does not exist on ClientBuilder
	//     .try_build()
	// For now, assume default builder is sufficient or custom config will be added if needed by other tests.
	// Adapters are typically resolved from the model string.
	Client::builder().build()
}

pub fn common_client_from_model(model_name_str: &str) -> Client {
	// Very basic dispatch for now, can be expanded.
	// This assumes model_name_str is like "adapter/model_name", e.g., "gemini/gemini-2.5-flash"
	if model_name_str.starts_with("gemini/")
		|| model_name_str.starts_with("gemini-2.5")
		|| model_name_str.starts_with("gemini-1.5")
	{
		common_client_gemini()
	} else if model_name_str.starts_with("anthropic/") {
		// Assuming a similar helper for anthropic, or direct build
		Client::builder().build() // Placeholder, adapt if anthropic needs specific setup
	} else if model_name_str.starts_with("openai/")
		|| model_name_str.starts_with("gpt-")
		|| model_name_str.starts_with("cohere/")
		|| model_name_str.starts_with("command-")
	{
		Client::builder().build() // Placeholder
	} else {
		// Default or error
		println!("Warning: common_client_from_model falling back to default client for model: {model_name_str}");
		Client::builder().build()
	}
}

// endregion: --- Common Clients
