use crate::Result;
use crate::adapter::anthropic::AnthropicAdapter;
use crate::adapter::cohere::CohereAdapter;
use crate::adapter::deepseek::{self, DeepSeekAdapter};
use crate::adapter::gemini::GeminiAdapter;
use crate::adapter::groq::{self, GroqAdapter};
use crate::adapter::openai::OpenAIAdapter;
use crate::adapter::xai::XaiAdapter;
use derive_more::Display;
use serde::{Deserialize, Serialize};

/// `AdapterKind` is an enum that represents the different types of adapters that can be used to interact with the API.
#[derive(Debug, Clone, Copy, Display, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AdapterKind {
	/// Main adapter type for the `OpenAI` service.
	OpenAI,
	/// Used for the Ollama adapter (currently, localhost only). Behind the scenes, it uses the `OpenAI` adapter logic.
	Ollama,
	/// Used for the Anthropic adapter.
	Anthropic,
	/// Used for the Cohere adapter.
	Cohere,
	/// Used for the Gemini adapter.
	Gemini,
	/// Used for the Groq adapter. Behind the scenes, it uses the `OpenAI` adapter logic with the necessary Groq differences (e.g., usage).
	Groq,
	/// For xAI
	Xai,
	/// For `DeepSeek`
	DeepSeek,
	/// For native llama.cpp integration with local models
	#[cfg(feature = "llamacpp")]
	LlamaCpp,
	// Note: Variants will probably be suffixed
	// AnthropicBedrock,
}

/// Serialization implementations
impl AdapterKind {
	/// Serialize to a static str
	#[must_use]
	pub const fn as_str(&self) -> &'static str {
		match self {
			Self::OpenAI => "OpenAI",
			Self::Ollama => "Ollama",
			Self::Anthropic => "Anthropic",
			Self::Cohere => "Cohere",
			Self::Gemini => "Gemini",
			Self::Groq => "Groq",
			Self::Xai => "xAi",
			Self::DeepSeek => "DeepSeek",
			#[cfg(feature = "llamacpp")]
			Self::LlamaCpp => "LlamaCpp",
		}
	}

	/// Serialize to a static str
	#[must_use]
	pub const fn as_lower_str(&self) -> &'static str {
		match self {
			Self::OpenAI => "openai",
			Self::Ollama => "ollama",
			Self::Anthropic => "anthropic",
			Self::Cohere => "cohere",
			Self::Gemini => "gemini",
			Self::Groq => "groq",
			Self::Xai => "xai",
			Self::DeepSeek => "deepseek",
			#[cfg(feature = "llamacpp")]
			Self::LlamaCpp => "llamacpp",
		}
	}
}

/// Utilities
impl AdapterKind {
	/// Get the default key environment variable name for the adapter kind.
	#[must_use]
	pub const fn default_key_env_name(&self) -> Option<&'static str> {
		match self {
			Self::OpenAI => Some(OpenAIAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Anthropic => Some(AnthropicAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Cohere => Some(CohereAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Gemini => Some(GeminiAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Groq => Some(GroqAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Xai => Some(XaiAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::DeepSeek => Some(DeepSeekAdapter::API_KEY_DEFAULT_ENV_NAME),
			Self::Ollama => None,
			#[cfg(feature = "llamacpp")]
			Self::LlamaCpp => None,
		}
	}
}

/// From Model implementations
impl AdapterKind {
	/// This is a default static mapping from model names to `AdapterKind`.
	///
	/// When more control is needed, the `ServiceTypeResolver` can be used
	/// to map a model name to any adapter and endpoint.
	///
	///  - `OpenAI`     - `starts_with` "gpt", "o3", "o1", "chatgpt"
	///  - `Anthropic`  - `starts_with` "claude"
	///  - `Cohere`     - `starts_with` "command"
	///  - `Gemini`     - `starts_with` "gemini" or "imagen"
	///  - `Groq`       - model in Groq models
	///  - `DeepSeek`   - model in `DeepSeek` models (deepseek.com)
	///  - `Ollama`     - For anything else
	///
	/// Note: At this point, this will never fail as the fallback is the Ollama adapter.
	///       This might change in the future, hence the `Result` return type.
	pub fn from_model(model: &str) -> Result<Self> {
		if model.starts_with("gpt")
			|| model.starts_with("o3")
			|| model.starts_with("o4")
			|| model.starts_with("o1")
			|| model.starts_with("chatgpt")
		{
			return Ok(Self::OpenAI);
		} else if model.starts_with("claude") {
			return Ok(Self::Anthropic);
		} else if model.starts_with("command") {
			return Ok(Self::Cohere);
		} else if model.starts_with("gemini") || model.starts_with("imagen") || model.starts_with("veo") {
			return Ok(Self::Gemini);
		} else if model.starts_with("grok") {
			return Ok(Self::Xai);
		} else if deepseek::MODELS.contains(&model) {
			return Ok(Self::DeepSeek);
		} else if groq::MODELS.contains(&model) {
			return Ok(Self::Groq);
		}
		#[cfg(feature = "llamacpp")]
		{
			if model.starts_with("local/") 
				|| model.starts_with("llama") 
				|| model.starts_with("mistral") 
				|| model.starts_with("phi") 
				|| model.ends_with(".gguf") 
			{
				return Ok(Self::LlamaCpp);
			}
		}
		
		// For now, fallback to Ollama
		Ok(Self::Ollama)
	}
}
