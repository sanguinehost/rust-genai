//! LlamaCpp adapter for native local model inference using llama.cpp bindings.
//!
//! This adapter provides native integration with llama.cpp for running local models
//! without requiring an HTTP server. It supports model caching, streaming, and 
//! full control over generation parameters.

#[cfg(feature = "llamacpp")]
mod adapter_impl;
#[cfg(feature = "llamacpp")]
pub mod model_manager;
#[cfg(feature = "llamacpp")]
mod streamer;
#[cfg(feature = "llamacpp")]
pub mod schema_to_grammar;
#[cfg(feature = "llamacpp")]
pub mod tool_templates;
#[cfg(feature = "llamacpp")]
pub mod tool_parser;

#[cfg(feature = "llamacpp")]
pub use adapter_impl::LlamaCppAdapter;
