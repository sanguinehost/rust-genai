//! ModelManager handles loading and caching of llama.cpp models to avoid
//! reloading the same model multiple times.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::OnceCell;

use llama_cpp_2::{
	llama_backend::LlamaBackend,
	model::params::LlamaModelParams,
	model::{LlamaChatTemplate, LlamaModel},
};

use crate::{Error, Result};

/// A cached loaded model with its associated metadata
#[derive(Debug, Clone)]
pub struct LoadedModel {
	pub model: Arc<LlamaModel>,
	pub chat_template: Option<LlamaChatTemplate>,
	pub model_path: PathBuf,
}

/// Singleton ModelManager that handles loading and caching llama.cpp models
#[derive(Debug)]
pub struct ModelManager {
	backend: Arc<LlamaBackend>,
	models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
}

static MODEL_MANAGER: OnceCell<Arc<ModelManager>> = OnceCell::const_new();

impl ModelManager {
	/// Get or initialize the global ModelManager instance
	pub async fn instance() -> Result<Arc<ModelManager>> {
		MODEL_MANAGER
			.get_or_try_init(|| async {
				let backend = LlamaBackend::init()
					.map_err(|e| Error::Internal(format!("Failed to initialize llama backend: {e}")))?;

				Ok(Arc::new(ModelManager {
					backend: Arc::new(backend),
					models: Arc::new(RwLock::new(HashMap::new())),
				}))
			})
			.await
			.cloned()
	}

	/// Get a reference to the backend
	pub fn backend(&self) -> &LlamaBackend {
		&self.backend
	}

	/// Load a model from the specified path, using cache if already loaded
	pub async fn load_model(&self, model_path: &Path) -> Result<Arc<LoadedModel>> {
		let path_str = model_path.to_string_lossy().to_string();

		// Check cache first
		{
			let models = self
				.models
				.read()
				.map_err(|e| Error::Internal(format!("Failed to read models cache: {e}")))?;

			if let Some(cached_model) = models.get(&path_str) {
				return Ok(cached_model.clone());
			}
		}

		// Load model if not cached
		let loaded_model = self.load_model_internal(model_path).await?;

		// Cache the loaded model
		{
			let mut models = self
				.models
				.write()
				.map_err(|e| Error::Internal(format!("Failed to write to models cache: {e}")))?;

			let loaded_model_arc = Arc::new(loaded_model);
			models.insert(path_str, loaded_model_arc.clone());
			Ok(loaded_model_arc)
		}
	}

	/// Internal method to actually load the model from disk
	async fn load_model_internal(&self, model_path: &Path) -> Result<LoadedModel> {
		if !model_path.exists() {
			return Err(Error::Internal(format!(
				"Model file not found: {}",
				model_path.display()
			)));
		}

		// Prepare model parameters
		let model_params = LlamaModelParams::default();

		// Load model on blocking task to avoid blocking async runtime
		let backend = self.backend.clone();
		let model_path = model_path.to_owned();
		// Load model directly (llama.cpp types are not Send-safe)
		let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
			.map_err(|e| Error::Internal(format!("Failed to load model: {e}")))?;

		// Try to get the default chat template
		let chat_template = model.chat_template(None).ok();

		Ok(LoadedModel {
			model: Arc::new(model),
			chat_template,
			model_path: model_path,
		})
	}

	/// Clear all cached models to free memory
	pub fn clear_cache(&self) -> Result<()> {
		let mut models = self
			.models
			.write()
			.map_err(|e| Error::Internal(format!("Failed to clear models cache: {e}")))?;

		models.clear();
		Ok(())
	}

	/// Get the number of cached models
	pub fn cache_size(&self) -> Result<usize> {
		let models = self
			.models
			.read()
			.map_err(|e| Error::Internal(format!("Failed to read models cache: {e}")))?;

		Ok(models.len())
	}
}

/// Resolve model path from various formats:
/// - Absolute path: "/path/to/model.gguf"  
/// - Relative path: "model.gguf" (relative to endpoint base)
/// - Model name: "llama-3.2" (looks in standard directories)
pub fn resolve_model_path(model_name: &str, base_path: Option<&str>) -> Result<PathBuf> {
	let path = Path::new(model_name);

	// If it's already an absolute path, use it directly
	if path.is_absolute() {
		return Ok(path.to_owned());
	}

	// If it ends with .gguf, it's likely a filename
	if model_name.ends_with(".gguf") {
		if let Some(base) = base_path {
			let full_path = Path::new(base).join(model_name);
			if full_path.exists() {
				return Ok(full_path);
			}
		}

		// Try common model directories
		let common_dirs = ["~/.cache/huggingface/hub", "~/.cache/llama-models", "./models"];

		for dir in &common_dirs {
			let expanded_dir = shellexpand::tilde(dir);
			let full_path = Path::new(expanded_dir.as_ref()).join(model_name);
			if full_path.exists() {
				return Ok(full_path);
			}
		}
	}

	// If we have a base path, try relative to it
	if let Some(base) = base_path {
		let full_path = Path::new(base).join(model_name);
		if full_path.exists() {
			return Ok(full_path);
		}
	}

	Err(Error::Internal(format!(
		"Could not resolve model path for: {model_name}"
	)))
}
