use crate::chat::ChatOptions;
use crate::resolver::{
	AuthResolver, IntoAuthResolverFn, IntoModelMapperFn, IntoServiceTargetResolverFn, ModelMapper,
	ServiceTargetResolver,
};
use crate::webc::WebClient;
use crate::{Client, ClientConfig};
use std::sync::Arc;

/// The builder for the `Client` structure.
///
/// - `ClientBuilder::default()`
/// - `Client::builder()`
#[derive(Debug, Default)]
pub struct ClientBuilder {
	web_client: Option<WebClient>,
	config: Option<ClientConfig>,
}

/// Builder methods
impl ClientBuilder {
	/// Create a new `ClientBuilder` with a custom `reqwest::Client`.
	#[must_use]
	pub fn with_reqwest(mut self, reqwest_client: reqwest::Client) -> Self {
		self.web_client = Some(WebClient::from_reqwest_client(reqwest_client));
		self
	}

	/// With a client configuration.
	#[must_use]
	pub fn with_config(mut self, config: ClientConfig) -> Self {
		self.config = Some(config);
		self
	}
}

/// Builder `ClientConfig` passthrough convenient setters.
/// The goal of these functions is to set nested values such as `ClientConfig`.
impl ClientBuilder {
	/// Set the `ChatOptions` for the `ClientConfig` of this `ClientBuilder`.
	/// This will create the `ClientConfig` if it is not present.
	/// Otherwise, it will just set the `client_config.chat_options`.
	#[must_use]
	pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		client_config.chat_options = Some(options);
		self
	}

	/// Set the authentication resolver for the `ClientConfig` of this `ClientBuilder`.
	#[must_use]
	pub fn with_auth_resolver(mut self, auth_resolver: AuthResolver) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		client_config.auth_resolver = Some(auth_resolver);
		self
	}

	/// Set the authentication resolver function for the `ClientConfig` of this `ClientBuilder`.
	#[must_use]
	pub fn with_auth_resolver_fn(mut self, auth_resolver_fn: impl IntoAuthResolverFn) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		let auth_resolver = AuthResolver::from_resolver_fn(auth_resolver_fn);
		client_config.auth_resolver = Some(auth_resolver);
		self
	}

	#[must_use]
	pub fn with_service_target_resolver(mut self, target_resolver: ServiceTargetResolver) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		client_config.service_target_resolver = Some(target_resolver);
		self
	}

	#[must_use]
	pub fn with_service_target_resolver_fn(mut self, target_resolver_fn: impl IntoServiceTargetResolverFn) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		let target_resolver = ServiceTargetResolver::from_resolver_fn(target_resolver_fn);
		client_config.service_target_resolver = Some(target_resolver);
		self
	}

	/// Set the model mapper for the `ClientConfig` of this `ClientBuilder`.
	#[must_use]
	pub fn with_model_mapper(mut self, model_mapper: ModelMapper) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		client_config.model_mapper = Some(model_mapper);
		self
	}

	/// Set the model mapper function for the `ClientConfig` of this `ClientBuilder`.
	#[must_use]
	pub fn with_model_mapper_fn(mut self, model_mapper_fn: impl IntoModelMapperFn) -> Self {
		let client_config = self.config.get_or_insert_with(ClientConfig::default);
		let model_mapper = ModelMapper::from_mapper_fn(model_mapper_fn);
		client_config.model_mapper = Some(model_mapper);
		self
	}
}

impl ClientBuilder {
	/// Build a new immutable `GenAI` client.
	#[must_use]
	pub fn build(self) -> Client {
		let inner = super::ClientInner {
			web_client: self.web_client.unwrap_or_default(),
			config: self.config.unwrap_or_default(),
		};
		Client { inner: Arc::new(inner) }
	}
}

impl Client {
	/// Executes a Veo video generation request.
	pub async fn generate_videos(
		&self,
		model: &str, // e.g., "veo-2.0-generate-001"
		request: crate::chat::VeoGenerateVideosRequest,
	) -> crate::Result<crate::chat::VeoGenerateVideosResponse> {
		self.exec_generate_videos_veo(model, request).await
	}

	/// Executes a request to get the status of a Veo video generation operation.
	pub async fn get_veo_operation_status(
		&self,
		model: &str, // The model used for the original generation, e.g., "veo-2.0-generate-001"
		operation_name: String,
	) -> crate::Result<crate::chat::VeoOperationStatusResponse> {
		self.exec_get_veo_operation_status(model, operation_name).await
	}
}
