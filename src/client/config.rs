use crate::adapter::AdapterDispatcher;
use crate::chat::ChatOptions;
use crate::client::ServiceTarget;
use crate::resolver::{AuthResolver, ModelMapper, ServiceTargetResolver};
use crate::{Error, ModelIden, Result};

/// The Client configuration used in the configuration builder stage.
#[derive(Debug, Default, Clone)]
pub struct ClientConfig {
	pub(super) auth_resolver: Option<AuthResolver>,
	pub(super) service_target_resolver: Option<ServiceTargetResolver>,
	pub(super) model_mapper: Option<ModelMapper>,
	pub(super) chat_options: Option<ChatOptions>,
}

/// Chainable setters related to the `ClientConfig`.
impl ClientConfig {
	/// Set the `AuthResolver` for the `ClientConfig`.
	/// Note: This will be called before the `service_target_resolver`, and if registered
	///       the `service_target_resolver` will receive this new value.
	#[must_use]
	pub fn with_auth_resolver(mut self, auth_resolver: AuthResolver) -> Self {
		self.auth_resolver = Some(auth_resolver);
		self
	}

	/// Set the `ModelMapper` for the `ClientConfig`.
	/// Note: This will be called before the `service_target_resolver`, and if registered
	///       the `service_target_resolver` will receive this new value.
	#[must_use]
	pub fn with_model_mapper(mut self, model_mapper: ModelMapper) -> Self {
		self.model_mapper = Some(model_mapper);
		self
	}

	/// Set the `ServiceTargetResolver` for this client config.
	///
	/// A `ServiceTargetResolver` is the last step before execution, allowing the users full
	/// control of the resolved Endpoint, `AuthData`, and `ModelIden`.
	#[must_use]
	pub fn with_service_target_resolver(mut self, service_target_resolver: ServiceTargetResolver) -> Self {
		self.service_target_resolver = Some(service_target_resolver);
		self
	}

	/// Set the default chat request options for the `ClientConfig`.
	#[must_use]
	pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
		self.chat_options = Some(options);
		self
	}
}

/// Getters for the fields of `ClientConfig` (as references).
impl ClientConfig {
	/// Get a reference to the `AuthResolver`, if it exists.
	#[must_use]
	pub const fn auth_resolver(&self) -> Option<&AuthResolver> {
		self.auth_resolver.as_ref()
	}

	#[must_use]
	pub const fn service_target_resolver(&self) -> Option<&ServiceTargetResolver> {
		self.service_target_resolver.as_ref()
	}

	/// Get a reference to the `ModelMapper`, if it exists.
	#[must_use]
	pub const fn model_mapper(&self) -> Option<&ModelMapper> {
		self.model_mapper.as_ref()
	}

	/// Get a reference to the `ChatOptions`, if they exist.
	#[must_use]
	pub const fn chat_options(&self) -> Option<&ChatOptions> {
		self.chat_options.as_ref()
	}
}

/// Resolvers
impl ClientConfig {
	pub async fn resolve_service_target(&self, model: ModelIden) -> Result<ServiceTarget> {
		// -- Resolve the Model first
		let model = self
			.model_mapper()
			.map_or_else(
				|| Ok(model.clone()),
				|model_mapper| model_mapper.map_model(model.clone()),
			)
			.map_err(|resolver_error| crate::Error::Resolver {
				model_iden: model.clone(),
				resolver_error,
			})?;

		// -- Get the auth
		let auth = if let Some(auth) = self.auth_resolver() {
			// resolve async which may be async
			auth.resolve(model.clone())
				.await
				.map_err(|err| Error::Resolver {
					model_iden: model.clone(),
					resolver_error: err,
				})?
				// default the resolver resolves to nothing
				.unwrap_or_else(|| AdapterDispatcher::default_auth(model.adapter_kind))
		} else {
			AdapterDispatcher::default_auth(model.adapter_kind)
		};

		// -- Get the default endpoint
		// For now, just get the default endpoint; the `resolve_target` will allow overriding it.
		let endpoint = AdapterDispatcher::default_endpoint(model.adapter_kind);

		// -- Resolve the service_target
		let service_target = ServiceTarget {
			model: model.clone(),
			auth,
			endpoint,
		};
		let service_target = match self.service_target_resolver() {
			Some(service_target_resolver) => {
				service_target_resolver
					.resolve(service_target)
					.await
					.map_err(|resolver_error| Error::Resolver {
						model_iden: model,
						resolver_error,
					})?
			}
			None => service_target,
		};

		Ok(service_target)
	}
}
