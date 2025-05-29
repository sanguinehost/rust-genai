use std::sync::Arc;

/// A construct to store the endpoint of a service.
/// It is designed to be efficiently clonable.
/// For now, it supports only `base_url`, but it may later have other URLs per "service name".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Endpoint {
	inner: Arc<str>,
}

/// Constructors
impl Endpoint {
	#[must_use]
	pub fn from_static(url: &'static str) -> Self {
		Self { inner: Arc::from(url) }
	}

	pub fn from_owned(url: impl Into<Arc<str>>) -> Self {
		Self { inner: url.into() }
	}
}

/// Getters
impl Endpoint {
	#[must_use]
	pub fn base_url(&self) -> &str {
		&self.inner
	}
}
