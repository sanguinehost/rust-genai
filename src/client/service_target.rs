use crate::ModelIden;
use crate::resolver::{AuthData, Endpoint};

/// A `ServiceTarget` represents the destination and necessary details for making a service call.
///
/// This structure contains:
/// - `endpoint`: The specific service endpoint to be contacted.
/// - `auth`: The authentication data required to access the service.
/// - `model`: The identifier of the model or resource associated with the service call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub struct ServiceTarget {
	pub endpoint: Endpoint,
	pub auth: AuthData,
	pub model: ModelIden,
}
