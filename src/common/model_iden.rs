use serde::{Deserialize, Serialize};

use crate::ModelName;
use crate::adapter::AdapterKind;
use derive_more::Display;

/// Holds the adapter kind and model name in an efficient, clonable way.
///
/// This struct represents the association between an adapter kind
/// and a model name, allowing for easy conversion and instantiation.
#[derive(Clone, Debug, Serialize, Deserialize, Display)]
#[display("{model_name} (adapter: {adapter_kind})")]
pub struct ModelIden {
	/// The adapter kind.
	pub adapter_kind: AdapterKind,
	/// The model name.
	pub model_name: ModelName,
}

/// Contructor
impl ModelIden {
	/// Create a new `ModelIden` with the given adapter kind and model name.
	pub fn new(adapter_kind: AdapterKind, model_name: impl Into<ModelName>) -> Self {
		Self {
			adapter_kind,
			model_name: model_name.into(),
		}
	}
}

impl ModelIden {
	/// Creates a new `ModelIden` with the specified name, or clones the existing one if the name is the same.
	#[must_use]
	pub fn from_name<T>(&self, new_name: T) -> Self
	where
		T: AsRef<str> + Into<String>,
	{
		let new_name_ref = new_name.as_ref();

		// If the names are the same, just return a clone
		if &*self.model_name == new_name_ref {
			self.clone()
		} else {
			let model_name = new_name.into();
			Self {
				adapter_kind: self.adapter_kind,
				model_name: model_name.into(),
			}
		}
	}

	/// Creates a new `ModelIden` with the specified name, or clones the existing one if the name is the same.
	/// NOTE: Might be deprecated in favor of [`from_name`]
	#[must_use]
	pub fn from_optional_name(&self, new_name: Option<String>) -> Self {
		new_name.map_or_else(|| self.clone(), |new_name| self.from_name(new_name))
	}

	#[deprecated(note = "use from_optional_name")]
	#[must_use]
	pub fn with_name_or_clone(&self, new_name: Option<String>) -> Self {
		self.from_optional_name(new_name)
	}
}

impl<T> From<(AdapterKind, T)> for ModelIden
where
	T: Into<ModelName>,
{
	fn from((adapter_kind, model_name): (AdapterKind, T)) -> Self {
		Self {
			adapter_kind,
			model_name: model_name.into(),
		}
	}
}
