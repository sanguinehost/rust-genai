//! Types specific to Imagen 3 Image Generation.

use crate::ModelIden;
use crate::chat::Usage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request for generating images using Imagen 3.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImagenGenerateImagesRequest {
	/// The prompt describing the image(s) to generate.
	pub prompt: String,
	/// The number of images to generate. Must be between 1 and 4.
	/// Defaults to 4 if not specified by the API, but we might default to 1.
	#[serde(rename = "sampleCount", skip_serializing_if = "Option::is_none")]
	pub number_of_images: Option<u8>,
	/// The desired aspect ratio of the generated images.
	/// Supported values: "1:1", "3:4", "4:3", "9:16", "16:9".
	/// Defaults to "1:1".
	#[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
	pub aspect_ratio: Option<String>,
	/// Controls the generation of people in images.
	/// Supported values: `"dont_allow"`, `"allow_adult"`, `"allow_all"`.
	/// Defaults to `"allow_adult"`.
	#[serde(rename = "personGeneration", skip_serializing_if = "Option::is_none")]
	pub person_generation: Option<String>,
	/// A negative prompt to guide the model away from certain concepts.
	#[serde(rename = "negativePrompt", skip_serializing_if = "Option::is_none")]
	pub negative_prompt: Option<String>,
	/// Seed for deterministic image generation.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub seed: Option<i64>,
	// Potentially other parameters like `language` if supported/needed.
}

/// Represents a single generated image from Imagen 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagenGeneratedImage {
	/// Base64 encoded image data.
	#[serde(rename = "bytesBase64Encoded")]
	pub image_bytes: Arc<str>,
	/// The seed used to generate this specific image.
	pub seed: Option<i64>,
	/// Reason why image generation finished (e.g., "DONE", `"SAFETY_FILTER"`).
	#[serde(rename = "finishReason")]
	pub finish_reason: Option<String>,
	// Potentially `error` or `status` fields if the API returns them per image.
}

/// Response from an Imagen 3 image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagenGenerateImagesResponse {
	/// List of generated images.
	#[serde(rename = "predictions")]
	// Assuming the images are in a "predictions" field as per Google's AI Platform predict pattern
	pub generated_images: Vec<ImagenGeneratedImage>,
	/// Usage statistics, if provided by the API.
	/// This might not be directly available or might be part of a different structure.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// The model identifier used for this request.
	pub model_iden: ModelIden,
	/// The provider-specific model identifier.
	pub provider_model_iden: ModelIden,
	// Deployment resource name, if applicable and returned
	// pub deployment_resource_name: Option<String>,
}

// Builder methods for ImagenGenerateImagesRequest
impl ImagenGenerateImagesRequest {
	pub fn new(prompt: impl Into<String>) -> Self {
		Self {
			prompt: prompt.into(),
			..Default::default()
		}
	}

	#[must_use]
	pub const fn with_number_of_images(mut self, count: u8) -> Self {
		self.number_of_images = Some(count);
		self
	}

	#[must_use]
	pub fn with_aspect_ratio(mut self, ratio: impl Into<String>) -> Self {
		self.aspect_ratio = Some(ratio.into());
		self
	}

	#[must_use]
	pub fn with_person_generation(mut self, setting: impl Into<String>) -> Self {
		self.person_generation = Some(setting.into());
		self
	}

	#[must_use]
	pub fn with_negative_prompt(mut self, neg_prompt: impl Into<String>) -> Self {
		self.negative_prompt = Some(neg_prompt.into());
		self
	}

	#[must_use]
	pub const fn with_seed(mut self, seed_val: i64) -> Self {
		self.seed = Some(seed_val);
		self
	}
}
