//! Types specific to Veo Video Generation.

use crate::ModelIden;
use crate::chat::Usage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request for generating videos using Veo.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VeoGenerateVideosRequest {
	/// The text prompt for the video. When present, the `image` parameter is optional.
	pub prompt: Option<String>,
	/// The image to use as the first frame for the video. When present, the `prompt` parameter is optional.
	/// This should be base64 encoded image data.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub image: Option<VeoImageInput>,
	/// Text string that describes anything you want to _discourage_ the model from generating.
	#[serde(rename = "negativePrompt", skip_serializing_if = "Option::is_none")]
	pub negative_prompt: Option<String>,
	/// Changes the aspect ratio of the generated video. Supported values are `"16:9"` and `"9:16"`.
	/// The default is `"16:9"`.
	#[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
	pub aspect_ratio: Option<String>,
	/// Allow the model to generate videos of people.
	/// Supported values: `"dont_allow"`, `"allow_adult"`, `"allow_all"`.
	#[serde(rename = "personGeneration", skip_serializing_if = "Option::is_none")]
	pub person_generation: Option<String>,
	/// Output videos requested, either `1` or `2`.
	#[serde(rename = "numberOfVideos", skip_serializing_if = "Option::is_none")]
	pub number_of_videos: Option<u8>,
	/// Length of each output video in seconds, between `5` and `8`.
	#[serde(rename = "durationSeconds", skip_serializing_if = "Option::is_none")]
	pub duration_seconds: Option<u8>,
	/// Enable or disable the prompt rewriter. Enabled by default.
	#[serde(rename = "enhancePrompt", skip_serializing_if = "Option::is_none")]
	pub enhance_prompt: Option<bool>,
}

/// Represents an image input for Veo video generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoImageInput {
	#[serde(rename = "bytesBase64Encoded")] // API expects bytesBase64Encoded
	pub image_bytes: Arc<str>,
	#[serde(rename = "mimeType")]
	pub mime_type: String,
}

/// Represents a single generated video from Veo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoGeneratedVideo {
	pub video: VeoVideoUri,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoVideoUri {
	pub uri: String,
}

/// Response from a Veo video generation request (initial long-running operation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoGenerateVideosResponse {
	/// The name of the long-running operation.
	pub operation_name: String,
	/// The model identifier used for this request.
	pub model_iden: ModelIden,
	/// The provider-specific model identifier.
	pub provider_model_iden: ModelIden,
}

/// Response from polling a Veo video generation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoOperationStatusResponse {
	/// Whether the operation is done.
	#[serde(default)]
	pub done: bool,
	/// The name of the operation.
	pub name: String,
	/// The response containing the generated videos, if done.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub response: Option<VeoOperationResult>,
	/// Any error encountered during the operation.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub error: Option<serde_json::Value>,
	/// The model identifier used for this request.
	pub model_iden: ModelIden,
	/// The provider-specific model identifier.
	pub provider_model_iden: ModelIden,
}

/// The actual result of a completed Veo operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoOperationResult {
	#[serde(rename = "generateVideoResponse")]
	pub generate_video_response: VeoGenerateVideoResponseInner,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoGenerateVideoResponseInner {
	#[serde(rename = "generatedSamples")]
	pub generated_samples: Vec<VeoGeneratedVideo>,
}

// Builder methods for VeoGenerateVideosRequest
impl VeoGenerateVideosRequest {
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	#[must_use]
	pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
		self.prompt = Some(prompt.into());
		self
	}

	#[must_use]
	pub fn with_image(mut self, image_bytes: Arc<str>, mime_type: impl Into<String>) -> Self {
		self.image = Some(VeoImageInput {
			image_bytes,
			mime_type: mime_type.into(),
		});
		self
	}

	#[must_use]
	pub fn with_negative_prompt(mut self, neg_prompt: impl Into<String>) -> Self {
		self.negative_prompt = Some(neg_prompt.into());
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
	pub const fn with_number_of_videos(mut self, count: u8) -> Self {
		self.number_of_videos = Some(count);
		self
	}

	#[must_use]
	pub const fn with_duration_seconds(mut self, duration: u8) -> Self {
		self.duration_seconds = Some(duration);
		self
	}

	#[must_use]
	pub const fn with_enhance_prompt(mut self, enhance: bool) -> Self {
		self.enhance_prompt = Some(enhance);
		self
	}
}
