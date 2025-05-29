//! Tests for Gemini Image Generation (Imagen 3 and Conversational)
//!
//! To run these tests:
//! `cargo test --test tests_p_gemini_image -- --nocapture`
//!
//! NOTE: These tests will call the Gemini API and WILL incur costs.
//! NOTE: These tests will write images to the `output/` directory.
//!

use genai::Client;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ImagenGenerateImagesRequest, MessageContent};
use std::fs::{self, File};
use std::io::Write;
use std::sync::Arc;
use std::time::SystemTime;

// --- Helpers ---

fn simple_client() -> Client {
	Client::builder().build()
}

fn get_timestamp_nanos() -> u128 {
	SystemTime::now()
		.duration_since(SystemTime::UNIX_EPOCH)
		.unwrap_or_default()
		.as_nanos()
}

fn save_image_bytes(file_name_prefix: &str, image_idx: usize, image_bytes: &Arc<str>) -> Result<(), String> {
	let timestamp = get_timestamp_nanos();
	let file_name = format!("output/{file_name_prefix}_{timestamp}_{image_idx}.png");

	// Ensure output directory exists
	fs::create_dir_all("output").map_err(|e| format!("Failed to create output directory: {e}"))?;

	let bytes = {
		use base64::Engine;
		let engine = base64::engine::general_purpose::STANDARD;
		engine
			.decode(image_bytes.as_ref())
			.map_err(|e| format!("Failed to decode base64 image: {e}"))?
	};
	let mut file = File::create(&file_name).map_err(|e| format!("Failed to create file {file_name}: {e}"))?;
	file.write_all(&bytes)
		.map_err(|e| format!("Failed to write to file {file_name}: {e}"))?;
	println!("      Saved image to: {file_name}");
	Ok(())
}

// --- Imagen 3 Tests ---

#[tokio::test]
async fn test_imagen_3_generate_simple_ok() -> Result<(), String> {
	let client = simple_client();
	let model_name = "imagen-3.0-generate-002"; // Ensure this model is available and enabled

	println!("->> test_imagen_3_generate_simple_ok - model: {model_name}");

	let request =
		ImagenGenerateImagesRequest::new("A photorealistic image of a futuristic city with flying cars at sunset.")
			.with_number_of_images(1)
			.with_aspect_ratio("16:9");

	let response = client
		.exec_generate_images_imagen(model_name, request)
		.await
		.map_err(|e| format!("API call failed: {e:?}"))?;

	assert!(!response.generated_images.is_empty(), "No images were generated.");
	println!(
		"   Generated {} image(s). First image seed: {:?}",
		response.generated_images.len(),
		response.generated_images.first().and_then(|img| img.seed)
	);

	for (idx, generated_image) in response.generated_images.iter().enumerate() {
		save_image_bytes("imagen3_simple", idx, &generated_image.image_bytes)?;
	}

	Ok(())
}

// --- Conversational Image Generation Tests ---

#[tokio::test]
async fn test_conversational_image_generation_ok() -> Result<(), String> {
	let client = simple_client();
	let model_name = "gemini-2.0-flash-preview-image-generation"; // Ensure this model is available

	println!("->> test_conversational_image_generation_ok - model: {model_name}");

	let user_message = ChatMessage::user("Create an image of a cute cat wearing a tiny hat.");
	let chat_request = ChatRequest::new(vec![user_message]);
	let chat_options = ChatOptions::default().with_response_modalities(vec!["TEXT".to_string(), "IMAGE".to_string()]);

	let response = client
		.exec_chat(model_name, chat_request, Some(&chat_options))
		.await
		.map_err(|e| format!("API call failed: {e:?}"))?;

	assert!(!response.contents.is_empty(), "No content in response.");

	let mut image_found = false;
	for (content_idx, content) in response.contents.iter().enumerate() {
		println!("   Response Content #{content_idx}:");
		match content {
			MessageContent::Text(text) => {
				println!("      Text: {text}");
			}
			MessageContent::Parts(parts) => {
				for (part_idx, part) in parts.iter().enumerate() {
					match part {
						genai::chat::ContentPart::Text(text) => {
							println!("      Part #{part_idx} Text: {text}");
						}
						genai::chat::ContentPart::Image { content_type, source } => {
							println!("      Part #{part_idx} Image - MimeType: {content_type}");
							if let genai::chat::MediaSource::Base64(b64_data) = source {
								save_image_bytes(
									"conversational_img",
									content_idx * 100 + part_idx, // Unique index for multiple images/contents
									b64_data,
								)?;
								image_found = true;
							}
						}
						genai::chat::ContentPart::Document { .. } => {
							println!("      Part #{part_idx} Document - Skipped for image tests");
						}
					}
				}
			}
			_ => {
				println!("      Unexpected content type: {content:?}");
			}
		}
	}

	assert!(image_found, "No image part found in the conversational response.");

	Ok(())
}
