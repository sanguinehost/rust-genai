//! Tests for Gemini Image Generation (Imagen 3 and Conversational)
//!
//! To run these tests:
//! `cargo test --test tests_p_gemini_image -- --nocapture`
//!
//! NOTE: These tests will call the Gemini API and WILL incur costs.
//! NOTE: These tests will write images to the `output/` directory.
//!

use genai::Client;
use genai::chat::{BinarySource, ChatMessage, ChatOptions, ChatRequest, ContentPart, ImagenGenerateImagesRequest};
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

// --- Imagen Tests ---

#[tokio::test]
async fn test_imagen_generate_simple_ok() -> Result<(), String> {
	let client = simple_client();
	let model_name = "imagen-4.0-fast-generate-001"; // Imagen 4 Fast ($0.02/image)

	println!("->> test_imagen_generate_simple_ok - model: {model_name}");

	let request =
		ImagenGenerateImagesRequest::new("A highly detailed, photorealistic image of a sleek sports car on a mountain road at golden hour.")
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
		save_image_bytes("imagen_simple", idx, &generated_image.image_bytes)?;
	}

	Ok(())
}

// --- Conversational Image Generation Tests ---

#[tokio::test]
async fn test_conversational_image_generation_ok() -> Result<(), String> {
	let client = simple_client();
	let model_name = "gemini-2.5-flash-image"; // Gemini native image generation model

	println!("->> test_conversational_image_generation_ok - model: {model_name}");

	let user_message = ChatMessage::user("Generate an image of a cute fluffy ginger tabby kitten with bright green eyes, wearing a small blue knitted hat that sits slightly askew on its head. The kitten should be sitting attentively looking at the camera.");
	let chat_request = ChatRequest::new(vec![user_message]);
	let chat_options = ChatOptions::default().with_response_modalities(vec!["Text".to_string(), "Image".to_string()]);

	let response = client
		.exec_chat(model_name, chat_request, Some(&chat_options))
		.await
		.map_err(|e| format!("API call failed: {e:?}"))?;

	// In v0.4.x, response.content is MessageContent (struct), not Vec<MessageContent>
	let content = &response.content;

	// Check if there are any parts
	let parts = content.parts();
	assert!(!parts.is_empty(), "No parts in response content.");

	let mut image_found = false;

	// Iterate through the parts
	for (part_idx, part) in parts.iter().enumerate() {
		println!("   Response Part #{part_idx}:");
		match part {
			ContentPart::Text(text) => {
				println!("      Text: {text}");
			}
			ContentPart::Binary(binary) => {
				println!("      Binary - MimeType: {:?}", binary.content_type);
				if let BinarySource::Base64(b64_data) = &binary.source {
					save_image_bytes(
						"conversational_img",
						part_idx,
						b64_data,
					)?;
					image_found = true;
				}
			}
			ContentPart::ToolCall(_) => {
				println!("      Part #{part_idx} ToolCall - Skipped for image tests");
			}
			ContentPart::ToolResponse(_) => {
				println!("      Part #{part_idx} ToolResponse - Skipped for image tests");
			}
		}
	}

	assert!(image_found, "No image part found in the conversational response.");

	Ok(())
}
