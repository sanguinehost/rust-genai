//! E2E Integration Tests for Image Generation
//!
//! To run these tests with a real Gemini API key:
//! ```bash
//! GEMINI_API_KEY=your-actual-api-key cargo test --test test_e2e_image_generation -- --nocapture
//! ```
//!
//! WARNING: These tests will make real API calls and incur costs!
//! The generated images will be saved to the `output/` directory.

use genai::Client;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ImagenGenerateImagesRequest, MessageContent};
use std::fs::File;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn save_image(filename: &str, base64_data: &str) -> Result<(), Box<dyn std::error::Error>> {
	let bytes = {
		use base64::Engine;
		let engine = base64::engine::general_purpose::STANDARD;
		engine.decode(base64_data)?
	};

	let mut file = File::create(filename)?;
	file.write_all(&bytes)?;
	println!("✅ Saved image to: {filename}");
	Ok(())
}

fn get_timestamp() -> u64 {
	SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

#[tokio::test]
#[ignore] // Ignore by default since it makes real API calls
async fn test_imagen3_simple_generation() -> Result<(), Box<dyn std::error::Error>> {
	let client = Client::builder().build();

	println!("\n🎨 Testing Imagen 3 Image Generation...");

	let request = ImagenGenerateImagesRequest::new(
		"A serene Japanese garden with cherry blossoms, koi pond, and a traditional wooden bridge at sunset",
	)
	.with_number_of_images(1)
	.with_aspect_ratio("16:9");

	let response = client.exec_generate_images_imagen("imagen-3.0-generate-002", request).await?;

	assert!(!response.generated_images.is_empty(), "No images generated");

	for (idx, image) in response.generated_images.iter().enumerate() {
		let filename = format!("output/imagen3_test_{}_{}.png", get_timestamp(), idx);
		save_image(&filename, &image.image_bytes)?;

		if let Some(reason) = &image.finish_reason {
			println!("  Finish reason: {reason}");
		}
	}

	println!("✅ Imagen 3 test completed successfully!");
	Ok(())
}

#[tokio::test]
#[ignore] // Ignore by default since it makes real API calls
async fn test_conversational_image_generation() -> Result<(), Box<dyn std::error::Error>> {
	let client = Client::builder().build();

	println!("\n💬 Testing Conversational Image Generation...");

	let message = ChatMessage::user(
		"Generate an image of a futuristic cityscape with flying vehicles and neon lights. Make it cyberpunk style.",
	);
	let request = ChatRequest::new(vec![message]);
	let options = ChatOptions::default().with_response_modalities(vec!["TEXT".to_string(), "IMAGE".to_string()]);

	let response = client
		.exec_chat("gemini-2.0-flash-preview-image-generation", request, Some(&options))
		.await?;

	assert!(!response.contents.is_empty(), "No content in response");

	let mut image_count = 0;
	for (content_idx, content) in response.contents.iter().enumerate() {
		match content {
			MessageContent::Text(text) => {
				println!("  Response text: {text}");
			}
			MessageContent::Parts(parts) => {
				for part in parts {
					match part {
						genai::chat::ContentPart::Text(text) => {
							println!("  Text part: {text}");
						}
						genai::chat::ContentPart::Image { source, .. } => {
							if let genai::chat::MediaSource::Base64(data) = source {
								let filename = format!(
									"output/conversational_test_{}_{}_{}.png",
									get_timestamp(),
									content_idx,
									image_count
								);
								save_image(&filename, data)?;
								image_count += 1;
							}
						}
						genai::chat::ContentPart::Document { .. } => {
							// Skip documents in image generation test
						}
					}
				}
			}
			_ => {}
		}
	}

	assert!(image_count > 0, "No images found in response");
	println!("✅ Conversational image generation test completed successfully!");
	Ok(())
}

#[tokio::test]
async fn test_quick_validation() {
	// This test just validates the setup without making API calls
	println!("\n🔍 Quick validation test...");

	// Check if output directory exists
	assert!(std::path::Path::new("output").exists(), "Output directory should exist");

	// Check if we can create a client
	let client = Client::builder().build();

	// Verify model names are recognized
	let imagen_model = client.default_model("imagen-3.0-generate-002");
	assert!(imagen_model.is_ok(), "Imagen model should be recognized");

	let gemini_model = client.default_model("gemini-2.0-flash-preview-image-generation");
	assert!(gemini_model.is_ok(), "Gemini model should be recognized");

	println!("✅ Quick validation passed!");
}

// Helper function to run all tests
#[tokio::test]
#[ignore]
async fn run_all_e2e_tests() -> Result<(), Box<dyn std::error::Error>> {
	println!("\n🚀 Running ALL E2E Image Generation Tests...\n");

	// Run validation first
	println!("\n🔍 Quick validation test...");
	assert!(std::path::Path::new("output").exists(), "Output directory should exist");
	let client = Client::builder().build();
	let _ = client.default_model("imagen-3.0-generate-002")?;
	let _ = client.default_model("gemini-2.0-flash-preview-image-generation")?;
	println!("✅ Quick validation passed!");

	// Check for API key
	if std::env::var("GEMINI_API_KEY").is_err() {
		println!("\n⚠️  GEMINI_API_KEY not set. Skipping real API tests.");
		println!("   To run the full E2E tests, set your API key:");
		println!(
			"   GEMINI_API_KEY=your-key cargo test --test test_e2e_image_generation run_all_e2e_tests -- --ignored --nocapture"
		);
		return Ok(());
	}

	// Run the actual tests
	println!("\n🔑 API key found. Running real API tests...\n");

	// Run Imagen 3 test
	match test_imagen3_generation_impl().await {
		Ok(()) => println!("✅ Imagen 3 test passed!"),
		Err(e) => println!("❌ Imagen 3 test failed: {e}"),
	}

	println!();

	// Run conversational test
	match test_conversational_generation_impl().await {
		Ok(()) => println!("✅ Conversational test passed!"),
		Err(e) => println!("❌ Conversational test failed: {e}"),
	}

	println!("\n🎉 All E2E tests completed!");
	Ok(())
}

// Implementation functions that can be called from the runner
async fn test_imagen3_generation_impl() -> Result<(), Box<dyn std::error::Error>> {
	let client = Client::builder().build();

	println!("\n🎨 Testing Imagen 3 Image Generation...");

	let request = ImagenGenerateImagesRequest::new(
		"A serene Japanese garden with cherry blossoms, koi pond, and a traditional wooden bridge at sunset",
	)
	.with_number_of_images(1)
	.with_aspect_ratio("16:9");

	let response = client.exec_generate_images_imagen("imagen-3.0-generate-002", request).await?;

	assert!(!response.generated_images.is_empty(), "No images generated");

	for (idx, image) in response.generated_images.iter().enumerate() {
		let filename = format!("output/imagen3_test_{}_{}.png", get_timestamp(), idx);
		save_image(&filename, &image.image_bytes)?;

		if let Some(reason) = &image.finish_reason {
			println!("  Finish reason: {reason}");
		}
	}

	Ok(())
}

async fn test_conversational_generation_impl() -> Result<(), Box<dyn std::error::Error>> {
	let client = Client::builder().build();

	println!("\n💬 Testing Conversational Image Generation...");

	let message = ChatMessage::user(
		"Generate an image of a futuristic cityscape with flying vehicles and neon lights. Make it cyberpunk style.",
	);
	let request = ChatRequest::new(vec![message]);
	let options = ChatOptions::default().with_response_modalities(vec!["TEXT".to_string(), "IMAGE".to_string()]);

	let response = client
		.exec_chat("gemini-2.0-flash-preview-image-generation", request, Some(&options))
		.await?;

	assert!(!response.contents.is_empty(), "No content in response");

	let mut image_count = 0;
	for (content_idx, content) in response.contents.iter().enumerate() {
		match content {
			MessageContent::Text(text) => {
				println!("  Response text: {text}");
			}
			MessageContent::Parts(parts) => {
				for part in parts {
					match part {
						genai::chat::ContentPart::Text(text) => {
							println!("  Text part: {text}");
						}
						genai::chat::ContentPart::Image { source, .. } => {
							if let genai::chat::MediaSource::Base64(data) = source {
								let filename = format!(
									"output/conversational_test_{}_{}_{}.png",
									get_timestamp(),
									content_idx,
									image_count
								);
								save_image(&filename, data)?;
								image_count += 1;
							}
						}
						genai::chat::ContentPart::Document { .. } => {
							// Skip documents in image generation test
						}
					}
				}
			}
			_ => {}
		}
	}

	assert!(image_count > 0, "No images found in response");
	Ok(())
}
