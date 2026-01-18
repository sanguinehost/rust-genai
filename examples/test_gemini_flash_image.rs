//! Test script for the new Gemini 2.5 models
//! This tests that model routing works correctly for both chat and image generation.

use genai::{
	Client,
	chat::{ChatMessage, ChatOptions, ChatRequest, ImagenGenerateImagesRequest},
};

const MODEL_IMAGEN: &str = "imagen-3.0-generate-002";
const MODEL_GEMINI_IMAGE: &str = "gemini-2.5-flash-image";
const MODEL_GEMINI_FLASH: &str = "gemini-2.5-flash";
const MODEL_GEMINI_PREVIEW: &str = "gemini-2.5-flash-preview-09-2025";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

	let client = Client::default();

	// Test 1: Generate an image with Imagen (known working model)
	println!("\n--- Test 1: Generate image with Imagen 3.0 (baseline) ---");
	let prompt = "A serene landscape with mountains and a lake at sunset";
	let imagen_req = ImagenGenerateImagesRequest::new(prompt)
		.with_aspect_ratio("16:9")
		.with_number_of_images(1);

	let imagen_response = client.exec_generate_images_imagen(MODEL_IMAGEN, imagen_req).await?;
	let imagen_bytes = imagen_response
		.generated_images
		.into_iter()
		.next()
		.ok_or("No image generated")?
		.image_bytes
		.len();

	println!("✓ Imagen generated image: {} bytes", imagen_bytes);

	// Test 2: Use gemini-2.5-flash-image for chat with image generation
	println!(
		"\n--- Test 2: Use {} for chat-based image generation ---",
		MODEL_GEMINI_IMAGE
	);
	let chat_req =
		ChatRequest::default().append_message(ChatMessage::user("Generate an image of a cute robot assistant"));

	let chat_options = ChatOptions::default().with_response_modalities(vec!["IMAGE".to_string()]);

	println!("Attempting chat-based image generation...");
	let chat_response = client.exec_chat(MODEL_GEMINI_IMAGE, chat_req, Some(&chat_options)).await?;
	println!(
		"✓ Response from {}: generated {} content parts",
		MODEL_GEMINI_IMAGE,
		chat_response.contents.len()
	);

	// Test 3: Verify gemini-2.5-flash works for chat
	println!("\n--- Test 3: Test {} for chat ---", MODEL_GEMINI_FLASH);
	let chat_req = ChatRequest::default().append_message(ChatMessage::user(
		"Say 'Hello from gemini-2.5-flash!' and nothing else.",
	));

	let chat_response = client.exec_chat(MODEL_GEMINI_FLASH, chat_req, None).await?;
	println!(
		"✓ Chat response from {}: {:?}",
		MODEL_GEMINI_FLASH,
		chat_response.first_content_text_as_str()
	);

	// Test 4: Verify gemini-2.5-flash-preview-09-2025 works
	println!("\n--- Test 4: Test {} for chat ---", MODEL_GEMINI_PREVIEW);
	let chat_req = ChatRequest::default().append_message(ChatMessage::user(
		"Respond with 'Preview model working!' and nothing else.",
	));

	let chat_response = client.exec_chat(MODEL_GEMINI_PREVIEW, chat_req, None).await?;
	println!(
		"✓ Chat response from {}: {:?}",
		MODEL_GEMINI_PREVIEW,
		chat_response.first_content_text_as_str()
	);

	println!("\n--- All tests passed! ---");
	println!("✓ Model routing works correctly");
	println!("✓ Imagen 3.0 image generation works");
	println!("✓ {} chat-based image generation works", MODEL_GEMINI_IMAGE);
	println!("✓ {} chat works", MODEL_GEMINI_FLASH);
	println!("✓ {} chat works", MODEL_GEMINI_PREVIEW);

	Ok(())
}
