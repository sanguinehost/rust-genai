//! Tests for Gemini Video Generation (Veo)
//!
//! To run these tests:
//! `cargo test --test tests_p_gemini_video -- --nocapture`
//!
//! NOTE: These tests will call the Gemini API and WILL incur costs.
//! NOTE: These tests will write videos to the `output/` directory.
//!

use tracing_subscriber::{EnvFilter, FmtSubscriber};

// Helper to initialize tracing, call this at the beginning of each test
fn init_tracing() {
	let subscriber = FmtSubscriber::builder()
		.with_env_filter(
			EnvFilter::from_default_env().add_directive("genai=trace".parse().expect("Invalid tracing directive")),
		)
		.with_test_writer() // Writes to the test output buffer, visible with --nocapture
		.try_init();
	// Ignore error if already initialized by another test
	let _ = subscriber;
}
use genai::Client;
use genai::chat::{ImagenGenerateImagesRequest, VeoGenerateVideosRequest};
use std::fs::{self, File};
use std::io::Write;
use std::time::{Duration, SystemTime};

// --- Constants ---
const MODEL_VEO: &str = "veo-2.0-generate-001";
const MODEL_IMAGEN: &str = "imagen-3.0-generate-002"; // For image-to-video tests

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

async fn save_video_from_uri(
	file_name_prefix: &str,
	video_idx: usize,
	video_uri: &str,
	api_key: &str,
) -> Result<(), String> {
	let timestamp = get_timestamp_nanos();
	let file_name = format!("output/{file_name_prefix}_{timestamp}_{video_idx}.mp4");

	// Ensure output directory exists
	fs::create_dir_all("output").map_err(|e| format!("Failed to create output directory: {e}"))?;

	let download_url = if video_uri.contains('?') {
		format!("{video_uri}&key={api_key}")
	} else {
		format!("{video_uri}?key={api_key}")
	};

	println!("      Downloading video from: {download_url}");
	let response = reqwest::get(&download_url) // Use the modified download_url
		.await
		.map_err(|e| format!("Failed to download video: {e}"))?;

	if !response.status().is_success() {
		return Err(format!(
			"Failed to download video from {download_url}, status: {}",
			response.status()
		));
	}

	let video_bytes = response.bytes().await.map_err(|e| format!("Failed to read video bytes: {e}"))?;

	let mut file = File::create(&file_name).map_err(|e| format!("Failed to create file {file_name}: {e}"))?;
	file.write_all(&video_bytes)
		.map_err(|e| format!("Failed to write to file {file_name}: {e}"))?;
	println!("      Saved video to: {file_name}");
	Ok(())
}

// --- Veo Tests ---

#[tokio::test]
async fn test_veo_generate_text_to_video_ok() -> Result<(), String> {
	const MAX_ATTEMPTS: u32 = 30; // 30 attempts * 10 seconds = 5 minutes max wait

	init_tracing();
	let client = simple_client();
	let model_name = MODEL_VEO;
	let api_key =
		std::env::var("GEMINI_API_KEY").map_err(|_| "GEMINI_API_KEY environment variable not set".to_string())?;

	println!("->> test_veo_generate_text_to_video_ok - model: {model_name}");

	let prompt = "A majestic eagle soaring over a snow-capped mountain range at sunrise.";
	let veo_req = VeoGenerateVideosRequest::new()
		.with_prompt(prompt)
		.with_aspect_ratio("16:9")
		.with_person_generation("dont_allow")
		.with_number_of_videos(1)
		.with_duration_seconds(5);

	let initial_res = client
		.exec_generate_videos_veo(model_name, veo_req)
		.await
		.map_err(|e| format!("Initial API call failed: {e:?}"))?;

	println!("   Initial Veo operation name: {}", initial_res.operation_name);

	// Poll for operation completion (Veo can take a few minutes)
	let mut operation_status;
	let mut attempts = 0;
	loop {
		operation_status = client
			.exec_get_veo_operation_status(model_name, initial_res.operation_name.clone())
			.await
			.map_err(|e| format!("Failed to get operation status during polling: {e:?}"))?;

		if operation_status.done {
			break;
		}

		attempts += 1;
		if attempts >= MAX_ATTEMPTS {
			break;
		}

		println!("   Operation not done yet. Waiting 10 seconds... (Attempt {attempts}/{MAX_ATTEMPTS})");
		tokio::time::sleep(Duration::from_secs(10)).await;
	}

	assert!(
		operation_status.done,
		"Veo video generation operation timed out or failed to complete."
	);

	if let Some(error) = operation_status.error {
		return Err(format!("Video generation failed with error: {error:?}"));
	}

	let result = operation_status
		.response
		.ok_or("No response found in completed operation status.")?;

	assert!(
		!result.generate_video_response.generated_samples.is_empty(),
		"No videos were generated."
	);
	println!(
		"   Generated {} video(s).",
		result.generate_video_response.generated_samples.len()
	);

	for (idx, generated_video) in result.generate_video_response.generated_samples.iter().enumerate() {
		save_video_from_uri("veo_text_to_video", idx, &generated_video.video.uri, &api_key).await?;
	}

	Ok(())
}

#[tokio::test]
async fn test_veo_generate_image_to_video_ok() -> Result<(), String> {
	const MAX_ATTEMPTS: u32 = 30; // 30 attempts * 10 seconds = 5 minutes max wait

	init_tracing();
	let client = simple_client();
	let veo_model_name = MODEL_VEO;
	let imagen_model_name = MODEL_IMAGEN;
	let api_key =
		std::env::var("GEMINI_API_KEY").map_err(|_| "GEMINI_API_KEY environment variable not set".to_string())?;

	println!(
		"->> test_veo_generate_image_to_video_ok - Veo model: {veo_model_name}, Imagen model: {imagen_model_name}"
	);

	// 1. Generate an image using Imagen 3 first
	let image_prompt = "A close-up of a fluffy white cloud in a blue sky.";
	let imagen_req = ImagenGenerateImagesRequest::new(image_prompt)
		.with_number_of_images(1)
		.with_aspect_ratio("16:9"); // Match the video aspect ratio

	let imagen_response = client
		.exec_generate_images_imagen(imagen_model_name, imagen_req)
		.await
		.map_err(|e| format!("Imagen API call failed: {e:?}"))?;

	let generated_image = imagen_response
		.generated_images
		.into_iter()
		.next()
		.ok_or("No image generated by Imagen.")?;

	println!(
		"   Generated image (first frame) bytes length: {}",
		generated_image.image_bytes.len()
	);

	// 2. Use the generated image to create a video with Veo
	let video_prompt = "A fluffy white cloud slowly drifting across a blue sky.";
	let veo_req = VeoGenerateVideosRequest::new()
		.with_prompt(video_prompt)
		.with_image(generated_image.image_bytes, "image/png") // Assuming PNG as default for Imagen output
		.with_aspect_ratio("16:9") // Veo only supports 16:9 and 9:16
		.with_number_of_videos(1)
		.with_duration_seconds(5);

	let initial_res = client
		.exec_generate_videos_veo(veo_model_name, veo_req)
		.await
		.map_err(|e| format!("Initial Veo (image-to-video) API call failed: {e:?}"))?;

	println!(
		"   Initial Veo (image-to-video) operation name: {}",
		initial_res.operation_name
	);

	// Poll for operation completion
	let mut operation_status;
	let mut attempts = 0;
	loop {
		operation_status = client
			.exec_get_veo_operation_status(veo_model_name, initial_res.operation_name.clone())
			.await
			.map_err(|e| format!("Failed to get operation status during image-to-video polling: {e:?}"))?;

		if operation_status.done {
			break;
		}

		attempts += 1;
		if attempts >= MAX_ATTEMPTS {
			break;
		}

		println!("   Image-to-video operation not done yet. Waiting 10 seconds... (Attempt {attempts}/{MAX_ATTEMPTS})");
		tokio::time::sleep(Duration::from_secs(10)).await;
	}

	assert!(
		operation_status.done,
		"Veo image-to-video generation operation timed out or failed to complete."
	);

	if let Some(error) = operation_status.error {
		return Err(format!("Image-to-video generation failed with error: {error:?}"));
	}

	let result = operation_status
		.response
		.ok_or("No response found in completed image-to-video operation status.")?;

	assert!(
		!result.generate_video_response.generated_samples.is_empty(),
		"No videos were generated from image."
	);
	println!(
		"   Generated {} video(s) from image.",
		result.generate_video_response.generated_samples.len()
	);

	for (idx, generated_video) in result.generate_video_response.generated_samples.iter().enumerate() {
		save_video_from_uri("veo_image_to_video", idx, &generated_video.video.uri, &api_key).await?;
	}

	Ok(())
}
