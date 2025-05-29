use genai::{
	Client,
	chat::{ImagenGenerateImagesRequest, VeoGenerateVideosRequest},
};
use std::time::Duration;

const MODEL_VEO: &str = "veo-2.0-generate-001";
const MODEL_IMAGEN: &str = "imagen-3.0-generate-002";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let client = Client::default();

	println!("\n--- Text-to-Video Generation ---");
	let prompt_text = "Panning wide shot of a calico kitten sleeping in the sunshine";
	let veo_req_text = VeoGenerateVideosRequest::new()
		.with_prompt(prompt_text)
		.with_aspect_ratio("16:9")
		.with_person_generation("dont_allow");

	let initial_veo_res = client.exec_generate_videos_veo(MODEL_VEO, veo_req_text).await?;
	println!(
		"Initial Veo response (operation name): {}",
		initial_veo_res.operation_name
	);

	let mut operation_status = client
		.exec_get_veo_operation_status(MODEL_VEO, initial_veo_res.operation_name.clone())
		.await?;

	while !operation_status.done {
		println!("Operation not done yet. Waiting 10 seconds...");
		tokio::time::sleep(Duration::from_secs(10)).await;
		operation_status = client
			.exec_get_veo_operation_status(MODEL_VEO, initial_veo_res.operation_name.clone())
			.await?;
	}

	if let Some(result) = operation_status.response {
		for generated_video in result.generate_video_response.generated_samples {
			println!("Generated video URI: {}", generated_video.video.uri);
			// In a real application, you would download the video from this URI.
			// For this example, we'll just print the URI.
			// Example of how you might download (requires `reqwest` feature and `tokio::fs`):
			// let video_bytes = reqwest::get(&generated_video.uri).await?.bytes().await?;
			// let mut file = File::create(format!("video_from_text_{n}.mp4"))?;
			// file.write_all(&video_bytes)?;
			// println!("Saved video_from_text_{n}.mp4");
		}
	} else if let Some(error) = operation_status.error {
		eprintln!("Video generation failed: {error:?}");
	}

	println!("\n--- Image-to-Video Generation ---");
	let prompt_image = "Panning wide shot of a calico kitten sleeping in the sunshine";
	let imagen_req = ImagenGenerateImagesRequest::new(prompt_image)
		.with_aspect_ratio("16:9")
		.with_number_of_images(1);

	let imagen_response = client.exec_generate_images_imagen(MODEL_IMAGEN, imagen_req).await?;
	let generated_image = imagen_response
		.generated_images
		.into_iter()
		.next()
		.ok_or("No image generated")?;

	println!(
		"Generated image (first frame) bytes length: {}",
		generated_image.image_bytes.len()
	);

	let veo_req_image = VeoGenerateVideosRequest::new()
		.with_prompt(prompt_image)
		.with_image(generated_image.image_bytes, "image/png") // Assuming PNG for simplicity
		.with_aspect_ratio("16:9")
		.with_number_of_videos(1);

	let initial_veo_res_image = client.exec_generate_videos_veo(MODEL_VEO, veo_req_image).await?;
	println!(
		"Initial Veo (image-to-video) response (operation name): {}",
		initial_veo_res_image.operation_name
	);

	let mut operation_status_image = client
		.exec_get_veo_operation_status(MODEL_VEO, initial_veo_res_image.operation_name.clone())
		.await?;

	while !operation_status_image.done {
		println!("Image-to-video operation not done yet. Waiting 10 seconds...");
		tokio::time::sleep(Duration::from_secs(10)).await;
		operation_status_image = client
			.exec_get_veo_operation_status(MODEL_VEO, initial_veo_res_image.operation_name.clone())
			.await?;
	}

	if let Some(result) = operation_status_image.response {
		for generated_video in result.generate_video_response.generated_samples {
			println!("Generated video (from image) URI: {}", generated_video.video.uri);
			// Similar download logic as above
		}
	} else if let Some(error) = operation_status_image.error {
		eprintln!("Image-to-video generation failed: {error:?}");
	}

	Ok(())
}
