//! Example showing how to get the list of models per `AdapterKind`
//! Note: Currently, only Ollama makes a dynamic query. Other adapters have a static list of models.

use genai::Client;
use genai::adapter::AdapterKind;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

	let kinds: &[AdapterKind] = &[
		AdapterKind::OpenAI,
		AdapterKind::Ollama,
		AdapterKind::Gemini,
		AdapterKind::Anthropic,
		AdapterKind::Groq,
		AdapterKind::Cohere,
	];

	let client = Client::default();

	for &kind in kinds {
		println!("\n--- Models for {kind}");
		let models = client.all_model_names(kind).await?;
		println!("{models:?}");
	}

	Ok(())
}
