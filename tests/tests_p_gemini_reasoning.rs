mod support;

use crate::support::{extract_stream_end, Result};
use genai::chat::{ChatMessage, ChatOptions, ChatRequest};
use genai::Client;

// "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25"
const MODEL: &str = "gemini-2.5-flash-preview-04-17";

// NOTE: For now just single test to make sure reasonning token get captured.

#[tokio::test]
async fn test_gemini_code_execution_reasoning() -> Result<()> {
	// -- Setup
	dotenvy::dotenv().ok(); // Load .env if available, for local testing
	let client = Client::default();

	let user_message = ChatMessage::user(
		"What is the sum of the first 10 prime numbers? Please show the Python code you used to calculate this, and then state the sum."
	);
	let chat_req = ChatRequest::new(vec![user_message]);

	let chat_options = ChatOptions::default()
		.with_capture_content(true)
		.with_capture_usage(true)
		.with_capture_reasoning_content(true)
		.with_gemini_thinking_budget(1024)
		.with_gemini_enable_code_execution(true);

	// -- Exec
	let chat_stream_response = client.exec_chat_stream(MODEL, chat_req, Some(&chat_options)).await?;
	let stream_extract = extract_stream_end(chat_stream_response.stream).await?;

	// -- Check Reasoning Content (from chunks)
	let reasoning_content = stream_extract.reasoning_content.ok_or("Should have reasoning_content from chunks")?;
	assert!(reasoning_content.contains("def ") || reasoning_content.contains("python"), "Reasoning content should contain Python code indicators.");
	println!("\n--- Reasoning Content (from Chunks): ---\n{}", reasoning_content);

	// -- Check Captured Reasoning Content (from StreamEnd)
	let captured_reasoning = stream_extract.stream_end.captured_reasoning_content.as_ref().ok_or("StreamEnd should have captured_reasoning_content")?;
	assert_eq!(captured_reasoning, &reasoning_content, "Captured reasoning content should match reasoning from chunks.");
	println!("\n--- Captured Reasoning Content (from StreamEnd): ---\n{}", captured_reasoning);

	// -- Check Usage for Reasoning Tokens
	let usage = stream_extract.stream_end.captured_usage.as_ref().ok_or("StreamEnd should have captured_usage")?;
	let completion_details = usage.completion_tokens_details.as_ref().ok_or("Usage should have completion_tokens_details")?;
	let reasoning_tokens = completion_details.reasoning_tokens.ok_or("Completion details should have reasoning_tokens")?;
	assert!(reasoning_tokens > 0, "Reasoning tokens should be greater than 0. Got: {}", reasoning_tokens);
	println!("\n--- Reasoning Tokens: {} ---", reasoning_tokens);

	// -- Check Main Content (final answer)
	let main_content = stream_extract.content.ok_or("Should have main content from chunks")?;
	assert!(main_content.contains("129"), "Main content should contain the sum '129'. Got: {}", main_content);
	println!("\n--- Main Content (from Chunks): ---\n{}", main_content);

	println!("\n--- Full StreamEnd Details: {:?} ---", stream_extract.stream_end);

	Ok(())
}
