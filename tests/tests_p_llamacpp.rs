//! Tests for the LlamaCpp adapter.
//!
//! These tests require the `llamacpp` feature to be enabled.
//! They also require actual model files to be present for integration tests.

#![cfg(feature = "llamacpp")]

use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatRequest};
use genai::{Client, Result};

// Note: These tests require actual model files to be present.
// They are designed to be skipped if models are not available.

#[tokio::test]
async fn test_adapter_kind_from_model() -> Result<()> {
	// Test model name detection for LlamaCpp
	assert_eq!(AdapterKind::from_model("local/model.gguf")?, AdapterKind::LlamaCpp);
	assert_eq!(AdapterKind::from_model("llama-3.2.gguf")?, AdapterKind::LlamaCpp);
	assert_eq!(AdapterKind::from_model("mistral-7b")?, AdapterKind::LlamaCpp);
	assert_eq!(AdapterKind::from_model("phi-3")?, AdapterKind::LlamaCpp);

	Ok(())
}

#[tokio::test]
async fn test_adapter_kind_serialization() {
	// Test serialization methods for LlamaCpp
	assert_eq!(AdapterKind::LlamaCpp.as_str(), "LlamaCpp");
	assert_eq!(AdapterKind::LlamaCpp.as_lower_str(), "llamacpp");
	assert_eq!(AdapterKind::LlamaCpp.default_key_env_name(), None);
}

#[tokio::test]
async fn test_model_manager_singleton() -> Result<()> {
	use genai::adapter::adapters::llamacpp::model_manager::ModelManager;

	// Test that ModelManager is a singleton
	let manager1 = ModelManager::instance().await?;
	let manager2 = ModelManager::instance().await?;

	// They should be the same instance (Arc comparison)
	assert!(std::ptr::eq(manager1.as_ref(), manager2.as_ref()));

	Ok(())
}

#[tokio::test]
async fn test_resolve_model_path() -> Result<()> {
	use genai::adapter::adapters::llamacpp::model_manager::resolve_model_path;

	// Test absolute path resolution
	let abs_path = "/absolute/path/to/model.gguf";
	let resolved = resolve_model_path(abs_path, None)?;
	assert_eq!(resolved.to_string_lossy(), abs_path);

	Ok(())
}

// Integration tests that require actual model files
// These will be skipped if no models are available

#[tokio::test]
#[ignore = "Requires actual GGUF model file"]
async fn test_llamacpp_chat_basic() -> Result<()> {
	// This test requires a real model file to be present
	// Set the model path to a real .gguf file to test
	let model_path = "local/test-model.gguf";

	let client = Client::default();

	// Create a simple chat request
	let messages = vec![ChatMessage::user("Hello, how are you?")];
	let chat_req = ChatRequest::new(messages);

	// Execute chat (this will use native LlamaCpp execution)
	let response = client.exec_chat(model_path, chat_req, None).await?;

	// Verify we got a response
	assert!(!response.contents.is_empty());
	println!("Response: {}", response.first_content_text_as_str().unwrap_or(""));

	Ok(())
}

#[tokio::test]
#[ignore = "Requires actual GGUF model file"]
async fn test_llamacpp_chat_streaming() -> Result<()> {
	// This test requires a real model file to be present
	let model_path = "local/test-model.gguf";

	let client = Client::default();

	// Create a simple chat request
	let messages = vec![ChatMessage::user("Tell me a short story.")];
	let chat_req = ChatRequest::new(messages);

	// Execute streaming chat
	let mut stream = client.exec_chat_stream(model_path, chat_req, None).await?;

	// Collect streaming responses
	let mut full_response = String::new();
	while let Some(stream_item) = stream.stream.next().await {
		match stream_item {
			Ok(genai::chat::ChatStreamEvent::Chunk(chunk)) => {
				full_response.push_str(&chunk.content);
				print!("{}", chunk.content);
			}
			Ok(genai::chat::ChatStreamEvent::End(_)) => {
				println!("\n--- Stream completed ---");
				break;
			}
			Err(e) => {
				eprintln!("Stream error: {}", e);
				break;
			}
			_ => {
				// Handle other event types (Start, ToolCall, ReasoningChunk)
			}
		}
	}

	// Verify we got some response
	assert!(!full_response.is_empty());

	Ok(())
}

#[tokio::test]
#[ignore = "Requires running llama.cpp server on localhost:8080"]
async fn test_llamacpp_server_tool_calling_direct() -> Result<()> {
	// Test tool calling with direct curl request to validate server
	use std::process::Command;

	let tool_payload = r#"{
        "messages": [{"role": "user", "content": "What's the weather like in London?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "stream": false
    }"#;

	let output = Command::new("curl")
		.args(&[
			"-X",
			"POST",
			"http://127.0.0.1:8080/v1/chat/completions",
			"-H",
			"Content-Type: application/json",
			"-d",
			tool_payload,
			"--max-time",
			"30",
		])
		.output()
		.expect("Failed to execute curl command");

	let response_text = String::from_utf8(output.stdout).expect("Invalid UTF-8 response");
	println!("Tool calling response: {}", response_text);

	// Verify we got a valid JSON response
	let response: serde_json::Value = serde_json::from_str(&response_text).expect("Server response is not valid JSON");

	// Check for basic OpenAI API structure
	assert!(response.get("choices").is_some(), "Response missing choices array");
	let choices = response["choices"].as_array().unwrap();
	assert!(!choices.is_empty(), "Choices array is empty");

	let first_choice = &choices[0];
	if let Some(message) = first_choice.get("message") {
		// Check if we got tool calls or text
		if let Some(tool_calls) = message.get("tool_calls") {
			println!("✓ Tool calls detected in response");
			let tool_calls_array = tool_calls.as_array().expect("tool_calls should be array");
			assert!(!tool_calls_array.is_empty(), "Tool calls array should not be empty");

			let first_tool_call = &tool_calls_array[0];
			if let Some(function) = first_tool_call.get("function") {
				if let Some(name) = function.get("name") {
					println!("Tool call function: {}", name.as_str().unwrap_or("unknown"));
				}
			}
		} else if let Some(content) = message.get("content") {
			println!("✓ Text response: {}", content.as_str().unwrap_or(""));
		}
	}

	Ok(())
}

#[tokio::test]
#[ignore = "Requires running llama.cpp server on localhost:8080"]
async fn test_llamacpp_server_multiple_tools() -> Result<()> {
	use genai::chat::Tool;
	use serde_json::json;

	let server_url = "http://127.0.0.1:8080";
	let client = Client::default();

	// Define multiple tools
	let weather_tool = Tool::new("get_weather")
		.with_description("Get current weather")
		.with_schema(json!({
			"type": "object",
			"properties": {
				"city": {"type": "string"}
			},
			"required": ["city"]
		}));

	let time_tool = Tool::new("get_time").with_description("Get current time").with_schema(json!({
		"type": "object",
		"properties": {
			"timezone": {"type": "string"}
		}
	}));

	let messages = vec![ChatMessage::user("What time is it in Tokyo and what's the weather like?")];

	let mut chat_req = ChatRequest::new(messages);
	chat_req.tools = Some(vec![weather_tool, time_tool]);

	let response = client.exec_chat(server_url, chat_req, None).await?;

	// Should get a response (may or may not contain tool calls depending on model behavior)
	assert!(!response.contents.is_empty());
	println!("Response: {}", response.first_content_text_as_str().unwrap_or(""));

	Ok(())
}

#[tokio::test]
#[ignore = "Requires running llama.cpp server on localhost:8080"]
async fn test_llamacpp_server_basic_chat() -> Result<()> {
	// Test basic chat functionality with the server using curl directly
	// This bypasses the genai client to test the server directly
	use std::process::Command;

	let output = Command::new("curl")
		.args(&[
			"-X",
			"POST",
			"http://127.0.0.1:8080/v1/chat/completions",
			"-H",
			"Content-Type: application/json",
			"-d",
			r#"{"messages":[{"role":"user","content":"Hello! Please respond with exactly 'Hello world'"}],"stream":false}"#,
			"--max-time",
			"30",
		])
		.output()
		.expect("Failed to execute curl command");

	let response_text = String::from_utf8(output.stdout).expect("Invalid UTF-8 response");
	println!("Direct server response: {}", response_text);

	// Verify we got a valid JSON response
	let response: serde_json::Value = serde_json::from_str(&response_text).expect("Server response is not valid JSON");

	// Check for basic OpenAI API structure
	assert!(response.get("choices").is_some(), "Response missing choices array");
	let choices = response["choices"].as_array().unwrap();
	assert!(!choices.is_empty(), "Choices array is empty");

	Ok(())
}

// Utility functions for testing
fn _create_test_messages() -> Vec<ChatMessage> {
	vec![
		ChatMessage::system("You are a helpful assistant."),
		ChatMessage::user("Hello, world!"),
	]
}
