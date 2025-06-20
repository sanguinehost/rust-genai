mod support;
mod tests_p_gemini_document; // Added for document understanding tests
use crate::support::{Check, common_tests};
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, Tool};
use genai::resolver::AuthData;
use serde_json::json;
type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>; // For tests.

// "gemini-2.0-flash", "gemini-2.0-flash-lite" (somehow function calling work with -lite)
// "gemini-2.5-flash-preview-05-20" "gemini-2.5-pro-preview-05-06"
const MODEL: &str = "gemini-2.5-flash-preview-05-20";

// region:    --- Chat

#[tokio::test]
async fn test_chat_simple_ok() -> Result<()> {
	common_tests::common_test_chat_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_multi_system_ok() -> Result<()> {
	common_tests::common_test_chat_multi_system_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_json_structured_ok() -> Result<()> {
	common_tests::common_test_chat_json_structured_ok(MODEL, Some(Check::USAGE)).await
}

#[tokio::test]
async fn test_chat_temperature_ok() -> Result<()> {
	common_tests::common_test_chat_temperature_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stop_sequences_ok() -> Result<()> {
	common_tests::common_test_chat_stop_sequences_ok(MODEL).await
}

// endregion: --- Chat

// region:    --- Chat Implicit Cache

// NOTE: This should eventually work with the 2.5 Flash, but right now, we do not get the cached_tokens
//       So, disabling
// #[tokio::test]
// async fn test_chat_cache_implicit_simple_ok() -> Result<()> {
// 	common_tests::common_test_chat_cache_implicit_simple_ok(MODEL).await
// }

// endregion: --- Chat Implicit Cache

// region:    --- Chat Stream Tests

#[tokio::test]
async fn test_chat_stream_simple_ok() -> Result<()> {
	common_tests::common_test_chat_stream_simple_ok(MODEL, None).await
}

#[tokio::test]
async fn test_chat_stream_capture_content_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_content_ok(MODEL).await
}

#[tokio::test]
async fn test_chat_stream_capture_all_ok() -> Result<()> {
	common_tests::common_test_chat_stream_capture_all_ok(MODEL, None).await
}

// endregion: --- Chat Stream Tests

// region:    --- Image Tests

// NOTE: Gemini does not seem to support URL
// #[tokio::test]
// async fn test_chat_image_url_ok() -> Result<()> {
// 	common_tests::common_test_chat_image_url_ok(MODEL).await
// }

#[tokio::test]
async fn test_chat_image_b64_ok() -> Result<()> {
	common_tests::common_test_chat_image_b64_ok(MODEL).await
}
// endregion: --- Image Test

// region:    --- Tool Tests

#[tokio::test]
async fn test_tool_simple_ok() -> Result<()> {
	common_tests::common_test_tool_simple_ok(MODEL, true).await
}

#[tokio::test]
async fn test_tool_full_flow_ok() -> Result<()> {
	common_tests::common_test_tool_full_flow_ok(MODEL, true).await
}
// endregion: --- Tool Tests

// region:    --- Resolver Tests

#[tokio::test]
async fn test_resolver_auth_ok() -> Result<()> {
	common_tests::common_test_resolver_auth_ok(MODEL, AuthData::from_env("GEMINI_API_KEY")).await
}

// endregion: --- Resolver Tests

// region:    --- List

#[tokio::test]
async fn test_list_models() -> Result<()> {
	common_tests::common_test_list_models(AdapterKind::Gemini, "gemini-1.5-pro").await
}

// endregion: --- List

// region:    --- New Advanced Options Tests

#[tokio::test]
async fn test_chat_top_k_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let chat_options = ChatOptions::default().with_top_k(40);
	let messages = vec![ChatMessage::user("What is the capital of France?")];
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	println!("test_chat_top_k_ok - Response Contents: {:?}", res.contents);
	Ok(())
}

#[tokio::test]
async fn test_chat_seed_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let chat_options = ChatOptions::default().with_seed(12345);
	let messages = vec![ChatMessage::user("Tell me a short joke.")];
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	println!("test_chat_seed_ok - Response Contents: {:?}", res.contents);
	Ok(())
}

#[tokio::test]
async fn test_chat_penalties_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let chat_options = ChatOptions::default().with_presence_penalty(0.5).with_frequency_penalty(0.5);
	let messages = vec![ChatMessage::user("Write a paragraph about recurring themes in storytelling.")];
	let chat_req = ChatRequest::new(messages);

	// Using gemini-1.5-pro for this test as it might have better support for penalties
	let res = client.exec_chat("gemini-1.5-pro", chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	println!("test_chat_penalties_ok - Response Contents: {:?}", res.contents);
	Ok(())
}

#[tokio::test]
async fn test_chat_candidate_count_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let expected_candidates: usize = 2;
	let chat_options = ChatOptions::default().with_candidate_count(expected_candidates.try_into()?);
	let messages = vec![ChatMessage::user(
		"Why is the sky blue? Give two slightly different explanations.",
	)]; // Prompt to encourage different responses
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	assert_eq!(
		res.contents.len(),
		expected_candidates,
		"Expected {} candidates, got {}",
		expected_candidates,
		res.contents.len()
	);

	println!(
		"test_chat_candidate_count_ok - Received {} candidates:",
		res.contents.len()
	);
	for (i, content) in res.contents.iter().enumerate() {
		println!("  Candidate {i}: {content:?}");
		match content {
			genai::chat::MessageContent::Text(text_content) => {
				assert!(!text_content.is_empty(), "Candidate {i} text should not be empty");
			}
			_ => panic!("Candidate {i} was not text as expected for this test"),
		}
	}
	Ok(())
}

#[tokio::test]
async fn test_chat_cached_content_id_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let chat_options = ChatOptions::default()
		.with_cached_content_id("cachedContents/some-test-id-that-does-not-exist-12345".to_string());
	let messages = vec![ChatMessage::user("What is 2 + 2?")];
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await;

	if let Err(e) = &res {
		println!("test_chat_cached_content_id_ok - Received error (potentially expected for invalid ID): {e}");
	} else if let Ok(response_ok) = res {
		assert!(!response_ok.contents.is_empty());
		println!(
			"test_chat_cached_content_id_ok - Response Contents: {:?}",
			response_ok.contents
		);
	}
	Ok(())
}

fn helper_dummy_tool_weather() -> Tool {
	Tool::new("get_current_weather".to_string())
		.with_description("Get the current weather in a given location".to_string())
		.with_schema(json!({
			"type": "object",
			"properties": {
				"location": {
					"type": "string",
					"description": "The city and state, e.g. San Francisco, CA"
				},
				"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
			},
			"required": ["location"]
		}))
}

fn helper_dummy_tool_recipe() -> Tool {
	Tool::new("find_recipe".to_string())
		.with_description("Find a recipe for a given dish".to_string())
		.with_schema(json!({
			"type": "object",
			"properties": {
				"dish": {
					"type": "string",
					"description": "The dish to search for, e.g. pasta carbonara"
				}
			},
			"required": ["dish"]
		}))
}

#[tokio::test]
async fn test_chat_function_calling_mode_none_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let tools = vec![helper_dummy_tool_weather()];
	let chat_options = ChatOptions::default().with_function_calling_mode("NONE".to_string());

	let messages = vec![ChatMessage::user("What's the weather like in London?")];
	let chat_req = ChatRequest::new(messages).with_tools(tools);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected text content in response");
	assert!(
		!matches!(
			res.contents.first().as_ref().unwrap(),
			genai::chat::MessageContent::ToolCalls(_)
		),
		"Expected no tool calls when mode is NONE"
	);
	println!(
		"test_chat_function_calling_mode_none_ok - Response Contents: {:?}",
		res.contents
	);
	Ok(())
}

#[tokio::test]
async fn test_chat_function_calling_mode_auto_with_tool_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let tools = vec![helper_dummy_tool_weather()];
	let chat_options = ChatOptions::default().with_function_calling_mode("AUTO".to_string());

	let messages = vec![ChatMessage::user("What's the weather like in London?")];
	let chat_req = ChatRequest::new(messages).with_tools(tools);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	assert!(
		matches!(
			res.contents.first().as_ref().unwrap(),
			genai::chat::MessageContent::ToolCalls(_)
		),
		"Expected tool call when mode is AUTO and prompt triggers tool"
	);
	if let Some(genai::chat::MessageContent::ToolCalls(tool_calls)) = res.contents.first() {
		println!("test_chat_function_calling_mode_auto_with_tool_ok - Tool calls: {tool_calls:?}");
	} else {
		println!("test_chat_function_calling_mode_auto_with_tool_ok - No tool calls, contents: {res:?}");
	}
	Ok(())
}

#[tokio::test]
async fn test_chat_allowed_function_names_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let tools = vec![helper_dummy_tool_weather(), helper_dummy_tool_recipe()];
	let chat_options = ChatOptions::default()
		.with_allowed_function_names(vec!["get_current_weather".to_string()])
		.with_function_calling_mode("ANY".to_string());

	let messages = vec![ChatMessage::user("What is the weather in Berlin?")];
	let chat_req = ChatRequest::new(messages).with_tools(tools.clone());

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	if let Some(genai::chat::MessageContent::ToolCalls(tool_calls)) = res.contents.first() {
		assert_eq!(tool_calls.len(), 1, "Expected exactly one tool call");
		assert_eq!(
			tool_calls[0].fn_name, "get_current_weather",
			"Expected 'get_current_weather' to be called"
		);
		println!("test_chat_allowed_function_names_ok (weather) - Tool calls: {tool_calls:?}");
	} else {
		panic!(
			"Expected a tool call for 'get_current_weather', got: {:?}",
			res.contents
		);
	}

	// Test the disallowed tool
	let chat_options_recipe_disallowed = ChatOptions::default()
		.with_allowed_function_names(vec!["get_current_weather".to_string()]) // Still only weather allowed
		.with_function_calling_mode("ANY".to_string()); // Must be ANY if allowed_function_names is set

	let messages_recipe = vec![ChatMessage::user("Find me a recipe for apple pie.")];
	// Ensure `tools` is available; it was cloned for the first request, so original `tools` is still here.
	let chat_req_recipe = ChatRequest::new(messages_recipe).with_tools(tools.clone());

	let res_recipe = client
		.exec_chat(MODEL, chat_req_recipe, Some(&chat_options_recipe_disallowed))
		.await?;
	assert!(
		!res_recipe.contents.is_empty(),
		"Expected content in response for recipe prompt"
	);

	if let Some(genai::chat::MessageContent::ToolCalls(tool_calls)) = res_recipe.contents.first() {
		println!("test_chat_allowed_function_names_ok (recipe) - Tool call(s) made: {tool_calls:?}");
		assert!(
			!tool_calls.is_empty(),
			"If ToolCalls variant, it should not be empty here based on previous runs."
		);
		for tool_call in tool_calls {
			assert_eq!(
				tool_call.fn_name, "get_current_weather",
				"If any tool is called with allowed_functions_names restricting to weather, it must be get_current_weather. Called: {}",
				tool_call.fn_name
			);
		}
	} else {
		// This case is also acceptable: the model chose not to call any tool.
		println!("test_chat_allowed_function_names_ok (recipe) - No tool call made (text response): {res_recipe:?}");
		assert!(
			matches!(
				res_recipe.contents.first().as_ref().unwrap(),
				genai::chat::MessageContent::Text(_)
			),
			"Expected a text response if no tool call was made."
		);
	}

	Ok(())
}

#[tokio::test]
async fn test_chat_enum_structured_ok() -> Result<()> {
	use genai::chat::{ChatResponseFormat, EnumSpec};

	let client = support::common_client_gemini();
	let messages = vec![ChatMessage::user("What type of instrument is an oboe?")];

	let enum_schema = json!({
		"type": "STRING",
		"enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
	});

	let chat_options =
		ChatOptions::default().with_response_format(ChatResponseFormat::EnumSpec(EnumSpec::new(enum_schema)));
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");

	if let Some(genai::chat::MessageContent::Text(text_content)) = res.contents.first() {
		println!("test_chat_enum_structured_ok - Response: {text_content}");
		assert!(
			["Percussion", "String", "Woodwind", "Brass", "Keyboard"]
				.iter()
				.any(|&s| text_content.trim() == s),
			"Response text '{text_content}' is not one of the expected enum values"
		);
	} else {
		panic!("Expected a text response, got: {:?}", res.contents);
	}

	Ok(())
}

#[tokio::test]
async fn test_chat_json_schema_structured_ok() -> Result<()> {
	use genai::chat::{ChatResponseFormat, JsonSchemaSpec};

	let client = support::common_client_gemini();
	let messages = vec![ChatMessage::user("Generate a user profile.")];

	let json_schema = json!({
		"title": "UserProfile",
		"type": "object",
		"properties": {
			"username": {
				"type": "string",
				"description": "User's unique name"
			},
			"age": {
				"type": "integer",
				"minimum": 0,
				"maximum": 120
			},
			"roles": {
				"type": "array",
				"minItems": 1,
				"items": {
					"type": "string",
					"enum": ["admin", "viewer"]
				}
			},
			"contact": {
				"anyOf": [
					{
						"type": "object",
						"properties": {
							"street": { "type": "string" },
							"city": { "type": "string" }
						},
						"required": ["street", "city"]
					},
					{ "type": "string" }
				]
			}
		},
		"required": ["username", "roles", "contact"]
	});

	let chat_options = ChatOptions::default()
		.with_response_format(ChatResponseFormat::JsonSchemaSpec(JsonSchemaSpec::new(json_schema)));
	let chat_req = ChatRequest::new(messages);

	// Use a model that supports JSON Schema (e.g., gemini-2.5-flash-preview-05-20 or newer)
	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");

	if let Some(genai::chat::MessageContent::Text(text_content)) = res.contents.first() {
		println!("test_chat_json_schema_structured_ok - Response: {text_content}");
		// Attempt to parse the response as JSON to validate its structure
		let parsed_json: serde_json::Value = serde_json::from_str(text_content)?;

		// Basic validation of the parsed JSON against the schema's expected top-level properties
		assert!(parsed_json.is_object(), "Expected JSON object response");
		assert!(parsed_json.get("username").is_some(), "Expected 'username' field");
		assert!(parsed_json.get("roles").is_some(), "Expected 'roles' field");
		assert!(parsed_json.get("contact").is_some(), "Expected 'contact' field");

		if let Some(roles_array) = parsed_json.get("roles").and_then(|v| v.as_array()) {
			assert!(!roles_array.is_empty(), "Expected at least one role");
			for role in roles_array {
				assert!(role.is_string(), "Role should be a string");
				let role_str = role.as_str().unwrap();
				assert!(
					role_str == "admin" || role_str == "viewer",
					"Role '{role_str}' is not a valid enum value"
				);
			}
		} else {
			panic!("Expected 'roles' to be an array");
		}

		if let Some(contact_value) = parsed_json.get("contact") {
			if contact_value.is_object() {
				assert!(
					contact_value.get("street").is_some(),
					"Expected 'street' in contact object"
				);
				assert!(contact_value.get("city").is_some(), "Expected 'city' in contact object");
			} else {
				assert!(contact_value.is_string(), "Expected 'contact' to be a string or object");
			}
		}
	} else {
		panic!("Expected a text response, got: {:?}", res.contents);
	}

	Ok(())
}

#[tokio::test]
async fn test_chat_safety_settings_ok() -> Result<()> {
	use genai::chat::{HarmCategory, HarmBlockThreshold, SafetySetting};
	
	let client = support::common_client_gemini();
	let safety_settings = vec![
		SafetySetting::new(HarmCategory::Harassment, HarmBlockThreshold::BlockMediumAndAbove),
		SafetySetting::new(HarmCategory::HateSpeech, HarmBlockThreshold::BlockOnlyHigh),
	];
	let chat_options = ChatOptions::default().with_safety_settings(safety_settings);
	let messages = vec![ChatMessage::user("Write a polite hello message.")];
	let chat_req = ChatRequest::new(messages);

	let res = client.exec_chat(MODEL, chat_req, Some(&chat_options)).await?;
	assert!(!res.contents.is_empty(), "Expected content in response");
	println!("test_chat_safety_settings_ok - Response Contents: {:?}", res.contents);

	Ok(())
}

// endregion: --- New Advanced Options Tests

// region:    --- Streaming Tool Call Tests

#[tokio::test]
async fn test_chat_stream_tool_call_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let weather_tool = Tool::new("get_weather".to_string())
		.with_description("Get the current weather for a location".to_string())
		.with_schema(json!({
			"type": "object",
			"properties": {
				"city": {
					"type": "string",
					"description": "The city name"
				},
				"country": {
					"type": "string",
					"description": "The country of the city"
				},
				"unit": {
					"type": "string",
					"enum": ["C", "F"],
					"description": "Temperature unit (C for Celsius, F for Fahrenheit)"
				}
			},
			"required": ["city", "country", "unit"]
		}));

	let messages = vec![ChatMessage::user("Get the weather for Tokyo, Japan in Celsius using the get_weather function.")];
	let chat_req = ChatRequest::new(messages).with_tools(vec![weather_tool]);
	
	// Force function calling with ANY mode
	let chat_options = ChatOptions::default().with_function_calling_mode("ANY".to_string());

	let response = client.exec_chat_stream(MODEL, chat_req, Some(&chat_options)).await?;
	let mut stream = response.stream;

	let mut tool_call_received = false;
	let mut start_received = false;
	let mut end_received = false;

	use futures::StreamExt;
	while let Some(stream_result) = stream.next().await {
		match stream_result? {
			genai::chat::ChatStreamEvent::Start => {
				start_received = true;
				println!("✓ Stream started");
			}
			genai::chat::ChatStreamEvent::ToolCall(tool_call) => {
				tool_call_received = true;
				println!("✓ Tool call received: {} with args: {:?}", tool_call.fn_name, tool_call.fn_arguments);
				assert_eq!(tool_call.fn_name, "get_weather");
				assert!(!tool_call.call_id.is_empty());
				assert!(tool_call.fn_arguments.is_object());
			}
			genai::chat::ChatStreamEvent::Chunk(chunk) => {
				println!("Content chunk: {}", chunk.content);
			}
			genai::chat::ChatStreamEvent::ReasoningChunk(reasoning) => {
				println!("Reasoning chunk: {}", reasoning.content);
			}
			genai::chat::ChatStreamEvent::End(_end_event) => {
				end_received = true;
				println!("✓ Stream ended");
				break;
			}
		}
	}

	assert!(start_received, "Expected stream start event");
	assert!(tool_call_received, "Expected tool call event in stream");
	assert!(end_received, "Expected stream end event");

	Ok(())
}

#[tokio::test]
async fn test_chat_stream_tool_full_flow_ok() -> Result<()> {
	use genai::chat::ToolResponse;

	let client = support::common_client_gemini();
	let calculator_tool = Tool::new("calculate".to_string())
		.with_description("Perform basic arithmetic calculations".to_string())
		.with_schema(json!({
			"type": "object",
			"properties": {
				"operation": {
					"type": "string",
					"enum": ["add", "subtract", "multiply", "divide"],
					"description": "The arithmetic operation to perform"
				},
				"a": {
					"type": "number",
					"description": "First number"
				},
				"b": {
					"type": "number",
					"description": "Second number"
				}
			},
			"required": ["operation", "a", "b"]
		}));

	// Step 1: Initial request with tool
	let mut messages = vec![ChatMessage::user("What is 23 + 19?")];
	let mut chat_req = ChatRequest::new(messages.clone()).with_tools(vec![calculator_tool]);

	// Step 2: Get tool call from streaming response  
	let chat_options = ChatOptions::default().with_function_calling_mode("ANY".to_string());
	let response = client.exec_chat_stream(MODEL, chat_req.clone(), Some(&chat_options)).await?;
	let mut stream = response.stream;

	let mut tool_call_opt = None;
	use futures::StreamExt;
	while let Some(stream_result) = stream.next().await {
		match stream_result? {
			genai::chat::ChatStreamEvent::ToolCall(tool_call) => {
				println!("✓ Tool call in stream: {} with args: {:?}", tool_call.fn_name, tool_call.fn_arguments);
				tool_call_opt = Some(tool_call);
			}
			genai::chat::ChatStreamEvent::End(_) => break,
			_ => {} // Ignore other events for this test
		}
	}

	let tool_call = tool_call_opt.expect("Expected tool call in stream");
	assert_eq!(tool_call.fn_name, "calculate");

	// Step 3: Simulate tool execution
	let a = tool_call.fn_arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
	let b = tool_call.fn_arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
	let operation = tool_call.fn_arguments.get("operation").and_then(|v| v.as_str()).unwrap_or("");

	let result = match operation {
		"add" => a + b,
		"subtract" => a - b,
		"multiply" => a * b,
		"divide" => if b != 0.0 { a / b } else { f64::NAN },
		_ => f64::NAN,
	};

	let tool_response = ToolResponse::new(
		tool_call.call_id.clone(),
		json!({
			"result": result,
			"operation": format!("{} {} {} = {}", a, operation, b, result)
		}).to_string(),
	);

	// Step 4: Continue conversation with tool response
	messages.push(ChatMessage::from(vec![tool_call]));
	messages.push(ChatMessage::from(tool_response));
	chat_req = ChatRequest::new(messages);

	// Step 5: Get final response as stream
	let response = client.exec_chat_stream(MODEL, chat_req, None).await?;
	let mut stream = response.stream;

	let mut final_content = String::new();
	while let Some(stream_result) = stream.next().await {
		match stream_result? {
			genai::chat::ChatStreamEvent::Chunk(chunk) => {
				final_content.push_str(&chunk.content);
			}
			genai::chat::ChatStreamEvent::End(_) => break,
			_ => {} // Ignore other events
		}
	}

	assert!(!final_content.is_empty(), "Expected final response content");
	assert!(final_content.contains("42") || final_content.contains("23") || final_content.contains("19"), 
		"Expected response to contain calculation result or input numbers");
	println!("✓ Full tool flow completed. Final response: {}", final_content);

	Ok(())
}

#[tokio::test]
async fn test_chat_stream_parallel_tool_calls_ok() -> Result<()> {
	let client = support::common_client_gemini();
	let weather_tool = helper_dummy_tool_weather();
	let recipe_tool = helper_dummy_tool_recipe();

	let messages = vec![ChatMessage::user("What's the weather in London and find me a recipe for pasta?")];
	let chat_req = ChatRequest::new(messages).with_tools(vec![weather_tool, recipe_tool]);
	
	let chat_options = ChatOptions::default().with_function_calling_mode("ANY".to_string());
	let response = client.exec_chat_stream(MODEL, chat_req, Some(&chat_options)).await?;
	let mut stream = response.stream;

	let mut tool_calls = Vec::new();
	use futures::StreamExt;
	while let Some(stream_result) = stream.next().await {
		match stream_result? {
			genai::chat::ChatStreamEvent::ToolCall(tool_call) => {
				println!("✓ Tool call received: {} with args: {:?}", tool_call.fn_name, tool_call.fn_arguments);
				tool_calls.push(tool_call);
			}
			genai::chat::ChatStreamEvent::End(_) => break,
			_ => {} // Ignore other events for this test
		}
	}

	// We expect at least one tool call, potentially parallel calls
	assert!(!tool_calls.is_empty(), "Expected at least one tool call");
	
	// Verify we got expected tool names
	let tool_names: Vec<&str> = tool_calls.iter().map(|tc| tc.fn_name.as_str()).collect();
	println!("Tool calls received: {:?}", tool_names);
	
	// Should have weather or recipe calls (or both for parallel calling)
	assert!(tool_names.iter().any(|&name| name == "get_current_weather" || name == "find_recipe"),
		"Expected weather or recipe tool calls");

	Ok(())
}

// endregion: --- Streaming Tool Call Tests
