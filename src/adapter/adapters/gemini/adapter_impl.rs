use crate::adapter::adapters::support::get_api_key;
use crate::adapter::gemini::GeminiStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet, ChatRequest, ChatResponse, ChatResponseFormat, ChatRole, ChatStream, ChatStreamResponse,
	CompletionTokensDetails, ContentPart, ImageSource, MessageContent, ToolCall, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Error, ModelIden, Result, ServiceTarget};
use reqwest::RequestBuilder;
use reqwest_eventsource::EventSource;
use serde_json::{Value, json};
use value_ext::JsonValueExt;

pub struct GeminiAdapter;

const MODELS: &[&str] = &["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"];

// curl \
//   -H 'Content-Type: application/json' \
//   -d '{"contents":[{"parts":[{"text":"Explain how AI works"}]}]}' \
//   -X POST 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=YOUR_API_KEY'

impl GeminiAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "GEMINI_API_KEY";
}

impl Adapter for GeminiAdapter {
	fn default_endpoint() -> Endpoint {
		const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	/// Note: For now, this returns the common models (see above)
	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	/// NOTE: As Google Gemini has decided to put their API_KEY in the URL,
	///       this will return the URL without the API_KEY in it. The API_KEY will need to be added by the caller.
	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		let base_url = endpoint.base_url();
		let model_name = model.model_name.clone();
		match service_type {
			ServiceType::Chat => format!("{base_url}models/{model_name}:generateContent"),
			ServiceType::ChatStream => format!("{base_url}models/{model_name}:streamGenerateContent?alt=sse"),
		}
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- api_key
		let api_key = get_api_key(auth, &model)?;

		// -- url
		let mut url = Self::get_service_url(&model, service_type, endpoint);
		// Append key with '&' if '?' is already present (due to alt=sse), otherwise use '?'
		if url.contains('?') {
			url = format!("{url}&key={api_key}");
		} else {
			url = format!("{url}?key={api_key}");
		}

		// -- parts
		let GeminiChatRequestParts {
			system,
			contents,
			final_tools_payload,
		} = Self::into_gemini_request_parts(model, chat_req, &options_set)?;

		// -- Playload
		let mut payload = json!({
			"contents": contents,
		});

		// -- headers (empty for gemini, since API_KEY is in url)
		let headers = vec![];

		// Note: It's unclear from the spec if the content of systemInstruction should have a role.
		//       Right now, it is omitted (since the spec states it can only be "user" or "model")
		//       It seems to work. https://ai.google.dev/api/rest/v1beta/models/generateContent
		if let Some(system) = system {
			payload.x_insert(
				"systemInstruction",
				json!({
					"parts": [ { "text": system }]
				}),
			)?;
		}

		// -- Tools
		if let Some(tools_payload) = final_tools_payload {
			payload.x_insert("tools", tools_payload)?;
		}

		// -- Response Format
		if let Some(ChatResponseFormat::JsonSpec(st_json)) = options_set.response_format() {
			// x_insert
			//     responseMimeType: "application/json",
			// responseSchema: {
			payload.x_insert("/generationConfig/responseMimeType", "application/json")?;
			let mut schema = st_json.schema.clone();
			schema.x_walk(|parent_map, name| {
				if name == "additionalProperties" {
					parent_map.remove("additionalProperties");
				}
				true
			});
			payload.x_insert("/generationConfig/responseSchema", schema)?;
		}

		// -- Add supported ChatOptions
		if let Some(temperature) = options_set.temperature() {
			payload.x_insert("/generationConfig/temperature", temperature)?;
		}

		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("/generationConfig/stopSequences", options_set.stop_sequences())?;
		}

		if let Some(max_tokens) = options_set.max_tokens() {
			payload.x_insert("/generationConfig/maxOutputTokens", max_tokens)?;
		}
		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("/generationConfig/topP", top_p)?;
		}

		// -- Add Gemini-specific thinkingBudget
		if let Some(thinking_budget) = options_set.gemini_thinking_budget() {
			payload.x_insert("/generationConfig/thinkingConfig/thinkingBudget", thinking_budget)?;
		}

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		_options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;
		// -- Capture the provider_model_iden
		// TODO: Need to be implemented (if available), for now, just clone model_iden
		let provider_model_name: Option<String> = body.x_remove("modelVersion").ok();
		let provider_model_iden = model_iden.with_name_or_clone(provider_model_name);

		let gemini_response = Self::body_to_gemini_chat_response(&model_iden.clone(), body)?;
		let GeminiChatResponse {
			content,
			reasoning_content,
			usage,
		} = gemini_response;

		let final_content = match content {
			Some(GeminiChatContent::Text(content)) => Some(MessageContent::from_text(content)),
			Some(GeminiChatContent::ToolCall(tool_call)) => Some(MessageContent::from_tool_calls(vec![tool_call])),
			None => None,
		};

		let final_reasoning_content = match reasoning_content {
			Some(GeminiChatContent::Text(text)) => Some(MessageContent::from_text(text)),
			// Assuming reasoning content from Gemini is primarily textual (e.g., code, code output)
			// If reasoning could also be a ToolCall itself, this would need adjustment.
			Some(GeminiChatContent::ToolCall(_)) => None,
			None => None,
		};

		let reasoning_content_str: Option<String> = match final_reasoning_content {
			Some(MessageContent::Text(text)) => Some(text),
			_ => None, // Other MessageContent types (like ToolCalls, Parts) won't be simple strings
		};

		Ok(ChatResponse {
			content: final_content,
			reasoning_content: reasoning_content_str,
			model_iden,
			provider_model_iden,
			usage,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let event_source = EventSource::new(reqwest_builder)?;

		let gemini_stream = GeminiStreamer::new(event_source, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(gemini_stream);

		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}
}

// region:    --- Support

/// Support functions for GeminiAdapter
impl GeminiAdapter {
	pub(super) fn body_to_gemini_chat_response(model_iden: &ModelIden, mut body: Value) -> Result<GeminiChatResponse> {
		// If the body has an `error` property, then it is assumed to be an error.
		if body.get("error").is_some() {
			return Err(Error::StreamEventError {
				model_iden: model_iden.clone(),
				body,
			});
		}

		let mut main_text_parts: Vec<String> = Vec::new();
		let mut reasoning_text_parts: Vec<String> = Vec::new();
		let mut tool_calls_vec: Vec<ToolCall> = Vec::new();

		// Safely access candidates and parts
		if let Some(candidates_val) = body.get_mut("candidates") {
			if let Some(candidates_array) = candidates_val.as_array_mut() {
				if !candidates_array.is_empty() {
					// Assuming the first candidate is the one we care about
					if let Some(content_val) = candidates_array[0].get_mut("content") {
						if let Some(parts_val) = content_val.get_mut("parts") {
							if let Some(parts_array) = parts_val.as_array_mut() {
								for part_value in parts_array.iter_mut() {
									if let Some(fc_val) = part_value.get("functionCall").cloned() {
										tool_calls_vec.push(ToolCall {
											call_id: fc_val.x_get::<String>("name").unwrap_or_else(|_| String::new()),
											fn_name: fc_val.x_get::<String>("name").unwrap_or_else(|_| String::new()),
											fn_arguments: fc_val.x_get::<Value>("args").unwrap_or_else(|_| Value::Null),
										});
									} else if let Some(ec_val) = part_value.get("executableCode").cloned() {
										if let Some(code_str) = ec_val.x_get::<String>("code").ok() {
											reasoning_text_parts.push(code_str);
										}
									} else if let Some(tco_val) = part_value.get("toolCodeOutput").cloned() {
										if let Some(output_str) = tco_val.x_get::<String>("output").ok() {
											reasoning_text_parts.push(output_str);
										}
									} else if let Some(text_str) =
										part_value.get("text").and_then(Value::as_str).map(String::from)
									{
										main_text_parts.push(text_str);
									}
								}
							}
						}
					}
				}
			}
		}

		let final_content = if !tool_calls_vec.is_empty() {
			// For non-streaming, if there are tool calls, that's the primary content.
			// We'll take the first tool call for GeminiChatContent::ToolCall.
			// `to_chat_response` will wrap it in a Vec.
			// Note: If multiple tool calls are possible in a single non-streaming response part from Gemini,
			// this only takes the first. ChatResponse handles Vec<ToolCall>.
			Some(GeminiChatContent::ToolCall(tool_calls_vec.remove(0)))
		} else if !main_text_parts.is_empty() {
			Some(GeminiChatContent::Text(main_text_parts.join("")))
		} else {
			None
		};

		let final_reasoning_content = if !reasoning_text_parts.is_empty() {
			Some(GeminiChatContent::Text(reasoning_text_parts.join("\n"))) // Join reasoning parts if multiple
		} else {
			None
		};

		let usage = body.x_take::<Value>("usageMetadata").map(Self::into_usage).unwrap_or_default();

		Ok(GeminiChatResponse {
			content: final_content,
			reasoning_content: final_reasoning_content,
			usage,
		})
	}

	/// See gemini doc: https://ai.google.dev/api/generate-content#UsageMetadata
	pub(super) fn into_usage(mut usage_value: Value) -> Usage {
		let prompt_tokens: Option<i32> = usage_value.x_take("promptTokenCount").ok();
		let completion_tokens: Option<i32> = usage_value.x_take("candidatesTokenCount").ok();
		let total_tokens: Option<i32> = usage_value.x_take("totalTokenCount").ok();

		// IMPORTANT: For Gemini, the `thoughts_token_count` (~reasoning_tokens) is not included
		//            in the root `candidatesTokenCount` (~completion_tokens).
		//            Therefore, some computation is needed to normalize it in the "OpenAI API Way,"
		//            meaning `completion_tokens` represents the total of completion tokens,
		//            and the details provide a breakdown of the specific components.

		let (completion_tokens, completion_tokens_details) =
			match (completion_tokens, usage_value.x_get_i64("thoughtsTokenCount").ok()) {
				(Some(c_tokens), Some(t_tokens)) => {
					let t_tokens = t_tokens as i32; // should be safe enough
					(
						Some(c_tokens + t_tokens),
						Some(CompletionTokensDetails {
							accepted_prediction_tokens: Some(c_tokens),
							rejected_prediction_tokens: None,
							reasoning_tokens: Some(t_tokens),
							audio_tokens: None,
						}),
					)
				}
				(None, Some(t_tokens)) => {
					(
						None,
						Some(CompletionTokensDetails {
							accepted_prediction_tokens: None,
							rejected_prediction_tokens: None,
							reasoning_tokens: Some(t_tokens as i32), // should be safe enough
							audio_tokens: None,
						}),
					)
				}
				(c_tokens, None) => (c_tokens, None),
			};

		Usage {
			prompt_tokens,
			// for now, None for Gemini
			prompt_tokens_details: None,

			completion_tokens,

			completion_tokens_details,

			total_tokens,
		}
	}

	/// Takes the genai ChatMessages and builds the System string and JSON Messages for Gemini.
	/// - Role mapping `ChatRole:User -> role: "user"`, `ChatRole::Assistant -> role: "model"`
	/// - `ChatRole::System` is concatenated (with an empty line) into a single `system` for the system instruction.
	///   - This adapter uses version v1beta, which supports `systemInstruction`
	/// - The eventual `chat_req.system` is pushed first into the "systemInstruction"
	fn into_gemini_request_parts(
		model_iden: ModelIden,
		chat_req: ChatRequest,
		options_set: &ChatOptionsSet, // Pass options_set to check for gemini_enable_code_execution
	) -> Result<GeminiChatRequestParts> {
		let mut contents: Vec<Value> = Vec::new();
		let mut systems: Vec<String> = Vec::new();

		if let Some(system) = chat_req.system {
			systems.push(system);
		}

		// -- Build message contents
		for msg in chat_req.messages {
			match msg.role {
				ChatRole::System => {
					let MessageContent::Text(content) = msg.content else {
						return Err(Error::MessageContentTypeNotSupported {
							model_iden: model_iden.clone(),
							cause: "Only MessageContent::Text supported for System role (for now)",
						});
					};
					systems.push(content)
				}
				ChatRole::User => {
					let user_parts = match msg.content {
						MessageContent::Text(content) => json!([{"text": content}]),
						MessageContent::Parts(parts) => {
							json!(
								parts
									.iter()
									.map(|part| match part {
										ContentPart::Text(text) => json!({"text": text.clone()}),
										ContentPart::Image { content_type, source } => {
											match source {
												ImageSource::Url(url) => json!({
													"file_data": {
														"mime_type": content_type,
														"file_uri": url
													}
												}),
												ImageSource::Base64(content) => json!({
													"inline_data": {
														"mime_type": content_type,
														"data": content
													}
												}),
											}
										}
									})
									.collect::<Vec<Value>>()
							)
						}
						MessageContent::ToolCalls(_) => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden: model_iden.clone(),
								cause: "ChatRole::User with MessageContent::ToolCalls not supported for Gemini (ToolCalls should be from Assistant)",
							});
						}
						MessageContent::ToolResponses(tool_responses) => {
							json!(
								tool_responses
									.into_iter()
									.map(|tool_response| {
										json!({
											"functionResponse": {
												"name": tool_response.call_id,
												"response": {
													"name": tool_response.call_id, // Gemini seems to require name here too
													"content": serde_json::from_str(&tool_response.content).unwrap_or(Value::Null),
												}
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
					};
					contents.push(json!({"role": "user", "parts": user_parts}));
				}
				ChatRole::Assistant => {
					let model_parts = match msg.content {
						MessageContent::Text(content) => json!([{"text": content}]),
						MessageContent::ToolCalls(tool_calls) => {
							json!(
								tool_calls
									.into_iter()
									.map(|tool_call| {
										json!({
											"functionCall": {
												"name": tool_call.fn_name,
												"args": tool_call.fn_arguments,
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
						_ => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden,
								cause: "Assistant MessageContent must be Text or ToolCalls for Gemini",
							});
						}
					};
					contents.push(json!({"role": "model", "parts": model_parts}));
				}
				ChatRole::Tool => {
					// For Gemini, Tool role messages (function responses) are sent with role: "user"
					// and the content part is a "functionResponse".
					// This is handled by the User role block if msg.content is ToolResponses.
					// So, direct ChatRole::Tool might not be used if ToolResponses are always wrapped in User role messages.
					// However, if ChatRole::Tool arrives with ToolResponses, we handle it like a User message with ToolResponses.
					let tool_parts = match msg.content {
						MessageContent::ToolResponses(tool_responses) => {
							json!(
								tool_responses
									.into_iter()
									.map(|tool_response| {
										json!({
											"functionResponse": {
												"name": tool_response.call_id,
												"response": {
													"name": tool_response.call_id,
													"content": serde_json::from_str(&tool_response.content).unwrap_or(Value::Null),
												}
											}
										})
									})
									.collect::<Vec<Value>>()
							)
						}
						_ => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden,
								cause: "ChatRole::Tool must contain MessageContent::ToolResponses for Gemini",
							});
						}
					};
					// Gemini expects functionResponse to be from the "user" role.
					contents.push(json!({"role": "user", "parts": tool_parts}));
				}
			}
		}

		let system = if !systems.is_empty() {
			Some(systems.join("\n"))
		} else {
			None
		};

		// -- Build tools payload
		let mut tool_definitions: Vec<Value> = Vec::new();

		if options_set.gemini_enable_code_execution().unwrap_or(false) {
			tool_definitions.push(json!({"codeExecution": {}}));
		}

		if let Some(function_tools) = chat_req.tools {
			if !function_tools.is_empty() {
				let mapped_fn_declarations = function_tools
					.into_iter()
					.map(|tool| {
						json!({
							"name": tool.name,
							"description": tool.description,
							"parameters": tool.schema, // Assuming tool.schema is already a valid JSON Schema Value
						})
					})
					.collect::<Vec<Value>>();
				// Note: Gemini API expects functionDeclarations to be nested under a Tool object.
				// If codeExecution is also present, they should be separate items in the top-level tools array.
				tool_definitions.push(json!({"functionDeclarations": mapped_fn_declarations}));
			}
		}

		let final_tools_payload = if !tool_definitions.is_empty() {
			Some(Value::Array(tool_definitions))
		} else {
			None
		};

		Ok(GeminiChatRequestParts {
			system,
			contents,
			final_tools_payload,
		})
	}
}

// struct Gemini

pub(super) struct GeminiChatResponse {
	pub content: Option<GeminiChatContent>,
	pub reasoning_content: Option<GeminiChatContent>,
	pub usage: Usage,
}

pub(super) enum GeminiChatContent {
	Text(String),
	ToolCall(ToolCall),
}

struct GeminiChatRequestParts {
	system: Option<String>,
	/// The chat history (user and assistant, except for the last user message which is a message)
	contents: Vec<Value>,

	/// The final JSON value for the "tools" array in the API request.
	final_tools_payload: Option<Value>,
}

// endregion: --- Support
