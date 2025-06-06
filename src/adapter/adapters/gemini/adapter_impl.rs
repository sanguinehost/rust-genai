use super::streamer::GeminiStreamer;
use crate::adapter::adapters::support::get_api_key;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet,
	ChatRequest,
	ChatResponse,
	ChatResponseFormat,
	ChatRole,
	ChatStream,
	ChatStreamResponse,
	CompletionTokensDetails,
	ContentPart,
	ImagenGenerateImagesRequest,
	ImagenGenerateImagesResponse, // Added for Imagen
	MediaSource,                  // Added for document understanding
	MessageContent,
	PromptTokensDetails,
	ReasoningEffort,
	ToolCall,
	Usage,
	VeoGenerateVideosRequest,
	VeoGenerateVideosResponse,
	VeoOperationResult,
	VeoOperationStatusResponse,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::{WebResponse, WebStream};
use crate::{Error, ModelIden, Result, ServiceTarget};
use reqwest::RequestBuilder;
use serde_json::{Value, json};
use value_ext::JsonValueExt;

pub struct GeminiAdapter;

const MODELS: &[&str] = &[
	"gemini-2.0-flash",
	"gemini-2.0-flash-lite",
	"gemini-2.5-flash-preview-05-20",
	"gemini-2.5-pro-preview-05-06",
	"gemini-2.5-pro-preview-06-05",
	"gemini-1.5-pro",
	"gemini-2.0-flash-preview-image-generation",
	"imagen-3.0-generate-002",
	"veo-2.0-generate-001",
];

// Per gemini doc (https://x.com/jeremychone/status/1916501987371438372)
const REASONING_ZERO: u32 = 0;
const REASONING_LOW: u32 = 1000;
const REASONING_MEDIUM: u32 = 8000;
const REASONING_HIGH: u32 = 24000;

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
		Ok(MODELS.iter().map(ToString::to_string).collect())
	}
	/// NOTE: As Google Gemini has decided to put their `API_KEY` in the URL,
	///       this will return the URL without the `API_KEY` in it. The `API_KEY` will need to be added by the caller.
	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		let base_url = endpoint.base_url();
		let model_name = model.model_name.clone();
		match service_type {
			ServiceType::Chat => format!("{base_url}models/{model_name}:generateContent"),
			ServiceType::ChatStream => format!("{base_url}models/{model_name}:streamGenerateContent"),
			ServiceType::ImageGenerationImagen => format!("{base_url}models/{model_name}:predict"),
			ServiceType::VideoGenerationVeo => format!("{base_url}models/{model_name}:predictLongRunning"),
		}
	}

	#[allow(clippy::too_many_lines)]
	#[allow(clippy::cognitive_complexity)]
	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- api_key
		let api_key = get_api_key(&auth, &model)?;

		// -- Reasoning Budget
		let (model, reasoning_effort) = match (model, options_set.reasoning_effort()) {
			// No explicity reasoning_effor, try to infer from model name suffix (supports -zero)
			(model, None) => {
				let model_name: &str = &model.model_name;
				if let Some((prefix, last)) = model_name.rsplit_once('-') {
					let reasoning = match last {
						"zero" => Some(ReasoningEffort::Budget(REASONING_ZERO)),
						"low" => Some(ReasoningEffort::Budget(REASONING_LOW)),
						"medium" => Some(ReasoningEffort::Budget(REASONING_MEDIUM)),
						"high" => Some(ReasoningEffort::Budget(REASONING_HIGH)),
						_ => None,
					};
					// create the model name if there was a `-..` reasoning suffix
					let model = if reasoning.is_some() {
						model.from_name(prefix)
					} else {
						model
					};

					(model, reasoning)
				} else {
					(model, None)
				}
			}
			// If reasoning effort, turn the low, medium, budget ones into Budget
			(model, Some(effort)) => {
				let effort = match effort {
					ReasoningEffort::Low => ReasoningEffort::Budget(REASONING_LOW),
					ReasoningEffort::Medium => ReasoningEffort::Budget(REASONING_MEDIUM),
					ReasoningEffort::High => ReasoningEffort::Budget(REASONING_HIGH),
					ReasoningEffort::Budget(budget) => ReasoningEffort::Budget(*budget),
				};
				(model, Some(effort))
			}
		};

		// -- parts
		let GeminiChatRequestParts {
			system,
			contents,
			tools,
		} = Self::into_gemini_request_parts(model.clone(), chat_req)?;

		// -- Playload
		let mut payload = json!({
			"contents": contents,
		});

		// -- Set the reasoning effort
		if let Some(ReasoningEffort::Budget(budget)) = reasoning_effort {
			payload.x_insert("/generationConfig/thinkingConfig/thinkingBudget", budget)?;
		}

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
		if let Some(tools) = tools {
			payload.x_insert(
				"tools",
				json!({
					"function_declarations": tools
				}),
			)?;
		}

		// -- Response Format
		if let Some(response_format) = options_set.response_format() {
			match response_format {
				ChatResponseFormat::JsonSpec(st_json) => {
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
				ChatResponseFormat::EnumSpec(enum_spec) => {
					payload.x_insert("/generationConfig/responseMimeType", enum_spec.mime_type.clone())?;
					payload.x_insert("/generationConfig/responseSchema", enum_spec.schema.clone())?;
				}
				ChatResponseFormat::JsonSchemaSpec(json_schema_spec) => {
					let model_name_str = &model.model_name;
					// Models known/expected to support response_json_schema with v1alpha
					let supports_v1alpha_json_schema =
						matches!(model_name_str.as_ref(), "gemini-2.5-flash-preview-05-20" | "gemini-2.5-pro-preview-05-06" | "gemini-2.5-pro-preview-06-05");

					payload.x_insert("/generationConfig/responseMimeType", "application/json")?;
					if supports_v1alpha_json_schema {
						payload.x_insert("/generationConfig/response_json_schema", json_schema_spec.schema.clone())?;
						// URL modification to v1alpha will be handled later
					} else {
						tracing::warn!(
							"GeminiAdapter: ChatResponseFormat::JsonSchemaSpec used with model '{}' which may not support it via v1alpha endpoint or with 'response_json_schema'. \
							Falling back to 'responseSchema' (OpenAPI subset) on the default (v1beta) endpoint. This may not work as expected or be ignored by the model.",
							model_name_str
						);
						payload.x_insert("/generationConfig/responseSchema", json_schema_spec.schema.clone())?;
					}
				}
				ChatResponseFormat::JsonMode => {
					// Gemini does not have a direct "json_mode" like OpenAI.
					// The closest is to set responseMimeType to application/json without a schema.
					// However, the documentation recommends using responseSchema for constrained JSON.
					// For now, we will not set anything for JsonMode, as it's typically handled by prompt engineering.
					// If a schema is not provided, the model is not constrained to output JSON.
					tracing::warn!(
						"GeminiAdapter: ChatResponseFormat::JsonMode is not directly supported. Consider using JsonSpec with a schema for constrained JSON output."
					);
				}
			}
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

		// -- Add newly supported ChatOptions
		if let Some(top_k) = options_set.top_k() {
			payload.x_insert("/generationConfig/topK", top_k)?;
		}
		if let Some(seed) = options_set.seed() {
			payload.x_insert("/generationConfig/seed", seed)?;
		}
		if let Some(presence_penalty) = options_set.presence_penalty() {
			payload.x_insert("/generationConfig/presencePenalty", presence_penalty)?;
		}
		if let Some(frequency_penalty) = options_set.frequency_penalty() {
			payload.x_insert("/generationConfig/frequencyPenalty", frequency_penalty)?;
		}

		// -- Candidate Count
		if let Some(candidate_count) = options_set.candidate_count() {
			payload.x_insert("/generationConfig/candidateCount", candidate_count)?;
			// TODO: Adapt response parsing to handle multiple candidates if candidate_count > 1
		}

		// -- Cached Content
		if let Some(cached_content_id) = options_set.cached_content_id() {
			payload.x_insert("cachedContent", cached_content_id)?;
		}

		// -- Tool Config
		let mut tool_config_payload = json!({});
		let mut function_calling_config_payload = json!({});
		let mut function_calling_config_added = false;

		if let Some(mode) = options_set.function_calling_mode() {
			function_calling_config_payload.x_insert("mode", mode)?;
			function_calling_config_added = true;
		}

		if let Some(allowed_names) = options_set.allowed_function_names() {
			if !allowed_names.is_empty() {
				function_calling_config_payload.x_insert("allowedFunctionNames", allowed_names)?;
				function_calling_config_added = true;
			}
		}

		if function_calling_config_added {
			tool_config_payload.x_insert("functionCallingConfig", function_calling_config_payload)?;
			payload.x_insert("toolConfig", tool_config_payload)?;
		}

		// -- Response Modalities (for Gemini image generation)
		if let Some(modalities) = options_set.response_modalities() {
			if !modalities.is_empty() {
				payload.x_insert("/generationConfig/responseModalities", modalities)?;
			}
		}

		// -- url
		// NOTE: Somehow, Google decided to put the API key in the URL.
		//       This should be considered an antipattern from a security point of view
		//       even if it is done by the well respected Google. Everybody can make mistake once in a while.
		// e.g., '...models/gemini-1.5-flash-latest:generateContent?key=YOUR_API_KEY'
		let mut final_url = Self::get_service_url(&model, service_type, endpoint);

		// Potentially modify URL for v1alpha if JsonSchemaSpec is used with a compatible model
		if let Some(ChatResponseFormat::JsonSchemaSpec(_)) = options_set.response_format() {
			let model_name_str = &model.model_name;
			if matches!(model_name_str.as_ref(), "gemini-2.5-flash-preview-05-20" | "gemini-2.5-pro-preview-05-06" | "gemini-2.5-pro-preview-06-05") {
				// Only replace if the URL is indeed a v1beta URL from the default endpoint.
				if final_url.starts_with("https://generativelanguage.googleapis.com/v1beta/") {
					final_url = final_url.replacen(
						"https://generativelanguage.googleapis.com/v1beta/",
						"https://generativelanguage.googleapis.com/v1alpha/",
						1,
					);
				}
			}
		}

		let final_url_with_key = format!("{final_url}?key={api_key}");

		Ok(WebRequestData {
			url: final_url_with_key,
			headers,
			payload,
		})
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;
		// -- Capture the provider_model_iden
		// TODO: Need to be implemented (if available), for now, just clone model_iden
		let provider_model_name: Option<String> = body.x_remove("modelVersion").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		let gemini_response = Self::body_to_gemini_chat_response(&model_iden, body, &options_set)?;
		let GeminiChatResponse { contents, usage } = gemini_response;

		// Map Vec<GeminiChatContent> to Vec<MessageContent>
		// Each GeminiChatContent now directly corresponds to a MessageContent
		// (as GeminiChatContent can hold multiple parts or a tool call)
		let response_contents: Vec<MessageContent> = contents.into_iter().map(Into::into).collect();

		// If the original request was for a single candidate (or default),
		// and we got no content, the response_contents vec will be empty.
		// Otherwise, it will contain one or more items.

		Ok(ChatResponse {
			contents: response_contents,
			reasoning_content: None,
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
		let web_stream = WebStream::new_with_pretty_json_array(reqwest_builder);

		let gemini_stream = GeminiStreamer::new(web_stream, model_iden.clone(), &options_set);
		let chat_stream = ChatStream::from_inter_stream(gemini_stream);

		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}
}

// region:    --- Support

/// Support functions for `GeminiAdapter`
impl GeminiAdapter {
	pub(super) fn body_to_gemini_chat_response(
		model_iden: &ModelIden,
		mut body: Value,
		options_set: &ChatOptionsSet<'_, '_>,
	) -> Result<GeminiChatResponse> {
		// If the body has an `error` property, then it is assumed to be an error.
		if body.get("error").is_some() {
			return Err(Error::StreamEventError {
				model_iden: model_iden.clone(),
				body,
			});
		}

		// Take all candidates
		let candidates = body.x_take::<Vec<Value>>("/candidates")?;
		let mut gemini_contents: Vec<GeminiChatContent> = Vec::new();

		for mut candidate_json in candidates {
			// A candidate can have multiple parts (text, image) or a functionCall.
			let candidate_content_parts_json = candidate_json.x_take::<Vec<Value>>("/content/parts")?;
			let mut content_parts: Vec<ContentPart> = Vec::new();
			let mut has_image_content = false;

			for mut part_json in candidate_content_parts_json {
				if let Ok(fc_json) = part_json.x_take::<Value>("functionCall") {
					// This part is a function call
					gemini_contents.push(GeminiChatContent::ToolCall(ToolCall {
						call_id: fc_json.x_get("name").unwrap_or_else(|_| String::new()), // Assuming name is used as call_id for Gemini
						fn_name: fc_json.x_get("name").unwrap_or_else(|_| String::new()),
						fn_arguments: fc_json.x_get("args").unwrap_or(Value::Null),
					}));
					// Assuming a functionCall part is exclusive in a candidate's content.parts array for simplicity here.
					// If not, the logic to combine with other parts would be more complex.
					content_parts.clear(); // Clear any previously parsed text/image parts for this candidate
					break; // Move to the next candidate
				} else if let Ok(text_value) = part_json.x_take::<Value>("text") {
					// Attempt to convert the Value to a String. This handles cases where the 'text' field
					// might be a number, boolean, or null, which should not happen for valid text content,
					// but provides robustness. More importantly, it ensures the full string is captured.
					let text = text_value.as_str().map_or_else(
						|| {
							tracing::warn!("GeminiAdapter: 'text' part was not a string, converting to debug string. Value: {:?}", text_value);
							text_value.to_string() // Fallback to debug string representation
						},
						|s| s.to_string(),
					);
					content_parts.push(ContentPart::Text(text));
				} else if let Ok(mut inline_data_json) = part_json.x_take::<Value>("inlineData") {
					let mime_type = inline_data_json.x_take::<String>("mimeType")?;
					let data = inline_data_json.x_take::<String>("data")?;
					content_parts.push(ContentPart::Image {
						content_type: mime_type,
						source: MediaSource::Base64(data.into()),
					});
					has_image_content = true;
				} else if let Ok(mut file_data_json) = part_json.x_take::<Value>("fileData") {
					let mime_type = file_data_json.x_take::<String>("mimeType")?;
					let file_uri = file_data_json.x_take::<String>("fileUri")?;
					// Determine if it's an image or document based on mime_type or context.
					// For simplicity, assuming PDF for now, but could be extended.
					if mime_type.starts_with("image/") {
						content_parts.push(ContentPart::Image {
							content_type: mime_type,
							source: MediaSource::Url(file_uri),
						});
						has_image_content = true;
					} else if mime_type.starts_with("application/pdf") {
						content_parts.push(ContentPart::Document {
							content_type: mime_type,
							source: MediaSource::Url(file_uri),
						});
					} else {
						// Handle other file types if necessary, or return an error
						tracing::warn!("Unsupported fileData mime_type: {}", mime_type);
					}
				}
			}

			if !content_parts.is_empty() {
				// If response_modalities requested an image but none was found in this candidate's parts
				if options_set
					.response_modalities()
					.is_some_and(|m| m.contains(&"IMAGE".to_string()))
					&& !has_image_content
				{
					// Add a warning part, as per user preference
					content_parts.push(ContentPart::Text(
						"[WARN] Image was requested but not generated for this candidate.".to_string(),
					));
				}
				gemini_contents.push(GeminiChatContent::Parts(content_parts));
			} else if !gemini_contents.iter().any(|gc| matches!(gc, GeminiChatContent::ToolCall(_))) {
				// If no parts were processed and it wasn't a tool call, it might be an error or empty response for this candidate
				println!(
					"Warning: Candidate resulted in no processable content (text, image, or tool_call): {candidate_json:?}"
				);
			}
		}

		// Usage is typically overall for the request, not per candidate.
		let usage = body.x_take::<Value>("usageMetadata").map(Self::into_usage).unwrap_or_default();

		Ok(GeminiChatResponse {
			contents: gemini_contents,
			usage,
		})
	}

	/// See gemini doc: <https://ai.google.dev/api/generate-content#UsageMetadata>
	pub(super) fn into_usage(mut usage_value: Value) -> Usage {
		let total_tokens: Option<i32> = usage_value.x_take("totalTokenCount").ok();

		// -- Compute prompt tokens
		let prompt_tokens: Option<i32> = usage_value.x_take("promptTokenCount").ok();
		// Note: https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/
		//       It does say `cached_content_token_count`, but in the json, it's probably
		//       `cachedContenTokenCount` (Could not verify for implicit cache, did not see it yet)
		// Note: It seems the promptTokenCount is inclusive of the cachedContentTokenCount
		//       see: https://ai.google.dev/gemini-api/docs/caching?lang=python#generate-content
		//       (this was for explicit caching, but should be the same for implicit)
		//       ```
		//       prompt_token_count: 696219
		//       cached_content_token_count: 696190
		//       candidates_token_count: 214
		//       total_token_count: 696433
		//       ```
		//       So, in short same as Open asi
		let g_cached_tokens: Option<i32> = usage_value.x_take("cachedContentTokenCount").ok();
		let prompt_tokens_details = g_cached_tokens.map(|g_cached_tokens| PromptTokensDetails {
			cache_creation_tokens: None,
			cached_tokens: Some(g_cached_tokens),
			audio_tokens: None,
		});

		// -- Compute completion tokens
		let g_candidate_tokens: Option<i32> = usage_value.x_take("candidatesTokenCount").ok();
		let g_thoughts_tokens: Option<i32> = usage_value.x_take("thoughtsTokenCount").ok();
		// IMPORTANT: For Gemini, the `thoughtsTokenCount` (~reasoning_tokens) is not included
		//            in the root `candidatesTokenCount` (~completion_tokens).
		//            Therefore, some computation is needed to normalize it in the "OpenAI API Way,"
		//            meaning `completion_tokens` represents the total of completion tokens,
		//            and the details provide a breakdown of the specific components.
		let (completion_tokens, completion_tokens_details) = match (g_candidate_tokens, g_thoughts_tokens) {
			(Some(c_tokens), Some(t_tokens)) => (
				Some(c_tokens + t_tokens),
				Some(CompletionTokensDetails {
					accepted_prediction_tokens: None,
					rejected_prediction_tokens: None,
					reasoning_tokens: Some(t_tokens),
					audio_tokens: None,
				}),
			),
			(None, Some(t_tokens)) => {
				(
					None,
					Some(CompletionTokensDetails {
						accepted_prediction_tokens: None,
						rejected_prediction_tokens: None,
						reasoning_tokens: Some(t_tokens), // should be safe enough
						audio_tokens: None,
					}),
				)
			}
			(c_tokens, None) => (c_tokens, None),
		};

		Usage {
			prompt_tokens,
			// for now, None for Gemini
			prompt_tokens_details,

			completion_tokens,

			completion_tokens_details,

			total_tokens,
		}
	}

	/// Takes the genai `ChatMessages` and builds the System string and JSON Messages for Gemini.
	/// - Role mapping `ChatRole:User -> role: "user"`, `ChatRole::Assistant -> role: "model"`
	/// - `ChatRole::System` is concatenated (with an empty line) into a single `system` for the system instruction.
	///   - This adapter uses version v1beta, which supports `systemInstruction`
	/// - The eventual `chat_req.system` is pushed first into the "systemInstruction"
	#[allow(clippy::too_many_lines)]
	fn into_gemini_request_parts(model_iden: ModelIden, chat_req: ChatRequest) -> Result<GeminiChatRequestParts> {
		let mut contents: Vec<Value> = Vec::new();
		let mut systems: Vec<String> = Vec::new();

		if let Some(system) = chat_req.system {
			systems.push(system);
		}

		// -- Build
		for msg in chat_req.messages {
			match msg.role {
				// For now, system goes as "user" (later, we might have adapter_config.system_to_user_impl)
				ChatRole::System => {
					let MessageContent::Text(content) = msg.content else {
						return Err(Error::MessageContentTypeNotSupported {
							model_iden,
							cause: "Only MessageContent::Text supported for this model (for now)",
						});
					};
					systems.push(content);
				}
				ChatRole::User => {
					let content = match msg.content {
						MessageContent::Text(content) => json!([{"text": content}]),
						MessageContent::Parts(parts) => {
							json!(
								parts
									.iter()
									.map(|part| match part {
										ContentPart::Text(text) => json!({"text": text.clone()}),
										ContentPart::Image { content_type, source }
										| ContentPart::Document { content_type, source } => {
											match source {
												MediaSource::Url(url) => json!({
													"file_data": {
														"mime_type": content_type,
														"file_uri": url
													}
												}),
												MediaSource::Base64(content) => json!({
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
					};

					contents.push(json!({"role": "user", "parts": content}));
				}
				ChatRole::Assistant => {
					match msg.content {
						MessageContent::Text(content) => {
							contents.push(json!({"role": "model", "parts": [{"text": content}]}));
						}
						MessageContent::ToolCalls(tool_calls) => contents.push(json!({
							"role": "model",
							"parts": tool_calls
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
						})),
						_ => {
							return Err(Error::MessageContentTypeNotSupported {
								model_iden,
								cause: "Only MessageContent::Text and MessageContent::ToolCalls supported for this model (for now)",
							});
						}
					};
				}
				ChatRole::Tool => {
					let content = match msg.content {
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
								cause: "ChatRole::Tool can only be MessageContent::ToolCall or MessageContent::ToolResponse",
							});
						}
					};

					contents.push(json!({"role": "user", "parts": content}));
				}
			}
		}

		let system = if systems.is_empty() {
			None
		} else {
			Some(systems.join("\n"))
		};

		let tools = chat_req.tools.map(|tools| {
			tools
				.into_iter()
				.map(|tool| {
					// TODO: Need to handle the error correctly
					// TODO: Needs to have a custom serializer (tool should not have to match to a provider)
					// NOTE: Right now, low probability, so, we just return null if cannot convert to value.
					json!({
						"name": tool.name,
						"description": tool.description,
						"parameters": tool.schema,
					})
				})
				.collect::<Vec<Value>>()
		});

		Ok(GeminiChatRequestParts {
			system,
			contents,
			tools,
		})
	}

	// -- Imagen 3 Specific Methods --
	pub fn to_imagen_generation_request_data(
		target: ServiceTarget,
		request: ImagenGenerateImagesRequest,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;
		let api_key = get_api_key(&auth, &model)?;
		let url = Self::get_service_url(&model, ServiceType::ImageGenerationImagen, endpoint);
		let url = format!("{url}?key={api_key}");

		// Construct the payload for Imagen 3
		// The request struct `ImagenGenerateImagesRequest` is already designed
		// to serialize into the format Imagen expects for the "instances" part.
		// We need to wrap it in an "instances" array and add "parameters".
		let mut parameters = json!({});
		if let Some(count) = request.number_of_images {
			parameters.x_insert("sampleCount", count)?;
		}
		if let Some(ratio) = &request.aspect_ratio {
			parameters.x_insert("aspectRatio", ratio)?;
		}
		if let Some(pg) = &request.person_generation {
			parameters.x_insert("personGeneration", pg)?;
		}
		if let Some(seed) = request.seed {
			parameters.x_insert("seed", seed)?;
		}
		// Negative prompt is part of the main request struct for Imagen, not parameters.

		let instance_payload = serde_json::to_value(request)?;

		let payload = json!({
			"instances": [instance_payload],
			"parameters": parameters,
		});

		let headers = vec![("Content-Type".to_string(), "application/json".to_string())];

		Ok(WebRequestData { url, headers, payload })
	}

	pub fn to_imagen_generation_response(
		model_iden: ModelIden,
		mut web_response: WebResponse,
	) -> Result<ImagenGenerateImagesResponse> {
		let provider_model_name: Option<String> = None; // Imagen predict API might not return this in the same way
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// The Imagen response has a `predictions` array.
		// Each item in `predictions` should match `ImagenGeneratedImage` structure.
		// {
		//   "predictions": [
		//     {
		//       "bytesBase64Encoded": "...",
		//       "seed": 12345,
		//       "finishReason": "DONE"
		//     }
		//   ]
		// }
		let generated_images = web_response
			.body
			.x_take::<Vec<crate::chat::ImagenGeneratedImage>>("predictions")?;

		Ok(ImagenGenerateImagesResponse {
			generated_images,
			usage: None, // Imagen API might not provide usage data in the same way as chat
			model_iden,
			provider_model_iden,
		})
	}

	// -- Veo Specific Methods --
	pub fn to_veo_generation_request_data(
		target: ServiceTarget,
		request: VeoGenerateVideosRequest,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;
		let api_key = get_api_key(&auth, &model)?;
		let url = Self::get_service_url(&model, ServiceType::VideoGenerationVeo, endpoint);
		let url = format!("{url}?key={api_key}");

		let mut instance_content = json!({});
		if let Some(prompt) = request.prompt {
			instance_content.x_insert("prompt", prompt)?;
		}
		if let Some(image_input) = request.image {
			// Serialize VeoImageInput and insert it under "image"
			let image_payload = serde_json::to_value(image_input)?;
			instance_content.x_insert("image", image_payload)?;
		}

		let mut parameters = json!({});
		if let Some(ratio) = &request.aspect_ratio {
			parameters.x_insert("aspectRatio", ratio)?;
		}
		if let Some(pg) = &request.person_generation {
			parameters.x_insert("personGeneration", pg)?;
		}
		if let Some(count) = request.number_of_videos {
			parameters.x_insert("sampleCount", count)?;
		}
		if let Some(duration) = request.duration_seconds {
			parameters.x_insert("durationSeconds", duration)?;
		}
		if let Some(enhance) = request.enhance_prompt {
			parameters.x_insert("enhancePrompt", enhance)?;
		}
		if let Some(neg_prompt) = request.negative_prompt {
			parameters.x_insert("negativePrompt", neg_prompt)?;
		}

		let payload = json!({
			"instances": [instance_content],
			"parameters": parameters,
		});

		let headers = vec![("Content-Type".to_string(), "application/json".to_string())];

		let mut redacted_payload = payload.clone();
		if let Some(instances) = redacted_payload["instances"].as_array_mut() {
			for instance in instances {
				// Redact bytes if it exists under image in the instance
				if let Some(image_obj) = instance.get_mut("image") {
					if let Some(image_map) = image_obj.as_object_mut() {
						if image_map.contains_key("bytes") {
							image_map.insert("bytes".to_string(), Value::String("<redacted>".to_string()));
						}
					}
				}
			}
		}

		tracing::trace!(
			target: "genai_gemini_adapter",
			"Veo Generation Request Payload: {}",
			serde_json::to_string_pretty(&redacted_payload).unwrap_or_else(|e| format!("Failed to serialize payload: {e}"))
		);

		Ok(WebRequestData { url, headers, payload })
	}

	pub fn to_veo_generation_response(
		model_iden: ModelIden,
		mut web_response: WebResponse,
	) -> Result<VeoGenerateVideosResponse> {
		let provider_model_name: Option<String> = None; // Veo predict API might not return this in the same way
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// The initial response for Veo is a long-running operation name.
		// { "name": "operations/..." }
		let operation_name = web_response.body.x_take::<String>("name")?;

		Ok(VeoGenerateVideosResponse {
			operation_name,
			model_iden,
			provider_model_iden,
		})
	}

	pub fn get_veo_operation_status_request_data(
		target: ServiceTarget,
		operation_name: &str,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;
		let api_key = get_api_key(&auth, &model)?;

		// The URL for polling an operation is different: BASE_URL/operations/{operation_name}?key={API_KEY}
		let base_url = endpoint.base_url();
		let url = format!("{base_url}{operation_name}?key={api_key}");

		let headers = vec![]; // No specific headers needed for GET
		let payload = json!({}); // No payload for GET

		tracing::trace!(
			target: "genai_gemini_adapter",
			"Veo Get Operation Status Request URL: {}, Payload: {}",
			url,
			serde_json::to_string_pretty(&payload).unwrap_or_else(|e| format!("Failed to serialize payload: {e}"))
		);

		Ok(WebRequestData { url, headers, payload })
	}

	pub fn to_veo_operation_status_response(
		model_iden: ModelIden,
		mut web_response: WebResponse,
	) -> Result<VeoOperationStatusResponse> {
		tracing::trace!(
			target: "genai_gemini_adapter",
			"Veo Operation Status Polling Response Body: {}",
			serde_json::to_string_pretty(&web_response.body).unwrap_or_else(|e| format!("Failed to serialize body: {e}"))
		);
		let provider_model_name: Option<String> = None;
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// Check for a top-level error object first. If present, it's an API error response.
		if let Ok(error_value) = web_response.body.x_take::<Value>("error") {
			return Err(Error::WebModelCall {
				model_iden,
				webc_error: crate::webc::Error::ResponseFailedStatus {
					status: reqwest::StatusCode::BAD_REQUEST, // Use appropriate StatusCode
					body: error_value.to_string(),
				},
			});
		}

		// If no top-level error, proceed to parse the expected operation status fields.
		// We need to clone the body if we want to log it fully in case of multiple parsing errors,
		// as x_take consumes parts of it. For now, log what's available at point of error.
		// Try to parse 'done'. If not found, assume false. If other parsing error, return Err.
		let actual_done = match web_response.body.x_take::<bool>("done") {
			Ok(d_val) => d_val,
			Err(ref e) if matches!(e, value_ext::JsonValueExtError::PropertyNotFound(_)) => {
				// Log that 'done' was not found and we're defaulting to false.
				// The body state here will be *after* attempting to take 'done'.
				eprintln!(
					"GeminiAdapter::to_veo_operation_status_response - 'done' field not found in polling response, assuming not completed. Body after attempt to take 'done': {:?}. Original error: {}",
					web_response.body, e
				);
				false // Assume not done if the field is missing
			}
			Err(e) => {
				// For any other error (e.g., WrongType), it's a genuine parsing problem.
				eprintln!(
					"GeminiAdapter::to_veo_operation_status_response - Error parsing 'done' field (not PropertyNotFound). Body after attempt to take 'done': {:?}. Error: {}",
					web_response.body, e
				);
				return Err(Error::from(e));
			}
		};

		// 'name' should always be present in operation status responses.
		let name = web_response.body.x_take::<String>("name").map_err(|e| {
			eprintln!(
				"GeminiAdapter::to_veo_operation_status_response - Error parsing 'name' field. Current body state (after 'done' processing): {:?}. Error: {}",
				web_response.body, e
			);
			Error::from(e)
		})?;

		// This 'error' field is part of the operation object itself, distinct from the top-level 'error'
		// checked at the beginning of this function. It indicates an error with the async operation.
		let operation_error_value = web_response.body.x_take::<Value>("error").ok();

		let response_result = if actual_done {
			if operation_error_value.is_some() {
				// If the operation is done and there's an error field, no successful response is expected.
				None
			} else {
				// If done and no operation_error_value, try to parse the successful response.
				match web_response.body.x_take::<VeoOperationResult>("response") {
					Ok(res) => Some(res),
					Err(e) => {
						// This case means done=true, no operation_error_value, but 'response' field is missing or malformed.
						eprintln!(
							"GeminiAdapter::to_veo_operation_status_response - Error parsing 'response' field when done=true and no operation error. Current body state: {:?}. Error: {:?}",
							web_response.body, e
						);
						// Propagate the error instead of returning None for the response.
						return Err(Error::from(e));
					}
				}
			}
		} else {
			// Not done yet.
			None
		};

		Ok(VeoOperationStatusResponse {
			done: actual_done,
			name,
			response: response_result,
			error: operation_error_value, // This is the error from the operation object
			model_iden,
			provider_model_iden,
		})
	}
}

// struct Gemini

pub(super) struct GeminiChatResponse {
	pub contents: Vec<GeminiChatContent>,
	pub usage: Usage,
}

pub(super) enum GeminiChatContent {
	Parts(Vec<ContentPart>),
	ToolCall(ToolCall),
	// Note: A simple Text variant might be redundant if Parts can contain a single Text part.
	// However, keeping it might simplify some direct text-only response scenarios if they exist.
	// For now, focusing on Parts for flexibility with multi-modal and ToolCall.
}

impl From<GeminiChatContent> for MessageContent {
	fn from(gemini_content: GeminiChatContent) -> Self {
		match gemini_content {
			GeminiChatContent::Parts(parts) => {
				// For backward compatibility, if we have a single text part, convert to Text
				if parts.len() == 1 {
					match &parts[0] {
						ContentPart::Text(text) => Self::Text(text.clone()),
						ContentPart::Image { .. } | ContentPart::Document { .. } => Self::Parts(parts),
					}
				} else {
					Self::Parts(parts)
				}
			}
			GeminiChatContent::ToolCall(tool_call) => Self::ToolCalls(vec![tool_call]),
		}
	}
}

struct GeminiChatRequestParts {
	system: Option<String>,
	/// The chat history (user and assistant, except for the last user message which is a message)
	contents: Vec<Value>,

	/// The tools to use
	tools: Option<Vec<Value>>,
}

// endregion: --- Support
