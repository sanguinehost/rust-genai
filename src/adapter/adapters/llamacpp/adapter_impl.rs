//! Core LlamaCpp adapter implementation for native local model inference.

use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use tokio::sync::mpsc;

use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
    ChatMessage, ChatOptionsSet, ChatRequest, ChatResponse, ChatRole, ChatStreamResponse,
    MessageContent, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Error, ModelIden, Result, ServiceTarget};
use reqwest::RequestBuilder;

use super::model_manager::{resolve_model_path, ModelManager};
use super::streamer::{create_streaming_channel, StreamChunk};
use super::schema_to_grammar::tools_to_gbnf;
use super::tool_templates::{detect_tool_config, apply_tool_template};
use super::tool_parser::{parse_tool_calls, contains_tool_calls};

/// Extract options that we need to pass to threads
#[derive(Clone, Debug)]
struct GenerationOptions {
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<i32>,
}

impl GenerationOptions {
    fn from_options_set(options: &ChatOptionsSet<'_, '_>) -> Self {
        Self {
            max_tokens: options.max_tokens(),
            temperature: options.temperature(),
            top_p: options.top_p(),
            seed: options.seed().map(|s| s as i32),
        }
    }
}

/// LlamaCpp adapter for native local model inference using llama.cpp bindings
pub struct LlamaCppAdapter;

impl Adapter for LlamaCppAdapter {
    fn default_endpoint() -> Endpoint {
        // Use a file:// scheme to indicate local model directory
        const BASE_PATH: &str = "file://~/.cache/llama-models/";
        Endpoint::from_static(BASE_PATH)
    }

    fn default_auth() -> AuthData {
        // No authentication needed for local models
        AuthData::from_single("llamacpp")
    }

    /// List available models in the model directory
    async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
        // For now, return a basic list. In practice, this could scan directories
        // for .gguf files or read from a config file
        Ok(vec![
            "llama-3.2.gguf".to_string(),
            "mistral-7b.gguf".to_string(),
            "phi-3.gguf".to_string(),
        ])
    }

    fn get_service_url(_model: &ModelIden, _service_type: ServiceType, _endpoint: Endpoint) -> Result<String> {
        // Not used for native adapter - we load models directly from filesystem
        Ok(String::new())
    }

    fn to_web_request_data(
        _target: ServiceTarget,
        _service_type: ServiceType,
        _chat_req: ChatRequest,
        _chat_options: ChatOptionsSet<'_, '_>,
    ) -> Result<WebRequestData> {
        // Not used for native adapter
        Err(Error::AdapterNotSupported {
            adapter_kind: AdapterKind::LlamaCpp,
            feature: "web requests (uses native execution)".to_string(),
        })
    }

    fn to_chat_response(
        _model_iden: ModelIden,
        _web_response: WebResponse,
        _options_set: ChatOptionsSet<'_, '_>,
    ) -> Result<ChatResponse> {
        // Not used for native adapter
        Err(Error::AdapterNotSupported {
            adapter_kind: AdapterKind::LlamaCpp,
            feature: "web responses (uses native execution)".to_string(),
        })
    }

    fn to_chat_stream(
        _model_iden: ModelIden,
        _reqwest_builder: RequestBuilder,
        _options_set: ChatOptionsSet<'_, '_>,
    ) -> Result<ChatStreamResponse> {
        // Not used for native adapter
        Err(Error::AdapterNotSupported {
            adapter_kind: AdapterKind::LlamaCpp,
            feature: "web streams (uses native streaming)".to_string(),
        })
    }

    fn to_embed_request_data(
        _service_target: ServiceTarget,
        _embed_req: crate::embed::EmbedRequest,
        _options_set: crate::embed::EmbedOptionsSet<'_, '_>,
    ) -> Result<WebRequestData> {
        // Not used for native adapter
        Err(Error::AdapterNotSupported {
            adapter_kind: AdapterKind::LlamaCpp,
            feature: "embeddings".to_string(),
        })
    }

    fn to_embed_response(
        _model_iden: ModelIden,
        _web_response: WebResponse,
        _options_set: crate::embed::EmbedOptionsSet<'_, '_>,
    ) -> Result<crate::embed::EmbedResponse> {
        // Not used for native adapter
        Err(Error::AdapterNotSupported {
            adapter_kind: AdapterKind::LlamaCpp,
            feature: "embeddings".to_string(),
        })
    }
}

impl LlamaCppAdapter {
    /// Execute chat completion natively using llama.cpp
    pub async fn exec_chat_native(
        target: ServiceTarget,
        chat_req: ChatRequest,
        options: ChatOptionsSet<'_, '_>,
    ) -> Result<ChatResponse> {
        let (_tx, _) = mpsc::channel::<StreamChunk>(1); // We don't use streaming for non-stream requests
        
        let generation_options = GenerationOptions::from_options_set(&options);
        let result = Self::generate_response(target, chat_req, generation_options, None).await?;
        Ok(result)
    }

    /// Execute streaming chat completion natively using llama.cpp
    pub async fn exec_chat_stream_native(
        target: ServiceTarget,
        chat_req: ChatRequest,
        options: ChatOptionsSet<'_, '_>,
    ) -> Result<ChatStreamResponse> {
        let (tx, stream_response) = create_streaming_channel(target.model.clone());

        // Extract options that can be sent across thread boundaries
        let generation_options = GenerationOptions::from_options_set(&options);

        // Extract fields we need from target (ServiceTarget doesn't impl Clone)
        let model_iden = target.model.clone();
        let base_url = target.endpoint.base_url().to_string();

        // Since llama.cpp types are not Send, we'll generate in a blocking task
        // and then send the chunks
        let chat_req_clone = chat_req.clone();
        let tx_clone = tx.clone();

        tokio::task::spawn_blocking(move || {
            // Generate the response and send chunks
            match futures::executor::block_on(Self::generate_response_blocking_with_data(model_iden, base_url, chat_req_clone, generation_options, Some(tx_clone))) {
                Ok(_) => {
                    // Done is sent from within generate_response_blocking
                }
                Err(e) => {
                    let _ = futures::executor::block_on(tx.send(StreamChunk::Error(e.to_string())));
                }
            }
        });

        Ok(stream_response)
    }

    /// Internal method to generate responses with optional streaming (blocking version)
    async fn generate_response_blocking(
        target: ServiceTarget,
        chat_req: ChatRequest,
        options: GenerationOptions,
        stream_tx: Option<mpsc::Sender<StreamChunk>>,
    ) -> Result<ChatResponse> {
        // Generate the response
        let result = Self::generate_response(target, chat_req, options, stream_tx.clone()).await;

        // Send done message for streaming
        if let Some(tx) = &stream_tx {
            match &result {
                Ok(_) => {
                    let _ = tx.send(StreamChunk::Done).await;
                }
                Err(_) => {
                    // Error already handled in the calling spawn_blocking
                }
            }
        }

        result
    }

    /// Helper for streaming that doesn't require full ServiceTarget
    async fn generate_response_blocking_with_data(
        model_iden: ModelIden,
        base_url: String,
        chat_req: ChatRequest,
        options: GenerationOptions,
        stream_tx: Option<mpsc::Sender<StreamChunk>>,
    ) -> Result<ChatResponse> {
        // Resolve model path
        let base_path = if base_url.starts_with("file://") {
            Some(&base_url["file://".len()..])
        } else {
            None
        };

        let model_path = resolve_model_path(&model_iden.model_name, base_path)?;

        // Generate the actual response
        let result = Self::generate_with_model_path(&model_iden, model_path, chat_req, options, stream_tx.clone()).await;

        // Send done message for streaming
        if let Some(tx) = &stream_tx {
            match &result {
                Ok(_) => {
                    let _ = tx.send(StreamChunk::Done).await;
                }
                Err(_) => {
                    // Error already handled in the calling spawn_blocking
                }
            }
        }

        result
    }

    /// Core generation logic that doesn't depend on ServiceTarget
    async fn generate_with_model_path(
        model_iden: &ModelIden,
        model_path: PathBuf,
        chat_req: ChatRequest,
        options: GenerationOptions,
        stream_tx: Option<mpsc::Sender<StreamChunk>>,
    ) -> Result<ChatResponse> {
        // Load model using ModelManager
        let model_manager = ModelManager::instance().await?;
        let loaded_model = model_manager.load_model(&model_path).await?;
        // Get backend from model manager
        let backend = model_manager.backend();

        // Detect tool configuration for this model
        let tool_config = detect_tool_config(&model_iden.model_name);

        // Create prompt with tool support if tools are provided
        let prompt = if let Some(tools) = &chat_req.tools {
            if !tools.is_empty() && tool_config.supports_tools {
                // Use model-specific tool template
                apply_tool_template(&chat_req.messages, tools, &tool_config)?
            } else {
                // No tools or model doesn't support tools, use regular chat template
                Self::create_regular_prompt(&chat_req.messages, &loaded_model)?
            }
        } else {
            // No tools, use regular chat template
            Self::create_regular_prompt(&chat_req.messages, &loaded_model)?
        };

        // Create context for generation
        let mut context_params = LlamaContextParams::default();

        // Configure context based on options
        if let Some(max_tokens) = options.max_tokens {
            // Set context size to accommodate input + output
            let context_size = (max_tokens * 2).max(2048);
            if let Some(ctx_size) = NonZeroU32::new(context_size) {
                context_params = context_params.with_n_ctx(Some(ctx_size));
            }
        }

        // Create context (llama.cpp contexts cannot be safely sent between threads)
        let mut context = loaded_model.model
            .new_context(backend, context_params)
            .map_err(|e| Error::Internal(format!("Failed to create context: {e}")))?;

        // Tokenize prompt
        let tokens = loaded_model.model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| Error::Internal(format!("Failed to tokenize prompt: {e}")))?;

        // Generate response
        let tokens_len = tokens.len();
        let generated_text = Self::generate_tokens(
            &loaded_model.model,
            &mut context,
            tokens,
            &options,
            stream_tx,
            chat_req.tools.as_deref(),
            &tool_config,
        )?;

        // Parse tool calls if present
        let content = if let Some(tools) = &chat_req.tools {
            if !tools.is_empty() && contains_tool_calls(&generated_text, &tool_config) {
                // Try to parse tool calls from the generated text
                let tool_calls = parse_tool_calls(&generated_text, &tool_config)?;
                if !tool_calls.is_empty() {
                    MessageContent::from_tool_calls(tool_calls)
                } else {
                    MessageContent::from_text(generated_text)
                }
            } else {
                MessageContent::from_text(generated_text)
            }
        } else {
            MessageContent::from_text(generated_text)
        };

        Ok(ChatResponse {
            content,
            reasoning_content: None,
            model_iden: model_iden.clone(),
            provider_model_iden: model_iden.clone(),
            usage: crate::chat::Usage {
                prompt_tokens: Some(tokens_len as i32),
                completion_tokens: None,
                total_tokens: None,
                ..Default::default()
            },
            captured_raw_body: None,
        })
    }

    /// Internal method to generate responses with optional streaming
    async fn generate_response(
        target: ServiceTarget,
        chat_req: ChatRequest,
        options: GenerationOptions,
        stream_tx: Option<mpsc::Sender<StreamChunk>>,
    ) -> Result<ChatResponse> {
        // Resolve model path
        let base_path = if target.endpoint.base_url().starts_with("file://") {
            Some(&target.endpoint.base_url()["file://".len()..])
        } else {
            None
        };

        let model_path = resolve_model_path(&target.model.model_name, base_path)?;

        // Delegate to the core generation logic
        Self::generate_with_model_path(&target.model, model_path, chat_req, options, stream_tx).await
    }

    /// Generate tokens using llama.cpp (synchronous because llama.cpp is not async)
    fn generate_tokens(
        model: &llama_cpp_2::model::LlamaModel,
        context: &mut LlamaContext<'_>,
        input_tokens: Vec<LlamaToken>,
        options: &GenerationOptions,
        stream_tx: Option<mpsc::Sender<StreamChunk>>,
        tools: Option<&[crate::chat::Tool]>,
        tool_config: &super::tool_templates::ToolConfig,
    ) -> Result<String> {
        let max_tokens = options.max_tokens.unwrap_or(512) as usize;
        let n_ctx = context.n_ctx() as usize;
        
        // Prepare batch
        let mut batch = LlamaBatch::new(512, 1);
        
        // Add input tokens to batch
        let last_index = (input_tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(input_tokens.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)
                .map_err(|e| Error::Internal(format!("Failed to add token to batch: {e}")))?;
        }

        // Process initial batch
        context.decode(&mut batch)
            .map_err(|e| Error::Internal(format!("Failed to decode initial batch: {e}")))?;

        // Prepare sampler
        let mut sampler = Self::create_sampler(options, model, tools, Some(tool_config))?;
        
        let mut generated_text = String::new();
        let mut n_cur = batch.n_tokens();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Generation loop
        for _ in 0..max_tokens {
            // Sample next token
            let token = sampler.sample(context, (batch.n_tokens() - 1) as i32);
            sampler.accept(token);

            // Check for end of generation
            if model.is_eog_token(token) {
                break;
            }

            // Convert token to text
            let token_bytes = model.token_to_bytes(token, Special::Tokenize)
                .map_err(|e| Error::Internal(format!("Failed to convert token to bytes: {e}")))?;
            
            let mut token_text = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&token_bytes, &mut token_text, false);
            
            generated_text.push_str(&token_text);

            // Send streaming chunk if streaming is enabled
            if let Some(tx) = &stream_tx {
                if let Err(_) = futures::executor::block_on(tx.send(StreamChunk::Delta(token_text))) {
                    // Receiver dropped, stop generation
                    break;
                }
            }

            // Prepare for next iteration
            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)
                .map_err(|e| Error::Internal(format!("Failed to add token to batch: {e}")))?;

            // Decode next token
            context.decode(&mut batch)
                .map_err(|e| Error::Internal(format!("Failed to decode batch: {e}")))?;

            n_cur += 1;

            // Check context limit
            if n_cur >= n_ctx as i32 {
                break;
            }
        }

        Ok(generated_text)
    }

    /// Convert genai ChatMessages to llama.cpp LlamaChatMessage format
    fn convert_chat_messages(messages: &[ChatMessage]) -> Result<Vec<LlamaChatMessage>> {
        let mut llama_messages = Vec::new();
        
        for message in messages {
            let role = match message.role {
                ChatRole::System => "system",
                ChatRole::User => "user", 
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            
            // Extract text content from MessageContent
            let content = Self::extract_text_content(&message.content)?;
            
            let llama_message = LlamaChatMessage::new(role.to_string(), content)
                .map_err(|e| Error::Internal(format!("Failed to create chat message: {e}")))?;
            
            llama_messages.push(llama_message);
        }
        
        Ok(llama_messages)
    }

    /// Extract text content from MessageContent
    fn extract_text_content(content: &MessageContent) -> Result<String> {
        // First, try to get text content
        if content.contains_text() {
            let texts = content.texts();
            return Ok(texts.join("\n"));
        }

        // If there are tool calls, convert them to text representation
        if content.contains_tool_call() {
            let tool_calls = content.tool_calls();
            let mut text_content = String::new();
            for tool_call in tool_calls {
                text_content.push_str(&format!(
                    "Tool call: {} with arguments: {}\n",
                    tool_call.fn_name,
                    tool_call.fn_arguments
                ));
            }
            return Ok(text_content);
        }

        // If there are tool responses, convert them to text
        if content.contains_tool_response() {
            let tool_responses = content.tool_responses();
            let mut text_content = String::new();
            for response in tool_responses {
                text_content.push_str(&format!(
                    "Tool response for {}: {}\n",
                    response.call_id,
                    response.content
                ));
            }
            return Ok(text_content);
        }

        // If content is empty, return empty string
        Ok(String::new())
    }

    /// Create regular prompt using chat template or simple format
    fn create_regular_prompt(messages: &[ChatMessage], loaded_model: &super::model_manager::LoadedModel) -> Result<String> {
        // Convert ChatRequest to llama.cpp format
        let llama_messages = Self::convert_chat_messages(messages)?;
        
        // Apply chat template to create prompt
        if let Some(template) = &loaded_model.chat_template {
            loaded_model.model
                .apply_chat_template(template, &llama_messages, true)
                .map_err(|e| Error::Internal(format!("Failed to apply chat template: {e}")))
        } else {
            // Fallback: create simple prompt format
            Ok(Self::create_simple_prompt(&llama_messages))
        }
    }

    /// Create a simple prompt format when no chat template is available
    fn create_simple_prompt(messages: &[LlamaChatMessage]) -> String {
        let mut prompt = String::new();
        
        for _message in messages {
            // This is a simplified approach - in practice we'd need to access
            // the role and content from LlamaChatMessage, which may require
            // additional methods in the llama-cpp-2 crate
            prompt.push_str(&format!("<message>\n"));
        }
        
        prompt
    }

    /// Create a sampler based on chat options and optional grammar for tool calling
    fn create_sampler(
        options: &GenerationOptions,
        model: &llama_cpp_2::model::LlamaModel,
        tools: Option<&[crate::chat::Tool]>,
        tool_config: Option<&super::tool_templates::ToolConfig>,
    ) -> Result<LlamaSampler> {
        let seed = options.seed.unwrap_or(1234);
        let mut samplers = Vec::new();
        
        // Add grammar sampler for tool calls if tools are provided
        if let (Some(tools), Some(config)) = (tools, tool_config) {
            if !tools.is_empty() && config.supports_tools {
                // Generate GBNF grammar from tools
                match tools_to_gbnf(tools) {
                    Ok(grammar) => {
                        // Use lazy grammar with trigger words
                        let trigger_tokens: Vec<llama_cpp_2::token::LlamaToken> = Vec::new(); // Could be populated with actual tokens
                        
                        if let Some(grammar_sampler) = LlamaSampler::grammar_lazy(
                            model,
                            &grammar,
                            "root",
                            &config.trigger_words,
                            &trigger_tokens,
                        ) {
                            samplers.push(grammar_sampler);
                        }
                    }
                    Err(_) => {
                        // If grammar generation fails, continue without grammar
                        // This allows the model to generate normally
                    }
                }
            }
        }
        
        // Add temperature sampling if specified
        if let Some(temperature) = options.temperature {
            samplers.push(LlamaSampler::temp(temperature as f32));
        }
        
        // Add top_p sampling if specified  
        if let Some(top_p) = options.top_p {
            samplers.push(LlamaSampler::top_p(top_p as f32, 1));
        }
        
        // Add random sampling with seed
        samplers.push(LlamaSampler::dist(seed as u32));
        
        // Use greedy as fallback
        if samplers.is_empty() {
            samplers.push(LlamaSampler::greedy());
        }
        
        Ok(LlamaSampler::chain(samplers, false))
    }
}