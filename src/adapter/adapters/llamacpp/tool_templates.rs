//! Model-specific tool calling templates and trigger detection
//!
//! Different models use different formats for tool calling. This module handles
//! the conversion between rust-genai's unified tool format and model-specific templates.

use crate::chat::{Tool, ChatMessage, ChatRole};
use crate::Result;

/// Tool calling configuration for different model families
#[derive(Debug, Clone)]
pub struct ToolConfig {
    /// Trigger words that activate grammar constraints
    pub trigger_words: Vec<String>,
    /// Whether this model supports native tool calling
    pub supports_tools: bool,
    /// Template format for tool calls
    pub format: ToolFormat,
}

#[derive(Debug, Clone)]
pub enum ToolFormat {
    /// Llama 3.x style with <|python_tag|> trigger
    Llama3 {
        trigger_tag: String,
    },
    /// Functionary v3 style
    Functionary {
        function_tag: String,
    },
    /// Hermes 2 Pro style
    Hermes,
    /// Generic JSON style
    Generic,
}

/// Detect tool calling configuration for a model based on its name
pub fn detect_tool_config(model_name: &str) -> ToolConfig {
    let model_lower = model_name.to_lowercase();
    
    if model_lower.contains("llama") && (model_lower.contains("3.") || model_lower.contains("3_")) {
        ToolConfig {
            trigger_words: vec!["<|python_tag|>".to_string()],
            supports_tools: true,
            format: ToolFormat::Llama3 {
                trigger_tag: "<|python_tag|>".to_string(),
            },
        }
    } else if model_lower.contains("functionary") {
        ToolConfig {
            trigger_words: vec!["<function=".to_string()],
            supports_tools: true,
            format: ToolFormat::Functionary {
                function_tag: "<function=".to_string(),
            },
        }
    } else if model_lower.contains("hermes") {
        ToolConfig {
            trigger_words: vec!["<tool_call>".to_string()],
            supports_tools: true,
            format: ToolFormat::Hermes,
        }
    } else {
        // Generic support for other models
        ToolConfig {
            trigger_words: vec!["```json".to_string(), "{\"name\":".to_string()],
            supports_tools: true,
            format: ToolFormat::Generic,
        }
    }
}

/// Apply tool calling template to messages for a specific model
pub fn apply_tool_template(
    messages: &[ChatMessage],
    tools: &[Tool],
    config: &ToolConfig,
) -> Result<String> {
    match &config.format {
        ToolFormat::Llama3 { trigger_tag } => apply_llama3_template(messages, tools, trigger_tag),
        ToolFormat::Functionary { function_tag } => apply_functionary_template(messages, tools, function_tag),
        ToolFormat::Hermes => apply_hermes_template(messages, tools),
        ToolFormat::Generic => apply_generic_template(messages, tools),
    }
}

fn apply_llama3_template(messages: &[ChatMessage], tools: &[Tool], trigger_tag: &str) -> Result<String> {
    let mut prompt = String::new();
    
    // Add system message with tool descriptions
    if !tools.is_empty() {
        prompt.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
        prompt.push_str("You are a helpful assistant with access to the following functions. ");
        prompt.push_str("Use them if required:\n\n");
        
        for tool in tools {
            prompt.push_str(&format!("Function: {}\n", tool.name));
            if let Some(desc) = &tool.description {
                prompt.push_str(&format!("Description: {}\n", desc));
            }
            if let Some(schema) = &tool.schema {
                prompt.push_str(&format!("Parameters: {}\n", schema));
            }
            prompt.push('\n');
        }
        
        prompt.push_str(&format!(
            "To call a function, use this format:\n{}\n{{\"name\": \"function_name\", \"arguments\": {{...}}}}\n\n<|eot_id|>\n\n",
            trigger_tag
        ));
    }
    
    // Add conversation messages
    for message in messages {
        match message.role {
            ChatRole::System => {
                if tools.is_empty() {  // Only add if we didn't add tools system message
                    prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    prompt.push_str(&extract_text_content(&message.content)?);
                    prompt.push_str("<|eot_id|>\n\n");
                }
            }
            ChatRole::User => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|eot_id|>\n\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|eot_id|>\n\n");
            }
            ChatRole::Tool => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str("Tool result: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|eot_id|>\n\n");
            }
        }
    }
    
    // Add assistant start
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    
    Ok(prompt)
}

fn apply_functionary_template(messages: &[ChatMessage], tools: &[Tool], function_tag: &str) -> Result<String> {
    let mut prompt = String::new();
    
    // Functionary uses a simpler format
    prompt.push_str("You are a helpful assistant that can call functions when needed.\n\n");
    
    if !tools.is_empty() {
        prompt.push_str("Available functions:\n");
        for tool in tools {
            prompt.push_str(&format!("- {}", tool.name));
            if let Some(desc) = &tool.description {
                prompt.push_str(&format!(": {}", desc));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
        
        prompt.push_str(&format!(
            "To call a function, use: {}function_name{{\"arg1\": \"value1\"}}\n\n",
            function_tag
        ));
    }
    
    // Add conversation
    for message in messages {
        match message.role {
            ChatRole::User => {
                prompt.push_str("User: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::Assistant => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::Tool => {
                prompt.push_str("Function result: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::System => {
                // Skip system messages as we added our own
            }
        }
    }
    
    prompt.push_str("Assistant: ");
    
    Ok(prompt)
}

fn apply_hermes_template(messages: &[ChatMessage], tools: &[Tool]) -> Result<String> {
    let mut prompt = String::new();
    
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("You are a helpful assistant that can use tools when needed.\n");
    
    if !tools.is_empty() {
        prompt.push_str("\nAvailable tools:\n");
        for tool in tools {
            prompt.push_str(&format!("- {}", tool.name));
            if let Some(desc) = &tool.description {
                prompt.push_str(&format!(": {}", desc));
            }
            prompt.push('\n');
        }
        
        prompt.push_str("\nTo use a tool, wrap your function call in <tool_call> tags with JSON:\n");
        prompt.push_str("<tool_call>\n{\"name\": \"function_name\", \"arguments\": {...}}\n</tool_call>\n");
    }
    
    prompt.push_str("<|im_end|>\n");
    
    for message in messages {
        match message.role {
            ChatRole::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                prompt.push_str("<|im_start|>tool\n");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::System => {
                // System message was already added
            }
        }
    }
    
    prompt.push_str("<|im_start|>assistant\n");
    
    Ok(prompt)
}

fn apply_generic_template(messages: &[ChatMessage], tools: &[Tool]) -> Result<String> {
    let mut prompt = String::new();
    
    if !tools.is_empty() {
        prompt.push_str("You have access to the following tools:\n");
        for tool in tools {
            prompt.push_str(&format!("- {}", tool.name));
            if let Some(desc) = &tool.description {
                prompt.push_str(&format!(": {}", desc));
            }
            prompt.push('\n');
        }
        
        prompt.push_str("\nTo use a tool, respond with JSON in this format:\n");
        prompt.push_str("```json\n{\"name\": \"tool_name\", \"arguments\": {...}}\n```\n\n");
    }
    
    // Simple conversation format
    for message in messages {
        match message.role {
            ChatRole::System => {
                prompt.push_str("System: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::User => {
                prompt.push_str("Human: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::Assistant => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
            ChatRole::Tool => {
                prompt.push_str("Tool: ");
                prompt.push_str(&extract_text_content(&message.content)?);
                prompt.push('\n');
            }
        }
    }
    
    prompt.push_str("Assistant: ");
    
    Ok(prompt)
}

/// Extract text content from MessageContent (helper function)
fn extract_text_content(content: &crate::chat::MessageContent) -> Result<String> {
    match content {
        crate::chat::MessageContent::Text(text) => Ok(text.clone()),
        crate::chat::MessageContent::Parts(parts) => {
            let mut text_content = String::new();
            for part in parts {
                match part {
                    crate::chat::ContentPart::Text(text) => {
                        text_content.push_str(&text);
                    }
                    _ => {
                        // Skip non-text parts for now
                        // TODO: Handle image content for multimodal models
                    }
                }
            }
            Ok(text_content)
        }
        crate::chat::MessageContent::ToolCalls(_) => {
            Ok("".to_string()) // Tool calls are handled separately
        }
        crate::chat::MessageContent::ToolResponses(_) => {
            Ok("".to_string()) // Tool responses are handled separately
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_detect_llama3_config() {
        let config = detect_tool_config("llama-3.2.gguf");
        assert!(config.supports_tools);
        assert!(config.trigger_words.contains(&"<|python_tag|>".to_string()));
        
        match config.format {
            ToolFormat::Llama3 { trigger_tag } => {
                assert_eq!(trigger_tag, "<|python_tag|>");
            }
            _ => panic!("Expected Llama3 format"),
        }
    }

    #[test]
    fn test_detect_functionary_config() {
        let config = detect_tool_config("functionary-v3.gguf");
        assert!(config.supports_tools);
        assert!(config.trigger_words.contains(&"<function=".to_string()));
    }

    #[test]
    fn test_detect_generic_config() {
        let config = detect_tool_config("unknown-model.gguf");
        assert!(config.supports_tools);
        assert!(matches!(config.format, ToolFormat::Generic));
    }

    #[test]
    fn test_apply_llama3_template() {
        let tool = Tool::new("get_weather")
            .with_description("Get weather information")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }));

        let message = crate::chat::ChatMessage {
            role: ChatRole::User,
            content: crate::chat::MessageContent::Text("What's the weather?".to_string()),
            options: None,
        };

        let config = ToolConfig {
            trigger_words: vec!["<|python_tag|>".to_string()],
            supports_tools: true,
            format: ToolFormat::Llama3 {
                trigger_tag: "<|python_tag|>".to_string(),
            },
        };

        let result = apply_tool_template(&[message], &[tool], &config).unwrap();
        
        assert!(result.contains("<|python_tag|>"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("What's the weather?"));
        assert!(result.contains("<|start_header_id|>"));
    }
}