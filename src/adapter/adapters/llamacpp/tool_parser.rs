//! Tool call parsing from model outputs
//!
//! Parses tool calls from generated text based on different model formats
//! and converts them to rust-genai ToolCall structures.

use crate::chat::{ToolCall, MessageContent};
use crate::{Error, Result};
use serde_json::Value;
use regex::Regex;
use super::tool_templates::{ToolFormat, ToolConfig};

/// Parse tool calls from generated text based on model configuration
pub fn parse_tool_calls(text: &str, config: &ToolConfig) -> Result<Vec<ToolCall>> {
    match &config.format {
        ToolFormat::Llama3 { trigger_tag } => parse_llama3_tools(text, trigger_tag),
        ToolFormat::Functionary { function_tag } => parse_functionary_tools(text, function_tag),
        ToolFormat::Hermes => parse_hermes_tools(text),
        ToolFormat::Generic => parse_generic_tools(text),
    }
}

/// Check if text contains tool call indicators based on model format
pub fn contains_tool_calls(text: &str, config: &ToolConfig) -> bool {
    config.trigger_words.iter().any(|trigger| text.contains(trigger))
}

/// Parse tool calls in Llama 3.x format
fn parse_llama3_tools(text: &str, trigger_tag: &str) -> Result<Vec<ToolCall>> {
    let mut tool_calls = Vec::new();
    
    // Look for trigger tag followed by JSON
    if let Some(start_pos) = text.find(trigger_tag) {
        let after_trigger = &text[start_pos + trigger_tag.len()..];
        
        // Find JSON content after the trigger
        if let Some(json_start) = after_trigger.find('{') {
            let json_part = &after_trigger[json_start..];
            
            // Try to find the end of the JSON object
            if let Some(json_end) = find_json_end(json_part) {
                let json_str = &json_part[..json_end + 1];
                
                match serde_json::from_str::<Value>(json_str) {
                    Ok(json_val) => {
                        if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                            tool_calls.push(tool_call);
                        }
                    }
                    Err(_) => {
                        // Try to fix common JSON issues and parse again
                        if let Ok(fixed_json) = fix_json_issues(json_str) {
                            if let Ok(json_val) = serde_json::from_str::<Value>(&fixed_json) {
                                if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                                    tool_calls.push(tool_call);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(tool_calls)
}

/// Parse tool calls in Functionary format
fn parse_functionary_tools(text: &str, function_tag: &str) -> Result<Vec<ToolCall>> {
    let mut tool_calls = Vec::new();
    
    // Regex to match <function=function_name{...}>
    let pattern = format!(r"{}(\w+)(\{{.*?\}})", regex::escape(function_tag));
    let re = Regex::new(&pattern).map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
    
    for cap in re.captures_iter(text) {
        if let (Some(fn_name), Some(args)) = (cap.get(1), cap.get(2)) {
            let function_name = fn_name.as_str().to_string();
            let args_str = args.as_str();
            
            match serde_json::from_str::<Value>(args_str) {
                Ok(arguments) => {
                    let tool_call = ToolCall {
                        call_id: generate_call_id(),
                        fn_name: function_name,
                        fn_arguments: arguments,
                    };
                    tool_calls.push(tool_call);
                }
                Err(_) => {
                    // Try to fix JSON and parse again
                    if let Ok(fixed_json) = fix_json_issues(args_str) {
                        if let Ok(arguments) = serde_json::from_str::<Value>(&fixed_json) {
                            let tool_call = ToolCall {
                                call_id: generate_call_id(),
                                fn_name: function_name,
                                fn_arguments: arguments,
                            };
                            tool_calls.push(tool_call);
                        }
                    }
                }
            }
        }
    }
    
    Ok(tool_calls)
}

/// Parse tool calls in Hermes format
fn parse_hermes_tools(text: &str) -> Result<Vec<ToolCall>> {
    let mut tool_calls = Vec::new();
    
    // Look for <tool_call>...</tool_call> tags
    let re = Regex::new(r"<tool_call>\s*(.*?)\s*</tool_call>")
        .map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
    
    for cap in re.captures_iter(text) {
        if let Some(json_match) = cap.get(1) {
            let json_str = json_match.as_str();
            
            match serde_json::from_str::<Value>(json_str) {
                Ok(json_val) => {
                    if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                        tool_calls.push(tool_call);
                    }
                }
                Err(_) => {
                    // Try to fix JSON issues
                    if let Ok(fixed_json) = fix_json_issues(json_str) {
                        if let Ok(json_val) = serde_json::from_str::<Value>(&fixed_json) {
                            if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                                tool_calls.push(tool_call);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(tool_calls)
}

/// Parse tool calls in generic JSON format
fn parse_generic_tools(text: &str) -> Result<Vec<ToolCall>> {
    let mut tool_calls = Vec::new();
    
    // Look for ```json ... ``` blocks first
    let re = Regex::new(r"```json\s*(.*?)\s*```")
        .map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
    
    let mut found_code_blocks = false;
    for cap in re.captures_iter(text) {
        found_code_blocks = true;
        if let Some(json_match) = cap.get(1) {
            let json_str = json_match.as_str();
            
            match serde_json::from_str::<Value>(json_str) {
                Ok(json_val) => {
                    if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                        tool_calls.push(tool_call);
                    }
                }
                Err(_) => {
                    // Try to fix JSON issues
                    if let Ok(fixed_json) = fix_json_issues(json_str) {
                        if let Ok(json_val) = serde_json::from_str::<Value>(&fixed_json) {
                            if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                                tool_calls.push(tool_call);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Only look for standalone JSON objects if no code blocks were found
    if !found_code_blocks {
        let json_re = Regex::new(r#"\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}"#)
            .map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
        
        for cap in json_re.captures_iter(text) {
            let json_str = cap.get(0).unwrap().as_str();
            
            match serde_json::from_str::<Value>(json_str) {
                Ok(json_val) => {
                    if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                        tool_calls.push(tool_call);
                    }
                }
                Err(_) => {
                    // Try to fix JSON issues
                    if let Ok(fixed_json) = fix_json_issues(json_str) {
                        if let Ok(json_val) = serde_json::from_str::<Value>(&fixed_json) {
                            if let Some(tool_call) = parse_tool_call_json(&json_val)? {
                                tool_calls.push(tool_call);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(tool_calls)
}

/// Parse a ToolCall from JSON value
fn parse_tool_call_json(json_val: &Value) -> Result<Option<ToolCall>> {
    if let Value::Object(obj) = json_val {
        if let (Some(name_val), Some(args_val)) = (obj.get("name"), obj.get("arguments")) {
            if let Some(name) = name_val.as_str() {
                let tool_call = ToolCall {
                    call_id: generate_call_id(),
                    fn_name: name.to_string(),
                    fn_arguments: args_val.clone(),
                };
                return Ok(Some(tool_call));
            }
        }
    }
    Ok(None)
}

/// Find the end of a JSON object by counting braces
fn find_json_end(json_str: &str) -> Option<usize> {
    let mut brace_count = 0;
    let mut in_string = false;
    let mut escaped = false;
    
    for (i, ch) in json_str.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => brace_count += 1,
            '}' if !in_string => {
                brace_count -= 1;
                if brace_count == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    
    None
}

/// Try to fix common JSON issues in generated text
fn fix_json_issues(json_str: &str) -> Result<String> {
    let mut fixed = json_str.trim().to_string();
    
    // Remove trailing commas
    fixed = fixed.replace(",}", "}").replace(",]", "]");
    
    // Fix unquoted keys (basic attempt)
    let key_re = Regex::new(r"(\w+):")
        .map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
    fixed = key_re.replace_all(&fixed, "\"$1\":").to_string();
    
    // Fix unquoted values (look for unquoted strings after colons or in arrays)
    // Match pattern like: "key": unquoted_value
    let value_re = Regex::new(r#":\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])"#)
        .map_err(|e| Error::AdapterError(format!("Regex error: {}", e)))?;
    fixed = value_re.replace_all(&fixed, ": \"$1\"$2").to_string();
    
    // Fix single quotes to double quotes
    fixed = fixed.replace("'", "\"");
    
    Ok(fixed)
}

/// Generate a unique call ID for tool calls
fn generate_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("call_{}", timestamp)
}

/// Create MessageContent from tool calls and remaining text
pub fn create_response_content(text: &str, tool_calls: Vec<ToolCall>) -> MessageContent {
    if tool_calls.is_empty() {
        MessageContent::Text(text.to_string())
    } else {
        // If we have tool calls, we might want to return them separately
        // For now, just return the tool calls
        MessageContent::ToolCalls(tool_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tool_templates::ToolFormat;

    #[test]
    fn test_parse_llama3_tool_call() {
        let config = ToolConfig {
            trigger_words: vec!["<|python_tag|>".to_string()],
            supports_tools: true,
            format: ToolFormat::Llama3 {
                trigger_tag: "<|python_tag|>".to_string(),
            },
        };
        
        let text = r#"I'll help you with the weather. <|python_tag|>
{"name": "get_weather", "arguments": {"city": "San Francisco"}}
Let me check the weather for you."#;
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
        
        if let Value::Object(args) = &tool_calls[0].fn_arguments {
            assert_eq!(args.get("city").unwrap().as_str().unwrap(), "San Francisco");
        } else {
            panic!("Expected object arguments");
        }
    }

    #[test]
    fn test_parse_functionary_tool_call() {
        let config = ToolConfig {
            trigger_words: vec!["<function=".to_string()],
            supports_tools: true,
            format: ToolFormat::Functionary {
                function_tag: "<function=".to_string(),
            },
        };
        
        let text = r#"I'll get the weather for you. <function=get_weather{"city": "New York"}>
The weather information will be retrieved."#;
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
    }

    #[test]
    fn test_parse_hermes_tool_call() {
        let config = ToolConfig {
            trigger_words: vec!["<tool_call>".to_string()],
            supports_tools: true,
            format: ToolFormat::Hermes,
        };
        
        let text = r#"I'll check the weather for you.

<tool_call>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool_call>

Let me get that information."#;
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
    }

    #[test]
    fn test_parse_generic_tool_call() {
        let config = ToolConfig {
            trigger_words: vec!["```json".to_string()],
            supports_tools: true,
            format: ToolFormat::Generic,
        };
        
        let text = r#"I'll check the weather using this tool:

```json
{"name": "get_weather", "arguments": {"city": "Paris"}}
```

This will get the current weather."#;
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
    }

    #[test]
    fn test_fix_json_issues() {
        let broken_json = r#"{"name": get_weather, "arguments": {"city": 'Paris',}}"#;
        let fixed = fix_json_issues(broken_json).unwrap();
        
        let parsed: Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(parsed["name"], "get_weather");
    }

    #[test]
    fn test_find_json_end() {
        let json_str = r#"{"name": "test", "nested": {"value": "data"}} extra text"#;
        let end_pos = find_json_end(json_str).unwrap();
        assert_eq!(&json_str[..end_pos + 1], r#"{"name": "test", "nested": {"value": "data"}}"#);
    }

    #[test]
    fn test_contains_tool_calls() {
        let config = ToolConfig {
            trigger_words: vec!["<|python_tag|>".to_string()],
            supports_tools: true,
            format: ToolFormat::Llama3 {
                trigger_tag: "<|python_tag|>".to_string(),
            },
        };
        
        assert!(contains_tool_calls("Some text <|python_tag|> more text", &config));
        assert!(!contains_tool_calls("Just regular text", &config));
    }
}