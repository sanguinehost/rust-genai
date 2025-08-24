#[cfg(feature = "llamacpp")]
mod llamacpp_tools_tests {
    use genai::chat::{ChatMessage, ChatRole, MessageContent, Tool};
    use genai::adapter::adapters::llamacpp::{
        schema_to_grammar::tools_to_gbnf,
        tool_templates::{detect_tool_config, apply_tool_template, ToolFormat},
        tool_parser::{parse_tool_calls, contains_tool_calls},
    };
    use serde_json::json;

    #[test]
    fn test_schema_to_gbnf_conversion() {
        let tool = Tool::new("get_weather")
            .with_description("Get weather information")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }));

        let result = tools_to_gbnf(&[tool]);
        assert!(result.is_ok());
        
        let grammar = result.unwrap();
        assert!(grammar.contains("root ::="));
        assert!(grammar.contains("get_weather"));
        assert!(grammar.contains("city"));
        assert!(grammar.contains("unit"));
        assert!(grammar.contains("name"));
        assert!(grammar.contains("arguments"));
    }

    #[test]
    fn test_tool_config_detection() {
        // Test Llama 3.x detection
        let llama_config = detect_tool_config("llama-3.2-instruct.gguf");
        assert!(llama_config.supports_tools);
        assert!(llama_config.trigger_words.contains(&"<|python_tag|>".to_string()));
        match llama_config.format {
            ToolFormat::Llama3 { trigger_tag } => {
                assert_eq!(trigger_tag, "<|python_tag|>");
            }
            _ => panic!("Expected Llama3 format"),
        }

        // Test Functionary detection
        let func_config = detect_tool_config("functionary-v3-small.gguf");
        assert!(func_config.supports_tools);
        assert!(func_config.trigger_words.contains(&"<function=".to_string()));

        // Test Hermes detection
        let hermes_config = detect_tool_config("hermes-2-pro.gguf");
        assert!(hermes_config.supports_tools);
        assert!(hermes_config.trigger_words.contains(&"<tool_call>".to_string()));

        // Test generic fallback
        let generic_config = detect_tool_config("random-model.gguf");
        assert!(generic_config.supports_tools);
        assert!(matches!(generic_config.format, ToolFormat::Generic));
    }

    #[test]
    fn test_llama3_template_application() {
        let tool = Tool::new("get_weather")
            .with_description("Get current weather")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }));

        let message = ChatMessage {
            role: ChatRole::User,
            content: MessageContent::Text("What's the weather in New York?".to_string()),
            options: None,
        };

        let config = detect_tool_config("llama-3.2.gguf");
        let result = apply_tool_template(&[message], &[tool], &config);
        
        assert!(result.is_ok());
        let prompt = result.unwrap();
        
        assert!(prompt.contains("<|python_tag|>"));
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("What's the weather in New York?"));
        assert!(prompt.contains("<|start_header_id|>"));
        assert!(prompt.contains("assistant<|end_header_id|>"));
    }

    #[test]
    fn test_tool_call_parsing_llama3() {
        let config = detect_tool_config("llama-3.2.gguf");
        
        let text = r#"I'll check the weather for you. <|python_tag|>
{"name": "get_weather", "arguments": {"city": "San Francisco", "unit": "C"}}

The weather information shows..."#;

        assert!(contains_tool_calls(text, &config));
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.fn_name, "get_weather");
        
        if let serde_json::Value::Object(args) = &tool_call.fn_arguments {
            assert_eq!(args.get("city").unwrap().as_str().unwrap(), "San Francisco");
            assert_eq!(args.get("unit").unwrap().as_str().unwrap(), "C");
        } else {
            panic!("Expected object arguments");
        }
    }

    #[test]
    fn test_tool_call_parsing_functionary() {
        let config = detect_tool_config("functionary-v3.gguf");
        
        let text = r#"I'll help you with that. <function=get_weather{"city": "London", "unit": "F"}>
The function will return the weather data."#;

        assert!(contains_tool_calls(text, &config));
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.fn_name, "get_weather");
        
        if let serde_json::Value::Object(args) = &tool_call.fn_arguments {
            assert_eq!(args.get("city").unwrap().as_str().unwrap(), "London");
            assert_eq!(args.get("unit").unwrap().as_str().unwrap(), "F");
        }
    }

    #[test]
    fn test_tool_call_parsing_hermes() {
        let config = detect_tool_config("hermes-2-pro.gguf");
        
        let text = r#"I can help you with the weather. Let me check that for you.

<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "C"}}
</tool_call>

This will give us the current weather conditions."#;

        assert!(contains_tool_calls(text, &config));
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.fn_name, "get_weather");
    }

    #[test]
    fn test_tool_call_parsing_generic() {
        let config = detect_tool_config("generic-model.gguf");
        
        let text = r#"I'll get the weather information for you:

```json
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}
```

This should return the current weather."#;

        assert!(contains_tool_calls(text, &config));
        
        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
    }

    #[test]
    fn test_multiple_tool_schema() {
        let weather_tool = Tool::new("get_weather")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }));

        let time_tool = Tool::new("get_time")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                }
            }));

        let result = tools_to_gbnf(&[weather_tool, time_tool]);
        assert!(result.is_ok());
        
        let grammar = result.unwrap();
        assert!(grammar.contains("get_weather"));
        assert!(grammar.contains("get_time"));
        assert!(grammar.contains("city"));
        assert!(grammar.contains("timezone"));
    }

    #[test]
    fn test_enum_schema_handling() {
        let tool = Tool::new("set_mode")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["light", "dark", "auto"]
                    }
                }
            }));

        let result = tools_to_gbnf(&[tool]);
        assert!(result.is_ok());
        
        let grammar = result.unwrap();
        assert!(grammar.contains("light"));
        assert!(grammar.contains("dark"));
        assert!(grammar.contains("auto"));
    }

    #[test]
    fn test_array_schema_handling() {
        let tool = Tool::new("process_list")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }));

        let result = tools_to_gbnf(&[tool]);
        assert!(result.is_ok());
        
        // Should not panic and should contain array syntax
        let grammar = result.unwrap();
        assert!(grammar.contains("["));
        assert!(grammar.contains("]"));
    }

    #[test]
    fn test_no_tools_provided() {
        let result = tools_to_gbnf(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No tools provided"));
    }

    #[test]
    fn test_broken_json_fixing() {
        let config = detect_tool_config("llama-3.2.gguf");
        
        // Test with broken JSON that should be fixable
        let text = r#"<|python_tag|>
{"name": get_weather, "arguments": {"city": 'San Francisco',}}
"#;

        let tool_calls = parse_tool_calls(text, &config).unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].fn_name, "get_weather");
    }
}