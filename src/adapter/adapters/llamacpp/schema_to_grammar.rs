//! JSON Schema to GBNF Grammar Conversion
//!
//! Converts JSON schema definitions from rust-genai Tool definitions to GBNF grammar
//! format for use with llama.cpp constrained generation.

use serde_json::Value;
use std::collections::HashMap;
use crate::{Error, Result};

/// Converts a JSON schema to GBNF grammar format
pub fn schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut converter = GrammarConverter::new();
    converter.convert_schema(schema)
}

/// Converts multiple tool schemas to a unified GBNF grammar for tool calling
pub fn tools_to_gbnf(tools: &[crate::chat::Tool]) -> Result<String> {
    if tools.is_empty() {
        return Err(Error::Internal("No tools provided for grammar generation".to_string()));
    }

    let mut converter = GrammarConverter::new();
    converter.convert_tools(tools)
}

struct GrammarConverter {
    rules: HashMap<String, String>,
    rule_counter: usize,
}

impl GrammarConverter {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
            rule_counter: 0,
        }
    }

    fn convert_tools(&mut self, tools: &[crate::chat::Tool]) -> Result<String> {
        // Create the root rule for tool calls
        let mut tool_choices = Vec::new();
        
        for tool in tools {
            let tool_name = &tool.name;
            let rule_name = format!("tool_{}", sanitize_name(tool_name));
            
            if let Some(schema) = &tool.schema {
                let params_rule = self.convert_value(schema, &format!("{}_params", rule_name))?;
                
                // Tool call structure: {"name": "tool_name", "arguments": {...}}
                let tool_rule = format!(
                    r#"{} ::= "{{"" ws "\"name\"" ws ":" ws "\"{}\"" ws "," ws "\"arguments\"" ws ":" ws {} ws "}}""#,
                    rule_name, tool_name, params_rule
                );
                
                self.rules.insert(rule_name.clone(), tool_rule);
                tool_choices.push(rule_name);
            } else {
                // Tool with no parameters
                let tool_rule = format!(
                    r#"{} ::= "{{"" ws "\"name\"" ws ":" ws "\"{}\"" ws "," ws "\"arguments\"" ws ":" ws "{{}}" ws "}}""#,
                    rule_name, tool_name
                );
                
                self.rules.insert(rule_name.clone(), tool_rule);
                tool_choices.push(rule_name);
            }
        }
        
        // Root rule that allows any of the tools
        let root_rule = format!("root ::= {}", tool_choices.join(" | "));
        self.rules.insert("root".to_string(), root_rule);
        
        // Add standard whitespace rule
        self.rules.insert("ws".to_string(), r#"ws ::= [ \t\n]*"#.to_string());
        
        Ok(self.build_grammar())
    }

    fn convert_schema(&mut self, schema: &Value) -> Result<String> {
        let root_rule = self.convert_value(schema, "root")?;
        
        // Add standard whitespace rule
        self.rules.insert("ws".to_string(), r#"ws ::= [ \t\n]*"#.to_string());
        
        Ok(self.build_grammar())
    }

    fn convert_value(&mut self, value: &Value, rule_name: &str) -> Result<String> {
        match value {
            Value::Object(obj) => self.convert_object(obj, rule_name),
            _ => Err(Error::Internal("Schema must be an object".to_string())),
        }
    }

    fn convert_object(&mut self, obj: &serde_json::Map<String, Value>, rule_name: &str) -> Result<String> {
        let schema_type = obj.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("object");

        match schema_type {
            "object" => self.convert_object_type(obj, rule_name),
            "array" => self.convert_array_type(obj, rule_name),
            "string" => Ok(self.convert_string_type(obj, rule_name)),
            "number" | "integer" => Ok(self.convert_number_type(rule_name)),
            "boolean" => Ok(self.convert_boolean_type(rule_name)),
            _ => Err(Error::Internal(format!("Unsupported schema type: {}", schema_type))),
        }
    }

    fn convert_object_type(&mut self, obj: &serde_json::Map<String, Value>, rule_name: &str) -> Result<String> {
        let mut property_rules = Vec::new();
        let mut required_props = Vec::new();
        let mut optional_props = Vec::new();

        // Get required properties
        if let Some(Value::Array(required)) = obj.get("required") {
            for req in required {
                if let Some(prop_name) = req.as_str() {
                    required_props.push(prop_name.to_string());
                }
            }
        }

        // Process properties
        if let Some(Value::Object(properties)) = obj.get("properties") {
            for (prop_name, prop_schema) in properties {
                let prop_rule_name = format!("{}_{}", rule_name, sanitize_name(prop_name));
                let prop_rule = self.convert_value(prop_schema, &prop_rule_name)?;
                
                let property_rule = format!(
                    r#"{}_prop ::= "\"{}\"" ws ":" ws {}"#,
                    sanitize_name(prop_name), prop_name, prop_rule
                );
                
                self.rules.insert(format!("{}_prop", sanitize_name(prop_name)), property_rule);
                
                if required_props.contains(prop_name) {
                    property_rules.push(format!("{}_prop", sanitize_name(prop_name)));
                } else {
                    optional_props.push(format!("{}_prop", sanitize_name(prop_name)));
                }
            }
        }

        // Build object rule
        let mut object_rule = r#""{" ws"#.to_string();
        
        if !property_rules.is_empty() {
            object_rule.push_str(&format!(" {} ", property_rules.join(" ws \",\" ws ")));
            
            // Add optional properties if any
            if !optional_props.is_empty() {
                object_rule.push_str(&format!(" (ws \",\" ws ({}))*", optional_props.join(" | ")));
            }
        }
        
        object_rule.push_str(r#" ws "}""#);
        
        self.rules.insert(rule_name.to_string(), format!("{} ::= {}", rule_name, object_rule));
        
        Ok(rule_name.to_string())
    }

    fn convert_array_type(&mut self, obj: &serde_json::Map<String, Value>, rule_name: &str) -> Result<String> {
        let items_rule = if let Some(items_schema) = obj.get("items") {
            let items_rule_name = format!("{}_item", rule_name);
            self.convert_value(items_schema, &items_rule_name)?
        } else {
            // Allow any JSON value if no items schema specified
            "json_value".to_string()
        };

        let array_rule = format!(
            r#"{} ::= "[" ws ({} (ws "," ws {})*)? ws "]""#,
            rule_name, items_rule, items_rule
        );

        self.rules.insert(rule_name.to_string(), array_rule);
        Ok(rule_name.to_string())
    }

    fn convert_string_type(&mut self, obj: &serde_json::Map<String, Value>, rule_name: &str) -> String {
        // Check for enum values
        if let Some(Value::Array(enum_values)) = obj.get("enum") {
            let mut choices = Vec::new();
            for enum_val in enum_values {
                if let Some(s) = enum_val.as_str() {
                    choices.push(format!(r#""\"{}\"=""#, s));
                }
            }
            
            if !choices.is_empty() {
                let enum_rule = format!("{} ::= {}", rule_name, choices.join(" | "));
                self.rules.insert(rule_name.to_string(), enum_rule);
                return rule_name.to_string();
            }
        }

        // Standard string rule
        let string_rule = format!(r#"{} ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"""#, rule_name);
        self.rules.insert(rule_name.to_string(), string_rule);
        rule_name.to_string()
    }

    fn convert_number_type(&mut self, rule_name: &str) -> String {
        let number_rule = format!(r#"{} ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?"#, rule_name);
        self.rules.insert(rule_name.to_string(), number_rule);
        rule_name.to_string()
    }

    fn convert_boolean_type(&mut self, rule_name: &str) -> String {
        let boolean_rule = format!(r#"{} ::= "true" | "false""#, rule_name);
        self.rules.insert(rule_name.to_string(), boolean_rule);
        rule_name.to_string()
    }

    fn build_grammar(&self) -> String {
        let mut grammar = String::new();
        
        // Add root rule first
        if let Some(root_rule) = self.rules.get("root") {
            grammar.push_str(root_rule);
            grammar.push('\n');
        }
        
        // Add all other rules
        for (name, rule) in &self.rules {
            if name != "root" {
                grammar.push_str(rule);
                grammar.push('\n');
            }
        }
        
        grammar
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::chat::Tool;

    #[test]
    fn test_simple_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        });

        let result = schema_to_gbnf(&schema).unwrap();
        assert!(result.contains("root ::="));
        assert!(result.contains("name"));
        assert!(result.contains("age"));
    }

    #[test]
    fn test_enum_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"]
                }
            }
        });

        let result = schema_to_gbnf(&schema).unwrap();
        assert!(result.contains("active"));
        assert!(result.contains("inactive"));
    }

    #[test]
    fn test_tools_to_gbnf() {
        let tool = Tool::new("get_weather")
            .with_schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }));

        let tools = vec![tool];
        let result = tools_to_gbnf(&tools).unwrap();
        
        assert!(result.contains("get_weather"));
        assert!(result.contains("city"));
        assert!(result.contains("name"));
        assert!(result.contains("arguments"));
    }
}