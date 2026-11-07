//! Chat prompt formatting using the model's own Jinja chat template
//! (`tokenizer.chat_template` in GGUF), rendered with `minijinja`.

use anyhow::Result;
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};

/// One chat message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// Message text.
    pub content: String,
    /// Tool calls made by an assistant message (OpenAI shape:
    /// `[{"id","type":"function","function":{"name","arguments"}}]`;
    /// `arguments` may be an object or a JSON string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// For `tool` messages: the id of the call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional author / tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Convenience constructor.
    ///
    /// # Example
    /// ```
    /// let m = aido::llm::chat::Message::new("user", "hi");
    /// assert_eq!(m.role, "user");
    /// ```
    pub fn new(role: &str, content: &str) -> Self {
        Message { role: role.into(), content: content.into(), ..Default::default() }
    }
}

/// Renders a conversation into the prompt string the model was trained on.
///
/// * **Input** — `template`: Jinja template (if `None`, a ChatML fallback is
///   used); `messages`; `bos`/`eos` token strings; `add_generation_prompt`;
///   `enable_thinking` (Qwen3 switch; ignored by other templates).
/// * **Output** — the formatted prompt.
///
/// # Example
/// ```
/// use aido::llm::chat::{render, Message};
/// let p = render(None, &[Message::new("user", "Hi")], "", "", true, false).unwrap();
/// assert!(p.contains("<|im_start|>user\nHi<|im_end|>"));
/// assert!(p.ends_with("<|im_start|>assistant\n"));
/// ```
pub fn render(template: Option<&str>, messages: &[Message], bos: &str, eos: &str, add_generation_prompt: bool, enable_thinking: bool) -> Result<String> {
    render_with_tools(template, messages, bos, eos, add_generation_prompt, enable_thinking, None)
}

/// [`render`] with tool definitions (OpenAI `tools` array) made available to
/// the template as `tools`. Templates without tool support get the tool
/// list appended to the first system message as JSON instructions instead
/// (ChatML fallback: Hermes-style `<tool_call>` protocol).
///
/// # Example
/// ```
/// use aido::llm::chat::{render_with_tools, Message};
/// let tools = serde_json::json!([{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}]);
/// let p = render_with_tools(None, &[Message::new("user", "time?")], "", "", true, false, Some(&tools)).unwrap();
/// assert!(p.contains("get_time") && p.contains("<tool_call>"));
/// ```
pub fn render_with_tools(template: Option<&str>, messages: &[Message], bos: &str, eos: &str, add_generation_prompt: bool, enable_thinking: bool, tools: Option<&serde_json::Value>) -> Result<String> {
    let supports_tools = template.map(|t| t.contains("tools")).unwrap_or(false);
    let mut msgs: Vec<Message> = messages.to_vec();
    if let (Some(tl), false) = (tools, supports_tools) {
        let note = format!(
            "# Tools\nYou may call one or more functions to assist with the user query. Function signatures:\n<tools>\n{}\n</tools>\nFor each call return a JSON object inside <tool_call></tool_call> tags: <tool_call>{{\"name\": <function-name>, \"arguments\": <args-json-object>}}</tool_call>",
            tl.as_array().map(|a| a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n")).unwrap_or_default()
        );
        match msgs.iter_mut().find(|m| m.role == "system") {
            Some(m) => m.content = format!("{}\n\n{}", m.content, note),
            None => msgs.insert(0, Message::new("system", &note)),
        }
    }
    // Templates expect tool-call arguments as objects.
    for m in msgs.iter_mut() {
        if let Some(serde_json::Value::Array(calls)) = m.tool_calls.as_mut() {
            for c in calls.iter_mut() {
                if let Some(a) = c.pointer_mut("/function/arguments") {
                    if let serde_json::Value::String(s) = a {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                            *a = v;
                        }
                    }
                }
            }
        }
        // ChatML fallback has no tool role: show results as user-visible tool output.
        if !supports_tools && m.role == "tool" {
            m.role = "user".into();
            m.content = format!("<tool_response>\n{}\n</tool_response>", m.content);
        }
        if !supports_tools && m.role == "assistant" {
            if let Some(serde_json::Value::Array(calls)) = &m.tool_calls {
                for c in calls {
                    m.content.push_str(&format!("\n<tool_call>\n{}\n</tool_call>", serde_json::json!({"name": c["function"]["name"], "arguments": c["function"]["arguments"]})));
                }
            }
        }
    }
    let tools_v: Option<serde_json::Value> = if supports_tools { tools.cloned() } else { None };
    let messages = &msgs[..];
    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let tpl = template.unwrap_or(CHATML);
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    minijinja_contrib::add_to_environment(&mut env);
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_function("strftime_now", |_fmt: String| -> String { "01 Jan 2026".to_string() });
    env.add_template("chat", tpl)?;
    let t = env.get_template("chat")?;
    let s = t.render(context! {
        messages => messages,
        bos_token => bos,
        eos_token => eos,
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
        tools => tools_v,
        date_string => "01 Jan 2026",
    })?;
    Ok(s)
}
