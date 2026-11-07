//! Chat semantics on top of raw generation: tool calls, separated
//! reasoning ("thinking"), streaming splitter, and automatic summarisation
//! of history that no longer fits the context.
//!
//! Models emit tool calls and reasoning as plain text in family-specific
//! formats; this module turns them into structured data:
//!
//! | Family | Tool call | Reasoning |
//! |---|---|---|
//! | Qwen 2.5 / 3, Hermes, many fine-tunes | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` | `<think>…</think>` |
//! | Llama 3.1+ | `<\|python_tag\|>` or a bare `{"name": …, "parameters": {…}}` | — |
//! | DeepSeek-R1 distills | — | `…</think>` (opening tag in the template) |

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A structured tool call extracted from the model output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call id (`call_<n>`).
    pub id: String,
    /// Function name.
    pub name: String,
    /// Arguments as a JSON value (object).
    pub arguments: Value,
}

impl ToolCall {
    /// OpenAI wire format `{"id","type":"function","function":{"name","arguments": "<json string>"}}`.
    ///
    /// # Example
    /// ```
    /// use aido::llm::conversation::ToolCall;
    /// let t = ToolCall { id: "call_0".into(), name: "f".into(), arguments: serde_json::json!({"x": 1}) };
    /// assert_eq!(t.openai()["function"]["arguments"], "{\"x\":1}");
    /// ```
    pub fn openai(&self) -> Value {
        json!({"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments.to_string()}})
    }
}

/// Assistant output split into its parts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed {
    /// Visible answer text.
    pub content: String,
    /// Reasoning (text inside `<think>…</think>`), if any.
    pub reasoning: Option<String>,
    /// Tool calls, if any.
    pub tool_calls: Vec<ToolCall>,
}

fn call_from(v: &Value, i: usize) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let mut args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or(json!({}));
    if let Value::String(s) = &args {
        args = serde_json::from_str(s).unwrap_or(Value::String(s.clone()));
    }
    Some(ToolCall { id: format!("call_{i}"), name, arguments: args })
}

/// Parses raw assistant text into content, reasoning and tool calls.
///
/// # Example
/// ```
/// use aido::llm::conversation::parse_assistant;
/// let p = parse_assistant("<think>need weather</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Kazan\"}}\n</tool_call>");
/// assert_eq!(p.reasoning.as_deref(), Some("need weather"));
/// assert_eq!(p.tool_calls[0].name, "get_weather");
/// assert_eq!(p.tool_calls[0].arguments["city"], "Kazan");
/// assert_eq!(p.content, "");
///
/// let llama = parse_assistant("{\"name\": \"add\", \"parameters\": {\"a\": 1, \"b\": 2}}");
/// assert_eq!(llama.tool_calls[0].arguments["b"], 2);
///
/// assert_eq!(parse_assistant("Hello").content, "Hello");
/// ```
pub fn parse_assistant(raw: &str) -> Parsed {
    let mut text = raw.to_string();
    let mut reasoning = None;
    if let Some(end) = text.find("</think>") {
        let start = text.find("<think>").map(|s| s + 7).filter(|s| *s <= end).unwrap_or(0);
        let r = text[start..end].trim().to_string();
        reasoning = if r.is_empty() { None } else { Some(r) };
        text = format!("{}{}", &text[..text.find("<think>").filter(|s| *s <= end).unwrap_or(0)], &text[end + 8..]);
    }
    let mut calls = Vec::new();
    // <tool_call> blocks
    while let Some(s) = text.find("<tool_call>") {
        let e = text[s..].find("</tool_call>").map(|e| s + e).unwrap_or(text.len());
        let body = text[s + 11..e].trim();
        if let Ok(v) = serde_json::from_str::<Value>(body) {
            if let Some(c) = call_from(&v, calls.len()) {
                calls.push(c);
            }
        }
        let end = (e + 12).min(text.len());
        text.replace_range(s..end, "");
    }
    // Llama-style: optional <|python_tag|>, then one or more JSON objects / a JSON list.
    if calls.is_empty() {
        let t = text.trim().trim_start_matches("<|python_tag|>").trim();
        if t.starts_with('{') || t.starts_with('[') {
            let mut de = serde_json::Deserializer::from_str(t).into_iter::<Value>();
            let mut found = Vec::new();
            let mut ok = true;
            while let Some(Ok(v)) = de.next() {
                let items = if let Value::Array(a) = v { a } else { vec![v] };
                for it in items {
                    match call_from(&it, found.len()) {
                        Some(c) if it.get("arguments").is_some() || it.get("parameters").is_some() => found.push(c),
                        _ => ok = false,
                    }
                }
            }
            if ok && !found.is_empty() {
                calls = found;
                text.clear();
            }
        }
    }
    Parsed { content: text.trim().to_string(), reasoning, tool_calls: calls }
}

/// What a streamed piece is.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    /// Visible text.
    Content(String),
    /// Reasoning text.
    Reasoning(String),
}

/// Incremental splitter for streaming: routes text inside `<think>` to
/// [`Piece::Reasoning`], holds back tool-call markup (it is reported
/// structured at the end), and never cuts a tag in half.
///
/// # Example
/// ```
/// use aido::llm::conversation::{StreamSplitter, Piece};
/// let mut s = StreamSplitter::new(false);
/// let mut out = Vec::new();
/// for p in ["<thi", "nk>plan</th", "ink>Hel", "lo"] { out.extend(s.push(p)); }
/// out.extend(s.finish());
/// let content: String = out.iter().filter_map(|p| if let Piece::Content(c) = p { Some(c.as_str()) } else { None }).collect();
/// let reason: String = out.iter().filter_map(|p| if let Piece::Reasoning(c) = p { Some(c.as_str()) } else { None }).collect();
/// assert_eq!((content.as_str(), reason.as_str()), ("Hello", "plan"));
/// ```
pub struct StreamSplitter {
    buf: String,
    in_think: bool,
    in_tool: bool,
}

const TAGS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

impl StreamSplitter {
    /// `starts_in_think`: the prompt already opened a `<think>` block.
    pub fn new(starts_in_think: bool) -> Self {
        StreamSplitter { buf: String::new(), in_think: starts_in_think, in_tool: false }
    }

    fn emit(&self, s: &str, out: &mut Vec<Piece>) {
        if s.is_empty() || self.in_tool {
            return;
        }
        out.push(if self.in_think { Piece::Reasoning(s.to_string()) } else { Piece::Content(s.to_string()) });
    }

    /// Feeds generated text; returns the pieces that are now certain.
    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            let next = TAGS.iter().filter_map(|t| self.buf.find(t).map(|i| (i, *t))).min_by_key(|x| x.0);
            match next {
                Some((i, tag)) => {
                    let before = self.buf[..i].to_string();
                    self.emit(&before, &mut out);
                    match tag {
                        "<think>" => self.in_think = true,
                        "</think>" => self.in_think = false,
                        "<tool_call>" => self.in_tool = true,
                        _ => self.in_tool = false,
                    }
                    self.buf.drain(..i + tag.len());
                }
                None => {
                    // Keep a possible partial tag at the end.
                    let keep = (1..=self.buf.len().min(12)).rev().find(|&k| self.buf.is_char_boundary(self.buf.len() - k) && TAGS.iter().any(|t| t.starts_with(&self.buf[self.buf.len() - k..]))).unwrap_or(0);
                    let cut = self.buf.len() - keep;
                    let head = self.buf[..cut].to_string();
                    self.emit(&head, &mut out);
                    self.buf.drain(..cut);
                    break;
                }
            }
        }
        out
    }

    /// Flushes the remaining text.
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        self.emit(&rest, &mut out);
        out
    }
}

/// System prompt used to compress old history.
pub const SUMMARY_PROMPT: &str = "You compress conversations. Summarize the conversation below in the language it is written in. Keep every fact, name, number, decision, open question and user preference; drop pleasantries. Write compact bullet points.";
