"""End-to-end API test for `aido serve` (standard library only).

Usage:  python tests/api_test.py [base_url]
Start the server first, e.g.
  aido serve -m qwen=models/Qwen3-0.6B-Q8_0.gguf -m d1=models/d1-omni \
             -m x2=models/baikal/Baikal_LoopSR_x2_v2.safetensors --sd models/sd-turbo --taesd models/taesd
Every check prints PASS/FAIL with timing; exit code = number of failures.
"""
import base64, io, json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MEDIA = "data/test_media"
fails = 0
results = []


def call(path, body=None, stream=False, timeout=900):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:"):
            d = line[5:].strip()
            events.append(d if d == "[DONE]" else json.loads(d))
    return events


def check(name, fn):
    global fails
    t = time.time()
    try:
        info = fn()
        ok = True
    except Exception as e:  # noqa
        info, ok = f"{type(e).__name__}: {e}", False
        if hasattr(e, "read"):
            info += " " + e.read().decode()[:300]
    dt = time.time() - t
    fails += not ok
    results.append((name, ok, dt, str(info)[:160]))
    print(f"{'PASS' if ok else 'FAIL'} {name:<44} {dt:6.2f}s  {str(info)[:160]}", flush=True)


def b64file(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def magic(b):
    for sig, n in ((b"\x89PNG", "png"), (b"\xff\xd8", "jpeg"), (b"RIFF", "webp"), (b"BM", "bmp")):
        if b.startswith(sig):
            return n
    return "raw"


# ---------------------------------------------------------------- registry
def models():
    d = call("/v1/models")["data"]
    kinds = {m["id"]: m["kind"] for m in d}
    assert {"llm", "decision", "upscale", "diffusion", "custom"} <= set(kinds.values()), kinds
    return kinds
check("GET /v1/models lists all kinds", models)
check("GET /health", lambda: call("/health"))

# ---------------------------------------------------------------- text
def chat(model):
    def f():
        r = call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 24, "temperature": 0})
        txt = r["choices"][0]["message"]["content"]
        assert "paris" in txt.lower(), txt
        return f"{txt.strip()!r} {r['usage']['aido']['tokens_per_s']:.1f} tok/s"
    return f
for m in ("qwen", "llama", "qwen2.5-0.5b-instruct-q4_0"):
    check(f"chat {m}", chat(m))

def chat_stream():
    ev = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 40, "temperature": 0, "stream": True}, stream=True)
    assert ev[-1] == "[DONE]"
    pieces = [e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if "choices" in e]
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
    assert ev[-2]["choices"][0]["finish_reason"] in ("stop", "length") and "usage" in ev[-2]
    txt = "".join(pieces)
    assert "3" in txt, txt
    return f"{len(pieces)} chunks: {txt.strip()[:60]!r}"
check("chat SSE streaming (OpenAI chunks)", chat_stream)

def completion():
    r = call("/v1/completions", {"model": "llama", "prompt": "The capital of Germany is", "max_tokens": 8, "temperature": 0})
    assert "berlin" in r["choices"][0]["text"].lower(), r
    return repr(r["choices"][0]["text"])
check("completions (raw prompt)", completion)

def completion_stream():
    ev = call("/v1/completions", {"model": "llama", "prompt": "1, 2, 3,", "max_tokens": 8, "temperature": 0, "stream": True}, stream=True)
    txt = "".join(e["choices"][0]["text"] for e in ev[:-1] if "choices" in e)
    assert "4" in txt, txt
    return repr(txt)
check("completions SSE streaming", completion_stream)

def json_mode():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "Give a JSON object with keys city and country for Paris."}], "max_tokens": 60, "temperature": 0, "response_format": {"type": "json_object"}})
    obj = json.loads(r["choices"][0]["message"]["content"])
    assert isinstance(obj, dict) and obj, obj
    return obj
check("chat response_format=json_object", json_mode)

def multipart_content():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hi."}]}], "max_tokens": 12, "temperature": 0})
    return repr(r["choices"][0]["message"]["content"])
check("chat with content-part array", multipart_content)

def stop_seq():
    r = call("/v1/completions", {"model": "llama", "prompt": "1, 2, 3, 4,", "max_tokens": 30, "temperature": 0, "stop": [","]})
    assert "," not in r["choices"][0]["text"], r
    return repr(r["choices"][0]["text"])
check("stop sequences", stop_seq)

def bad_model():
    try:
        call("/v1/chat/completions", {"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    except urllib.error.HTTPError as e:
        assert e.code == 400
        return json.loads(e.read())["error"]["message"]
    raise AssertionError("no error")
check("unknown model -> 400 JSON error", bad_model)

def wrong_kind():
    try:
        call("/v1/chat/completions", {"model": "x2", "messages": [{"role": "user", "content": "x"}]})
    except urllib.error.HTTPError as e:
        return json.loads(e.read())["error"]["message"]
    raise AssertionError("no error")
check("wrong model kind -> error", wrong_kind)

# ---------------------------------------------------------------- chat features
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def tool_call():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
                                      "tools": TOOLS, "max_tokens": 120, "temperature": 0})
    c = r["choices"][0]
    assert c["finish_reason"] == "tool_calls", c
    tc = c["message"]["tool_calls"][0]
    args = json.loads(tc["function"]["arguments"])
    assert tc["function"]["name"] == "get_weather" and "paris" in args["city"].lower(), tc
    return tc["function"]
check("tools: model emits structured tool_call", tool_call)

def tool_roundtrip():
    msgs = [{"role": "user", "content": "What is the weather in Paris right now?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "{\"temperature_c\": 17, \"sky\": \"rain\"}"}]
    r = call("/v1/chat/completions", {"model": "qwen", "messages": msgs, "tools": TOOLS, "max_tokens": 80, "temperature": 0})
    txt = r["choices"][0]["message"]["content"] or ""
    assert "17" in txt or "rain" in txt.lower(), txt
    return repr(txt[:100])
check("tools: answer uses tool result", tool_roundtrip)

def tool_stream():
    ev = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "Weather in Tokyo?"}], "tools": TOOLS, "max_tokens": 120, "temperature": 0, "stream": True}, stream=True)
    calls = [e["choices"][0]["delta"]["tool_calls"] for e in ev[:-1] if "choices" in e and "tool_calls" in e["choices"][0]["delta"]]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if "choices" in e)
    assert calls and "<tool_call>" not in text, (calls, text)
    return calls[0][0]["function"]
check("tools: streaming delta.tool_calls, no markup leak", tool_stream)

def thinking():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "What is 17*3? Answer with the number."}], "max_tokens": 400, "temperature": 0, "enable_thinking": True})
    m = r["choices"][0]["message"]
    assert m.get("reasoning_content") and "<think>" not in (m["content"] or "") and "51" in m["content"], m
    return f"reasoning {len(m['reasoning_content'])} chars, answer {m['content'].strip()!r}"
check("thinking: reasoning_content separated", thinking)

def thinking_stream():
    ev = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "What is 6*7?"}], "max_tokens": 300, "temperature": 0, "enable_thinking": True, "stream": True}, stream=True)
    rs = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in ev[:-1] if "choices" in e)
    ct = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if "choices" in e)
    assert rs and "42" in ct and "think>" not in ct, (rs[:80], ct)
    return f"{len(rs)} reasoning chars, answer {ct.strip()[:40]!r}"
check("thinking: streamed reasoning_content", thinking_stream)

def n_choices():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "Name a random fruit."}], "max_tokens": 20, "temperature": 1.0, "n": 3, "seed": 5})
    assert len(r["choices"]) == 3 and [c["index"] for c in r["choices"]] == [0, 1, 2], r
    return [c["message"]["content"].strip()[:20] for c in r["choices"]]
check("n=3 alternative answers", n_choices)

def system_msg():
    r = call("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "system", "content": "You are a pirate. Always start your answer with 'Arr'."}, {"role": "user", "content": "Hello!"}], "max_tokens": 30, "temperature": 0})
    t = r["choices"][0]["message"]["content"]
    assert t.strip().lower().startswith("arr"), t
    return repr(t[:60])
check("system message is followed", system_msg)

def prompt_cache():
    hist = [{"role": "system", "content": "You are a helpful assistant. " * 40}, {"role": "user", "content": "Say one word."}]
    call("/v1/chat/completions", {"model": "qwen", "messages": hist, "max_tokens": 4, "temperature": 0})
    r = call("/v1/chat/completions", {"model": "qwen", "messages": hist + [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Another word."}], "max_tokens": 4, "temperature": 0})
    cached = r["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached > 200, r["usage"]
    return f"{cached}/{r['usage']['prompt_tokens']} prompt tokens from cache"
check("token cache: follow-up reuses prefix", prompt_cache)

def memory_flow():
    u = f"test_user_{int(time.time())}"
    call("/v1/memory", {"user": u, "text": "The user's cat is called Murzik and is 4 years old."})
    call("/v1/memory", {"user": u, "text": "The user lives in Kazan."})
    hits = call(f"/v1/memory?user={u}&query=cat%20name&k=1")["data"]
    assert "Murzik" in hits[0]["text"], hits
    r = call("/v1/chat/completions", {"model": "qwen", "user": u, "messages": [{"role": "user", "content": "What is my cat's name? Answer with the name."}], "max_tokens": 200, "temperature": 0})
    txt = r["choices"][0]["message"]["content"]
    assert "murzik" in txt.lower(), (txt, r["usage"])
    return f"{txt.strip()[:40]!r}, memories_used={r['usage']['aido']['memories_used']}"
check("long-term memory (vector search) in chat", memory_flow)

def auto_summary():
    long_hist = []
    for i in range(30):
        long_hist += [{"role": "user", "content": f"Fact {i}: item number {i} costs {i * 7} dollars. " * 12}, {"role": "assistant", "content": "Noted."}]
    long_hist.append({"role": "user", "content": "Say OK."})
    r = call("/v1/chat/completions", {"model": "qwen", "messages": long_hist, "max_tokens": 8, "temperature": 0})
    n = r["usage"]["aido"]["summarized_messages"]
    assert n > 0, r["usage"]
    return f"{n} messages folded, prompt {r['usage']['prompt_tokens']} tokens"
check("auto-summary of history beyond the context", auto_summary)

# ---------------------------------------------------------------- custom SDK model
def predict_seq():
    out = []
    for seq, nxt in (([2, 4, 6, 8, 10], 12), ([5, 10, 15, 20, 25], 30), ([1, 2, 4, 8, 16], 32)):
        r = call("/v1/predict", {"model": "seq", "input": {"sequence": seq}})
        out.append((r["output"]["next"], nxt))
    ok = sum(a == b for a, b in out)
    assert ok >= 2, out
    return f"{ok}/3 correct {out}"
check("custom SDK model /v1/predict (seq)", predict_seq)

def predict_bad_input():
    try:
        call("/v1/predict", {"model": "seq", "input": {"sequence": "abc"}})
    except urllib.error.HTTPError as e:
        return json.loads(e.read())["error"]["message"]
    raise AssertionError("no error")
check("custom model input validation", predict_bad_input)

def models_io():
    d = {m["id"]: m for m in call("/v1/models")["data"]}
    assert d["seq"]["kind"] == "custom" and d["seq"]["io"]["inputs"][0]["name"] == "sequence"
    return d["seq"]["io"]["task"]
check("/v1/models exposes I/O signature", models_io)

# ---------------------------------------------------------------- decision
SUPPORT_Q = {
    "intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": "Money back", "track": "Where is my order", "cancel": "Cancel the order", "other": "Anything else"}},
    "topics": {"type": "choice", "multiple": True, "instructions": "Which topics are mentioned?", "criteria": ["delivery", "payment", "quality", "price"]},
    "anger": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "angry", "furious"]},
    "days": {"type": "number", "instructions": "How many days late is the order?", "range": [0, 30], "step": 1, "unit": "days"},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}
def decide_text():
    r = call("/v1/decide", {"model": "d1", "state": "Customer: My package is 5 days late and I want my money back NOW. This is ridiculous!", "questions": SUPPORT_Q})
    a = r["answers"]
    assert set(a) == set(SUPPORT_Q), r
    assert a["refund"]["noul"] > 0.5 and a["intent"]["choice"] == "refund", a
    return json.dumps(a, ensure_ascii=False)[:150] + f" {r.get('timing_ms'):.0f} ms"
check("decide: text, 4 answer types", decide_text)

def decide_image(fmt):
    def f():
        q = {"animal": {"type": "choice", "instructions": "What animal is in the picture?", "criteria": ["cat", "dog", "bird", "car"]},
             "count": {"type": "number", "instructions": "How many animals are visible?", "range": [0, 10], "step": 1}}
        r = call("/v1/decide", {"model": "d1", "questions": q, "media": [{"type": "image", "data": b64file(f"{MEDIA}/cats.{fmt}")}]})
        a = r["answers"]
        assert a["animal"]["choice"] == "cat", a
        return json.dumps(a)[:150]
    return f
for fmt in ("jpg", "png", "webp", "bmp", "gif", "tiff"):
    check(f"decide: image as {fmt}", decide_image(fmt))

def decide_data_url():
    r = call("/v1/decide", {"model": "d1", "questions": {"cat": {"type": "noul", "instructions": "Is there a cat in the picture?"}},
                            "media": [{"type": "image", "url": "data:image/jpeg;base64," + b64file(f"{MEDIA}/cats.jpg")}]})
    assert r["answers"]["cat"]["noul"] > 0.5, r
    return json.dumps(r["answers"])
check("decide: image as data URL", decide_data_url)

def decide_audio():
    q = {"topic": {"type": "choice", "instructions": "What is the speaker asking about?", "criteria": ["weather", "order delivery", "music", "cancel subscription"]}}
    r = call("/v1/decide", {"model": "d1", "questions": q, "media": [{"type": "audio", "data": b64file(f"{MEDIA}/speech_weather.wav")}]})
    assert r["answers"]["topic"]["choice"] == "weather", r
    return json.dumps(r["answers"])
check("decide: audio (wav)", decide_audio)

def decide_box():
    r = call("/v1/decide", {"model": "d1", "questions": {"where": {"type": "box", "instructions": "Where is the cat?"}}, "media": [{"type": "image", "path": f"{MEDIA}/cats.jpg"}]})
    return json.dumps(r["answers"])
check("decide: bounding box answer (path input)", decide_box)

def decide_image_and_text():
    q = {"match": {"type": "noul", "instructions": "Does the picture match the listing description?"}}
    r = call("/v1/decide", {"model": "d1", "state": "Listing: two cats sleeping on a couch", "questions": q, "media": [{"type": "image", "path": f"{MEDIA}/cats.png"}]})
    return json.dumps(r["answers"])
check("decide: text + image together", decide_image_and_text)

# ---------------------------------------------------------------- upscale
def upscale(fmt, out):
    def f():
        r = call("/v1/images/upscale", {"model": "x2", "image": b64file(f"{MEDIA}/{fmt}"), "loops": 1, "output_format": out})
        data = base64.b64decode(r["data"][0]["b64_json"])
        assert r["width"] == 2 * r["input"]["width"], r
        if out == "raw":
            assert len(data) == r["width"] * r["height"] * 3
        else:
            assert magic(data) == ("jpeg" if out == "jpg" else out), magic(data)
        return f"{r['input']['width']}x{r['input']['height']} -> {r['width']}x{r['height']} {out} {len(data)} B, {r['ms']:.0f} ms"
    return f
for inp, out in (("a64.png", "png"), ("a64.png", "jpeg"), ("a64.png", "webp"), ("a64.png", "bmp"), ("a64.png", "raw")):
    check(f"upscale {inp} -> {out}", upscale(inp, out))

def upscale_url():
    r = call("/v1/images/upscale", {"image": "data:image/png;base64," + b64file(f"{MEDIA}/a64.png"), "loops": 1, "response_format": "url", "output_format": "jpeg"})
    assert r["data"][0]["url"].startswith("data:image/jpeg;base64,")
    return "data URL ok, default model " + r["model"]
check("upscale default model, url response", upscale_url)

# ---------------------------------------------------------------- diffusion
def image(out):
    def f():
        r = call("/v1/images/generations", {"prompt": "a cute red fox, digital art", "size": "512x512", "steps": 1, "seed": 7, "output_format": out})
        data = base64.b64decode(r["data"][0]["b64_json"])
        assert magic(data) == out, magic(data)
        open(f"{MEDIA}/api_fox.{out}", "wb").write(data)
        return f"{len(data)} B, {r['ms']:.0f} ms, {r['timings']}"
    return f
for out in ("png", "jpeg", "webp"):
    check(f"image generation -> {out}", image(out))

def image_stream():
    ev = call("/v1/images/generations", {"prompt": "a lighthouse at sunset", "size": "512x512", "steps": 4, "seed": 3, "stream": True}, stream=True)
    prev = [e for e in ev if isinstance(e, dict) and e.get("type") == "preview"]
    fin = [e for e in ev if isinstance(e, dict) and e.get("type") == "final"]
    assert len(prev) == 4 and fin and ev[-1] == "[DONE]", [e.get("type") if isinstance(e, dict) else e for e in ev]
    return f"{len(prev)} previews (steps {[p['step'] for p in prev]}) + final"
check("image generation SSE step previews", image_stream)

def image_n2():
    r = call("/v1/images/generations", {"prompt": "pixel art castle", "size": "256x256", "n": 2, "seed": 1, "response_format": "url"})
    assert len(r["data"]) == 2 and r["data"][0]["url"].startswith("data:image/png")
    return "2 images as data URLs"
check("image generation n=2, 256x256, url", image_n2)

# ---------------------------------------------------------------- state
def loaded():
    d = call("/v1/models")["data"]
    return {m["id"]: m["loaded"] for m in d}
check("models loaded/unloaded state", loaded)

print(f"\n{len(results) - fails}/{len(results)} passed")
json.dump([{"name": n, "ok": o, "seconds": round(t, 2), "info": i} for n, o, t, i in results], open("data/api_test_results.json", "w"), indent=1, ensure_ascii=False)
sys.exit(fails)
