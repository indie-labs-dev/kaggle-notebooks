# SKILL: qwen-personal-backend

Reference for `notebooks/qwen-personal-backend/`.

Read this file before executing any plan for this notebook.

---

## What we are building

Personal AI backend: Qwen3-8B on Kaggle T4, connected to CF Worker via WebSocket.
No tunnels. No ngrok. Kaggle only makes outgoing connections — allowed by Kaggle ToS.

---

## Architecture (PoC-003 — WebSocket)

```
Chrome Extension
    ↓ POST /process  (with text, mode, language, previousExplanation)
CF Durable Object (permanent, stateful)
    ↓ buildPrompt() → messages array → sends over WebSocket
Kaggle Notebook (Qwen3-8B)
    ↓ generate(messages) → sends result over WebSocket
CF Durable Object
    ↓ normalizeAnalyzeResult() for analyze mode
Chrome Extension ← result
```

### Why WebSocket over Durable Objects

- **No timeouts**: WS connection lives for the entire session, analyze ~80s inference is fine
- **No KV polling**: task sent directly over socket, result returned directly
- **Auto-reconnect**: if Kaggle restarts, DO rejects pending requests cleanly, Kaggle reconnects after model reload
- **CF Worker free plan**: no 30s CPU limit issue — DO handles long-lived connections

### Key design principle

Kaggle is a CLIENT. CF Worker is the SERVER.
Kaggle makes one outgoing WS connection. CF Worker never calls Kaggle.
Prompts are built by CF Worker (R-Searcher logic), Kaggle only runs inference.

---

## CF Worker endpoints

| Method | Path | Who calls | What it does |
|---|---|---|---|
| GET | /connect | Kaggle | WebSocket upgrade, stays connected |
| POST | /process | Extension | Validates, builds prompt, sends over WS, waits for result |
| GET | /health | Anyone | Returns connected/disconnected + timestamps |

## Durable Object state

```javascript
this.ws             // WebSocket to Kaggle (null if disconnected)
this.pending        // Map: taskId → { resolve, timer }
this.connectedAt    // ISO timestamp of last connect
this.disconnectedAt // ISO timestamp of last disconnect
```

## Reconnect flow

```
Kaggle restarts
    → WS close event fires in DO
    → DO sets disconnectedAt, rejects all pending with backend_disconnected
    → Kaggle loads model (~5 min)
    → Kaggle connects → WS open → DO sets connectedAt, ws = new socket
    → Everything works again, Extension doesn't need reconfiguration
```

---

## Notebook structure (PoC-003)

```
Cell 1 — Setup: HF token, pip install, model load
Cell 2 — WebSocket loop: generate() + run_ws_loop() + idle timeout + UI
```

---

## Cell 1 — Setup (canonical, verified working)

```python
import subprocess, sys, os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"

PACKAGES = ["transformers", "accelerate", "bitsandbytes", "ipywidgets", "websockets"]

MODEL_ID = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",          # MUST be "auto" — spans both T4 GPUs
    low_cpu_mem_usage=True,     # MUST be True — avoids CPU RAM OOM during load
)
```

**Critical lessons learned:**
- `MODEL_ID = "Qwen/Qwen3-8B"` — NOT `Qwen3-8B-Instruct` (404 error)
- `device_map="auto"` — NOT `"cuda:0"` (OOM on single T4)
- `low_cpu_mem_usage=True` — required, avoids CPU RAM OOM during load
- HF token required — gated repo, add via Kaggle Secrets → `HF_TOKEN`

---

## generate() — canonical

```python
def generate(messages: list, max_tokens: int = 800, temperature: float = 0.3) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3 only — disables chain-of-thought for speed
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # model.device — NOT "cuda:0", model may span both GPUs with device_map="auto"
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, ...)
    new_tokens = output[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

---

## WebSocket loop — canonical

```python
async def run_ws_loop(worker_url: str, idle_timeout_min: int = 30):
    ws_url = worker_url.replace("https://", "wss://") + "/connect"
    async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10) as ws:
        # ping_interval=30 — keepalive ping every 30s, prevents idle disconnect
        # ping_timeout=10  — if no pong in 10s, connection is dead
        while True:
            if time.time() - last_task_time > idle_timeout_min * 60:
                break  # auto-stop — GPU not wasted when no tasks
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            task = json.loads(raw)
            content = generate(task["messages"], task["max_tokens"], task["temperature"])
            await ws.send(json.dumps({"id": task["id"], "content": content}))
```

---

## CF Worker — prompts (ported from R-Searcher)

Worker builds prompts from R-Searcher's original prompt templates:
- `SMART_ANALYZE_PROMPT(text, language)` — analyze mode
- `SMART_EXPLAIN_PROMPT(text, language)` — explain mode
- `SMART_EXPLAIN_REPHRASE/EXAMPLE/APPLICATION/IMPORTANCE_PROMPT` — follow-up modes
- `buildPrompt(mode, text, language, previousExplanation)` — dispatcher

Worker also normalizes analyze responses:
- `normalizeAnalyzeResult(raw, language)` — parses <<<SECTION_N>>> delimiters
- Fallback logic for when model doesn't follow delimiter format

---

## CF Worker bindings (wrangler.toml)

```toml
name = "qwen-personal-backend"
main = "worker.js"
compatibility_date = "2024-01-01"

[[durable_objects.bindings]]
name       = "BACKEND"
class_name = "QwenBackend"

[[migrations]]
tag         = "v1"
new_classes = ["QwenBackend"]
```

No KV needed — all state is in Durable Object memory.

---

## Idle timeout — why it matters

Kaggle gives 30h GPU/week. Using GPU 12h non-stop for 0 tasks wastes quota.
`idle_timeout_min` slider (default 30 min) auto-stops the loop when no tasks arrive.
This makes usage patterns look like normal ML workflows — GPU runs when needed, stops when not.

---

## Known issues

| Issue | Cause | Fix |
|---|---|---|
| `Qwen3-8B-Instruct` 404 | Wrong repo name | Use `Qwen/Qwen3-8B` |
| OOM with `device_map="cuda:0"` | Single T4 not enough | Use `device_map="auto"` + `low_cpu_mem_usage=True` |
| 401 Unauthorized | HF token missing | Add HF_TOKEN to Kaggle Secrets |
| WS disconnects after ~60s idle | No keepalive | `ping_interval=30` in websockets.connect() |
| backend_disconnected error | Kaggle restarted | Wait ~5 min for model reload, notebook reconnects automatically |
| Analyze slow | ~80s inference on T4 | Expected — WS has no timeout, Extension waits as long as needed |
| Session dies after 12h | Kaggle limit | User restarts notebook — personal backend pattern |
