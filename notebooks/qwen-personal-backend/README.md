# Qwen Personal Backend

Personal AI backend: Qwen3-8B on Kaggle T4 connected to CF Worker via WebSocket.

## Status

- [x] PoC-001: load model + smoke test ✅
- [x] PoC-002: long polling (replaced by WebSocket) ✅
- [x] PoC-003: WebSocket via CF Durable Object ✅
- [ ] PoC-004: R-Searcher Extension integration

## How it works

```
Chrome Extension → POST /process → CF Durable Object
                                         ↕ WebSocket (persistent)
                                   Kaggle Notebook (Qwen3-8B)
```

1. Kaggle notebook opens a WebSocket to CF Worker on startup
2. Extension sends requests to CF Worker as usual (`POST /process`)
3. CF Worker builds the prompt (R-Searcher logic) and sends it over WebSocket to Kaggle
4. Kaggle runs inference and sends result back over the same WebSocket
5. CF Worker returns result to Extension

No tunnels. No ngrok. Kaggle only makes outgoing connections.

## Why WebSocket

- No timeouts — analyze mode takes ~80s, WebSocket has no request timeout
- No KV polling — task sent directly, result returned directly
- Auto-reconnect — if Kaggle restarts, notebook reconnects after model reload
- CF Worker free plan compatible — Durable Objects handle long-lived connections

## Idle auto-stop

The notebook auto-stops the WebSocket loop after N minutes without tasks (default 30 min).
This prevents wasting Kaggle GPU quota when not actively using the extension.
Kaggle gives 30h GPU/week — idle-stop makes usage responsible.

## Requirements

- Kaggle GPU: T4 x2 (both GPUs needed — model uses `device_map="auto"`)
- Kaggle Internet: On
- HuggingFace account + token (Qwen3-8B is a gated repo)
- Cloudflare account (free tier — Durable Objects included)

## How to get a HuggingFace token

1. Go to [huggingface.co](https://huggingface.co) → Sign in or create account
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token** → Name it (e.g. `kaggle-qwen`) → Role: **Read** → **Generate token**
4. Copy the token (starts with `hf_...`)
5. Accept the Qwen3-8B license at [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) → click **Agree**
6. In Kaggle notebook → **Add-ons** → **Secrets** → **Add new secret**
   - Name: `HF_TOKEN`
   - Value: your token (`hf_...`)
7. Toggle the secret on for your notebook

## How to deploy the CF Worker

```bash
cd notebooks/qwen-personal-backend/worker
wrangler deploy
```

Durable Objects are available on Cloudflare Workers free plan.

## How to run the notebook

1. Enable GPU (T4 x2) and Internet in Kaggle notebook settings
2. Add HF_TOKEN to Kaggle Secrets (see above)
3. Run All (Shift+F10) — model loads in ~3-5 min
4. Enter Worker URL and click **▶ Connect**
5. Set idle stop time (default 30 min) — notebook auto-stops when not in use

## What happens on notebook restart

1. WebSocket closes → CF Worker rejects any pending requests with `backend_disconnected`
2. Kaggle loads model (~5 min)
3. Notebook reconnects → everything works again
4. Extension doesn't need reconfiguration — Worker URL stays the same

## Model

- `Qwen/Qwen3-8B` — correct repo ID (NOT `Qwen3-8B-Instruct`, that's 404)
- Loaded in 4-bit via bitsandbytes (~5 GiB VRAM across both T4 GPUs)
- `device_map="auto"` distributes layers across both GPUs automatically

## Stack

- **Qwen3-8B** 4-bit (bitsandbytes) — inference
- **websockets** — Python WS client for Kaggle↔Worker communication
- **CF Durable Objects** — stateful WebSocket relay, permanent URL
- **ipywidgets** — notebook UI

## Files

```
notebooks/qwen-personal-backend/
├── notebook.ipynb      — Kaggle notebook (model load + WS loop)
└── worker/
    ├── worker.js       — CF Worker with Durable Object + R-Searcher prompts
    └── wrangler.toml   — CF deployment config
```
