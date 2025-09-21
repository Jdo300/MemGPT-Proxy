# Letta OpenAI Proxy

This project provides a FastAPI server that exposes a minimal OpenAI-compatible API and forwards requests to a [Letta](https://docs.letta.com/) agent server.

## Features
- `POST /v1/chat/completions` – accepts OpenAI-style chat completion requests and routes the latest message to a Letta agent.
- `GET /v1/models` – lists available agents on the Letta server as if they were OpenAI models.
- Basic support for tool calling and streaming.

## Setup
```bash
pip install -r requirements.txt
```

## Running
```bash
uvicorn main:app --reload
```

The proxy assumes a Letta server is available at `https://your-letta-server.com`.

## Testing
```bash
# non-streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Milo","messages":[{"role":"user","content":"What\'s two plus two?"}]}'

# streaming
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Milo","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```
