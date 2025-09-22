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

## Configuration

The proxy uses environment variables for configuration. Set the following variables:

### Required Environment Variables

- `LETTA_BASE_URL` - URL of your Letta server (default: `https://your-letta-server.com`)
- `LETTA_API_KEY` - API key for Letta authentication (optional for local servers)

### For Letta Cloud
```bash
export LETTA_BASE_URL="https://api.letta.com"
export LETTA_API_KEY="your-letta-api-key"
export LETTA_PROJECT="your-project-name"
```

### For Local Letta Server
```bash
export LETTA_BASE_URL="http://localhost:8283"
# LETTA_API_KEY is optional for local servers
```

## Running
```bash
uvicorn main:app --reload
```

## Health Check
```bash
curl http://localhost:8000/health
```

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
