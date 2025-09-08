# Letta OpenWebUI Pipeline

This repository provides a custom [OpenWebUI](https://github.com/open-webui) pipeline
that proxies chat requests to a [Letta](https://docs.letta.com/) agent.
The pipeline allows an OpenWebUI user to talk to a Letta agent as if it were a
regular model while streaming intermediate status updates such as reasoning and
tool usage.

## Files

- `letta_pipeline.py` – pipeline definition.
- `requirements.txt` – Python dependencies.
- `test_pipeline.py` – mock test demonstrating how to call the pipeline.

## Installation

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `letta_pipeline.py` into your OpenWebUI pipelines directory or package
   it according to the [OpenWebUI Pipelines](https://github.com/open-webui/pipelines)
   documentation.
3. Restart the OpenWebUI pipelines service so the new pipeline is detected.

## Valve Configuration

The pipeline exposes several valves which can be configured through the
OpenWebUI settings interface:

| Valve | Type | Default | Description |
|-------|------|---------|-------------|
| `letta_server_url` | string | `https://jetson-letta.resonancegroupusa.com` | Base URL of the Letta server. |
| `agent_name` | string | `Milo` | Name of the Letta agent to interact with. Case-sensitive. |
| `api_key` | secret string | _(empty)_ | Optional API key for the Letta server. Leave blank if not required. |
| `request_timeout` | integer | `120` | Maximum time (seconds) to wait for responses. |

## How It Works

Only the latest user message is forwarded to the Letta server. The Letta server
maintains its own memory and conversation state. Responses are streamed back to
OpenWebUI, including intermediate events such as reasoning steps and tool calls.

## Testing

A simple mock test is provided:

```bash
python test_pipeline.py
```

The test script simulates the OpenWebUI environment and prints the streamed
outputs received from the Letta agent.

## License

See `LICENSE` for details.
