# Product Context

## Why this project exists
This project provides a bridge between AI applications that support the OpenAI Chat Completions API and Letta agents. It allows any OpenAI-compatible client to communicate with Letta's stateful agents without requiring custom integration code.

## What problems it solves
1. **API Compatibility**: Many AI applications and frameworks only support OpenAI's chat completions API format
2. **Agent Integration**: Provides seamless access to Letta's stateful agents through a familiar interface
3. **Developer Experience**: Reduces friction when integrating Letta agents into existing AI workflows
4. **Tool Calling Support**: Enables function calling capabilities between OpenAI clients and Letta agents

## How it should work
1. **API Translation**: The proxy receives OpenAI-formatted requests and translates them to Letta API calls
2. **Agent Mapping**: Maps OpenAI "model" names to Letta agent IDs
3. **Message Conversion**: Converts between OpenAI message formats and Letta message formats
4. **Response Formatting**: Translates Letta responses back to OpenAI-compatible responses
5. **Streaming Support**: Maintains streaming functionality for real-time interactions
6. **Tool Calling**: Handles function calling between the OpenAI client and Letta agents

## Key Features
- `POST /v1/chat/completions` - OpenAI-compatible chat completion endpoint
- `GET /v1/models` - Lists Letta agents as OpenAI models
- Support for streaming responses
- Tool calling integration
- Async/await support for performance