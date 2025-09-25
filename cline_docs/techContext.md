# Technical Context

## Technologies used
- **Python 3.8+**: Core programming language
- **FastAPI**: Web framework for building the API endpoints
- **Uvicorn**: ASGI server for running the FastAPI application
- **letta-client**: Official Letta Python SDK for API communication
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous programming for better performance

## Development setup
### Installation
```bash
pip install -r requirements.txt
```

### Running the application
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Non-streaming test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Milo","messages":[{"role":"user","content":"What'\''s two plus two?"}]}'

# Streaming test
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Milo","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Technical constraints
1. **Hardcoded Configuration**: Letta server URL is hardcoded in the application
2. **Global State**: Client and agent mapping stored as global variables
3. **No Authentication**: Currently no authentication mechanism implemented
4. **Limited Error Recovery**: Basic error handling without retry logic
5. **Single Server**: Designed to work with one Letta server instance

## Letta SDK Integration
### Client Configuration
- **Local Server**: `AsyncLetta(base_url="http://localhost:8283")`
- **Letta Cloud**: `AsyncLetta(token="LETTA_API_KEY", project="default-project")`
- **Async Support**: Uses `AsyncLetta` for non-blocking operations

### Key SDK Features Used
- `client.agents.list()` - Retrieve available agents
- `client.agents.messages.create()` - Send messages to agents
- `client.agents.messages.create_stream()` - Streaming responses
- Message types: `MessageCreate`, `AssistantMessage`, `ToolCallMessage`, `ToolReturnMessage`

## Performance Considerations
- Async/await patterns for concurrent request handling
- Streaming support for real-time responses
- Connection pooling through HTTP client
- Efficient message format conversion