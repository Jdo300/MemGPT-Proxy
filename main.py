import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from letta_client import AsyncLetta, MessageCreate
from letta_client.types import (
    AssistantMessage,
    TextContent,
    ToolCallMessage,
    ToolReturnMessage,
)

from proxy_tool_bridge import ProxyToolBridge, initialize_proxy_bridge, get_proxy_bridge

# Configuration from environment variables
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

client: Optional[AsyncLetta] = None
agent_map: Dict[str, str] = {}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


def validate_configuration() -> None:
    """Validate that required environment variables are set."""
    if LETTA_BASE_URL == "http://localhost:8283":
        logger.warning("LETTA_BASE_URL is using default value. Please set LETTA_BASE_URL environment variable.")
    if not LETTA_API_KEY:
        logger.warning("LETTA_API_KEY not set. Authentication may fail for Letta Cloud.")

@app.on_event("startup")
async def startup_event() -> None:
    global client, agent_map

    # Validate configuration
    validate_configuration()

    # Configure client with API key if provided
    client_kwargs = {"base_url": LETTA_BASE_URL}
    if LETTA_API_KEY:
        # For Letta Cloud, use token and project
        if "letta.com" in LETTA_BASE_URL:
            client_kwargs.update({
                "token": LETTA_API_KEY,
                "project": os.getenv("LETTA_PROJECT", "default-project")
            })
        else:
            # For local servers, API key might be used differently
            client_kwargs["token"] = LETTA_API_KEY

    client = AsyncLetta(**client_kwargs)

    try:
        agents = await client.agents.list()
        agent_map = {agent.name: agent.id for agent in agents}
        logger.info(f"Connected to Letta server. Found {len(agents)} agents.")

        # Initialize proxy tool bridge
        initialize_proxy_bridge(client)
        logger.info("Proxy tool bridge initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to connect to Letta server: {e}")
        raise


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    agents = await client.agents.list()
    data = [
        {
            "id": agent.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "letta",
        }
        for agent in agents
    ]
    return {"object": "list", "data": data}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "letta_base_url": LETTA_BASE_URL,
        "letta_connected": client is not None,
        "agents_loaded": len(agent_map) if agent_map else 0
    }


async def _prepare_message(last: Dict[str, Any]) -> MessageCreate:
    role = last.get("role")
    if role == "user":
        content = last.get("content", "")
        return MessageCreate(role="user", content=[TextContent(text=content)])
    else:
        # For now, only handle user messages. Tool messages would require
        # more complex handling with the Letta SDK
        raise HTTPException(status_code=400, detail=f"Unsupported message role: {role}. Currently only 'user' role is supported.")


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    # Handle model mapping - Open WebUI might use different model names than agent names
    agent_id = None
    if body.model in agent_map:
        agent_id = agent_map[body.model]
    else:
        # Try to find a matching agent (case-insensitive, partial match)
        for agent_name, aid in agent_map.items():
            if body.model.lower() in agent_name.lower() or agent_name.lower() in body.model.lower():
                agent_id = aid
                logger.info(f"Model '{body.model}' mapped to agent '{agent_name}'")
                break

        # If still no match, use the first available agent as fallback
        if not agent_id and agent_map:
            first_agent_name = list(agent_map.keys())[0]
            agent_id = agent_map[first_agent_name]
            logger.warning(f"Model '{body.model}' not found, using fallback agent '{first_agent_name}'")

    if not agent_id:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}. Available models: {list(agent_map.keys())}")

    actual_agent_name = list(agent_map.keys())[list(agent_map.values()).index(agent_id)]
    logger.info(f"Using agent: {actual_agent_name} for model: {body.model}")

    # Log tool information if provided
    if body.tools:
        logger.info(f"Request includes {len(body.tools)} tools: {[tool['function']['name'] for tool in body.tools]}")
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages required")
    message = await _prepare_message(body.messages[-1])

    # Handle tool results if provided (non-streaming case)
    if body.tool_results and not body.stream:
        # Process tool results and send them to Letta
        tool_return_messages = []
        for tool_result in body.tool_results:
            tool_return_messages.append(
                MessageCreate(
                    role="tool",
                    content=[TextContent(text=json.dumps(tool_result.get("result", "")))],
                    tool_call_id=tool_result.get("tool_call_id")
                )
            )
        
        # Send tool results to Letta agent
        if tool_return_messages:
            await client.agents.messages.create(
                agent_id=agent_id,
                messages=tool_return_messages
            )

    if body.stream:
        async def event_stream():
            assert client is not None
            try:
                # Sync tools with agent if provided
                if body.tools:
                    proxy_bridge = get_proxy_bridge()
                    await proxy_bridge.sync_agent_tools(agent_id, body.tools)
                    logger.info(f"Synced {len(body.tools)} tools with agent {agent_id}")

                # Prepare messages for streaming - include tool results if present
                streaming_messages = []
                if body.tool_results:
                    # Add tool return messages first, then the user message
                    tool_return_messages = []
                    for tool_result in body.tool_results:
                        tool_return_messages.append(
                            MessageCreate(
                                role="tool",
                                content=[TextContent(text=json.dumps(tool_result.get("result", "")))],
                                tool_call_id=tool_result.get("tool_call_id")
                            )
                        )
                    streaming_messages.extend(tool_return_messages)
                
                streaming_messages.append(message)
                
                async for event in client.agents.messages.create_stream(
                    agent_id=agent_id,
                    messages=streaming_messages,
                    stream_tokens=True
                ):
                    # Create unique ID for this chunk
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

                    # Handle reasoning/thinking messages
                    if hasattr(event, 'message_type') and event.message_type == 'reasoning_message':
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                        "reasoning": event.reasoning
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    # Handle assistant content messages
                    elif hasattr(event, 'message_type') and event.message_type == 'assistant_message':
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": event.content
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    # Handle tool calls
                    elif hasattr(event, 'message_type') and event.message_type == 'tool_call_message':
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": event.tool_call.tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": event.tool_call.name,
                                                    "arguments": event.tool_call.arguments,
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    # Handle stop/completion
                    elif hasattr(event, 'message_type') and event.message_type == 'stop_reason':
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": event.stop_reason,
                                }
                            ],
                        }

                    # Legacy handling for events without message_type
                    elif isinstance(event, ToolCallMessage):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": event.tool_call.tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": event.tool_call.name,
                                                    "arguments": event.tool_call.arguments,
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    elif isinstance(event, AssistantMessage):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": event.content
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    else:
                        # Skip unknown event types
                        continue

                    # Yield the chunk immediately for real-time streaming
                    yield f"data: {json.dumps(chunk)}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "error",
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "streaming_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

            # Send final DONE message
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    assert client is not None
    
    # Handle tool results if provided (non-streaming case)
    if body.tool_results:
        # Process tool results and send them to Letta
        tool_return_messages = []
        for tool_result in body.tool_results:
            tool_return_messages.append(
                MessageCreate(
                    role="tool",
                    content=[TextContent(text=json.dumps(tool_result.get("result", "")))],
                    tool_call_id=tool_result.get("tool_call_id")
                )
            )
        
        # Send tool results to Letta agent
        if tool_return_messages:
            await client.agents.messages.create(
                agent_id=agent_id,
                messages=tool_return_messages
            )
    
    # Sync tools with agent if provided
    if body.tools:
        proxy_bridge = get_proxy_bridge()
        await proxy_bridge.sync_agent_tools(agent_id, body.tools)
        logger.info(f"Synced {len(body.tools)} tools with agent {agent_id}")

    resp = await client.agents.messages.create(agent_id=agent_id, messages=[message])
    assistant_messages: List[AssistantMessage] = []
    tool_calls: List[ToolCallMessage] = []
    tool_returns: List[ToolReturnMessage] = []
    for m in resp.messages:
        if isinstance(m, AssistantMessage):
            assistant_messages.append(m)
        elif isinstance(m, ToolCallMessage):
            tool_calls.append(m)
        elif isinstance(m, ToolReturnMessage):
            tool_returns.append(m)

    response_message: Dict[str, Any]
    finish_reason = "stop"

    # Handle tool calls according to OpenAI API format
    if tool_calls and not tool_returns and not assistant_messages:
        response_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "index": i,
                    "id": tc.tool_call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_call.name,
                        "arguments": tc.tool_call.arguments,
                    },
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
        finish_reason = "tool_calls"
    else:
        # Handle regular assistant messages - combine all content
        combined_content = ""
        for msg in assistant_messages:
            if msg.content:
                if combined_content:
                    combined_content += "\n\n"
                combined_content += msg.content
        
        # Include reasoning from the first message that has it
        reasoning_content = ""
        for msg in assistant_messages:
            if hasattr(msg, 'reasoning') and msg.reasoning:
                reasoning_content = msg.reasoning
                break
        
        response_message = {
            "role": "assistant",
            "content": combined_content
        }
        
        # If there's reasoning content, include it in the content for OpenAI compatibility
        if reasoning_content:
            if combined_content:
                response_message["content"] = f"{combined_content}\n\n[Reasoning: {reasoning_content}]"
            else:
                response_message["content"] = reasoning_content

    usage = resp.usage
    openai_resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }
    return JSONResponse(openai_resp)
