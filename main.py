import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from letta_client import AsyncLetta, MessageCreate
from letta_client.types import (
    AssistantMessage,
    TextContent,
    ToolCallMessage,
    ToolReturnMessage,
)

from proxy_tool_bridge import ProxyToolBridge, initialize_proxy_bridge, get_proxy_bridge
from proxy_overlay import ProxyOverlayManager

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
overlay_manager: Optional[ProxyOverlayManager] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


def _normalize_content(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block.get("text", "")))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _collect_system_content(messages: List[Dict[str, Any]]) -> Optional[str]:
    system_chunks: List[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_chunks.append(_normalize_content(msg.get("content")))
    if not system_chunks:
        return None
    filtered = [chunk for chunk in system_chunks if chunk]
    if not filtered:
        return None
    return "\n\n".join(filtered)


def _extract_latest_user_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _normalize_content(msg.get("content"))
            if text:
                return text
            return ""
    return None


def _extract_trailing_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return []
    last_assistant_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = idx
    trailing = messages[last_assistant_idx + 1 :] if last_assistant_idx >= 0 else messages
    return [msg for msg in trailing if msg.get("role") == "tool"]


def validate_configuration() -> None:
    """Validate that required environment variables are set."""
    if LETTA_BASE_URL == "http://localhost:8283":
        logger.warning("LETTA_BASE_URL is using default value. Please set LETTA_BASE_URL environment variable.")
    if not LETTA_API_KEY:
        logger.warning("LETTA_API_KEY not set. Authentication may fail for Letta Cloud.")

@app.on_event("startup")
async def startup_event() -> None:
    global client, agent_map, overlay_manager

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

    # Debug: Check if the URL scheme is causing issues
    print(f"Creating Letta client with base_url: {LETTA_BASE_URL}")
    print(f"URL scheme: {LETTA_BASE_URL.split('://')[0] if '://' in LETTA_BASE_URL else 'No scheme'}")

    

    client = AsyncLetta(**client_kwargs)

    try:
        agents = await client.agents.list()
        agent_map = {agent.name: agent.id for agent in agents}
        logger.info(f"Connected to Letta server. Found {len(agents)} agents.")
    except Exception as e:
        logger.warning(f"Could not connect to Letta server on startup: {e}")
        logger.warning("Agent list will be populated on first request")
        agent_map = {}

    overlay_manager = ProxyOverlayManager(client)

    # Initialize proxy tool bridge (only if client is working)
    if agent_map:
        initialize_proxy_bridge(client)
        logger.info("Proxy tool bridge initialized successfully.")


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


if os.getenv("PROXY_DEBUG_SESSIONS") == "1":

    @app.get("/debug/sessions")
    async def debug_sessions() -> Dict[str, Any]:
        if overlay_manager is None:
            raise HTTPException(status_code=503, detail="Overlay manager unavailable")
        return overlay_manager.debug_dump()


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    # Handle model mapping - require exact agent name match
    agent_id = None
    if body.model in agent_map:
        agent_id = agent_map[body.model]
        logger.info(f"Model '{body.model}' matched to agent '{body.model}'")
    else:
        # No fallbacks allowed - require exact match
        available_agents = list(agent_map.keys())
        logger.error(f"Model '{body.model}' not found. Available agents: {available_agents}")
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}. Available models: {available_agents}")

    actual_agent_name = list(agent_map.keys())[list(agent_map.values()).index(agent_id)]
    logger.info(f"Using agent: {actual_agent_name} for model: {body.model}")

    # Log request information
    if body.tools:
        logger.info(f"Request includes {len(body.tools)} tools: {[tool['function']['name'] for tool in body.tools]}")
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages required")
    
    system_content = _collect_system_content(body.messages)

    if overlay_manager is None:
        raise HTTPException(status_code=500, detail="Overlay manager unavailable")

    headers_map = {k.lower(): v for k, v in request.headers.items()}
    session_id = overlay_manager.derive_session_id(agent_id, system_content, headers_map)
    overlay_changed, fallback_messages = await overlay_manager.apply_overlay(
        agent_id, session_id, system_content
    )

    # Prepare tool return messages from tool_results or trailing tool messages
    tool_return_messages: List[MessageCreate] = []
    seen_tool_call_ids = set()
    if body.tool_results:
        for tool_result in body.tool_results:
            tool_call_id = tool_result.get("tool_call_id")
            seen_tool_call_ids.add(tool_call_id)
            tool_payload = tool_result.get("result", "")
            tool_return_messages.append(
                MessageCreate(
                    role="tool",
                    content=[TextContent(text=json.dumps(tool_payload))],
                    tool_call_id=tool_call_id,
                )
            )

    for tool_msg in _extract_trailing_tool_messages(body.messages):
        tool_call_id = tool_msg.get("tool_call_id")
        if tool_call_id and tool_call_id in seen_tool_call_ids:
            continue
        tool_content = _normalize_content(tool_msg.get("content"))
        tool_return_messages.append(
            MessageCreate(
                role="tool",
                content=[TextContent(text=tool_content)],
                tool_call_id=tool_call_id,
            )
        )
        if tool_call_id:
            seen_tool_call_ids.add(tool_call_id)

    latest_user_text = _extract_latest_user_message(body.messages)
    user_messages_to_send: List[MessageCreate] = []
    if latest_user_text is not None:
        user_messages_to_send.append(
            MessageCreate(role="user", content=[TextContent(text=latest_user_text)])
        )

    # Compose full message payload in the correct order
    outbound_messages = [*fallback_messages, *tool_return_messages, *user_messages_to_send]

    logger.info(
        "Forwarding to Letta agent=%s session=%s overlay_changed=%s new_user_messages=%d stream=%s",
        actual_agent_name,
        session_id,
        overlay_changed,
        len(user_messages_to_send),
        body.stream,
    )

    if body.tools:
        proxy_bridge = get_proxy_bridge()
        await proxy_bridge.sync_agent_tools(agent_id, body.tools)
        logger.info(f"Synced {len(body.tools)} tools with agent {agent_id}")

    if body.stream and not outbound_messages:
        async def empty_stream():
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"
            primer = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(primer, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if body.stream:
        async def event_stream():
            assert client is not None
            # Keep one id for the entire stream (OpenAI behavior)
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"
            try:
                # Optional: primer delta with assistant role (improves client compatibility)
                primer = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(primer, ensure_ascii=False)}\n\n"
                
                async for event in client.agents.messages.create_stream(
                    agent_id=agent_id,
                    messages=outbound_messages,
                    stream_tokens=True
                ):
                    # Build chunk from event
                    if hasattr(event, 'message_type') and event.message_type == 'reasoning_message':
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": None,
                                        "reasoning": event.reasoning
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    elif hasattr(event, 'message_type') and event.message_type == 'assistant_message':
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": event.content,
                                        "reasoning": None
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    elif hasattr(event, 'message_type') and event.message_type == 'tool_call_message':
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
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

                    elif hasattr(event, 'message_type') and event.message_type == 'stop_reason':
                        chunk = {
                            "id": stream_id,
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
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
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
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": event.content,
                                        "reasoning": None
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                    else:
                        # Skip unknown event types
                        continue

                    # Yield the chunk immediately for real-time streaming (preserve raw '<' '>')
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_chunk = {
                    "id": stream_id,
                    "object": "error",
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "streaming_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

            # Send final DONE message
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    assert client is not None
    
    assert client is not None

    if not outbound_messages:
        openai_resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return Response(content=json.dumps(openai_resp, ensure_ascii=False), media_type="application/json")

    resp = await client.agents.messages.create(agent_id=agent_id, messages=outbound_messages)
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
    # Preserve raw "<" and ">" in non-stream JSON too
    return Response(content=json.dumps(openai_resp, ensure_ascii=False), media_type="application/json")
