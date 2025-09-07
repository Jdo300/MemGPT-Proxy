import json
import time
import uuid
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
from letta_client.types import Tool as LettaTool

LETTA_BASE_URL = "https://jetson-letta.resonancegroupusa.com"

app = FastAPI()

client: Optional[AsyncLetta] = None
agent_map: Dict[str, str] = {}
tool_ids: Dict[str, str] = {}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
async def startup_event() -> None:
    global client, agent_map
    client = AsyncLetta(base_url=LETTA_BASE_URL)
    agents = await client.agents.list()
    agent_map = {agent.name: agent.id for agent in agents}


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


async def _prepare_message(last: Dict[str, Any]) -> MessageCreate:
    role = last.get("role")
    if role == "user":
        content = last.get("content", "")
        return MessageCreate(role="user", content=[TextContent(text=content)])
    elif role == "tool":
        from letta_client.types import ToolReturnContent

        content = last.get("content", "")
        tool_call_id = last.get("tool_call_id")
        if not tool_call_id:
            raise HTTPException(status_code=400, detail="tool_call_id required for tool messages")
        return MessageCreate(
            role="assistant",
            content=[
                ToolReturnContent(
                    tool_call_id=tool_call_id,
                    content=content,
                    is_error=False,
                )
            ],
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported message role")


async def _ensure_tools(agent_id: str, tools: List[Dict[str, Any]]) -> None:
    """Create and attach tools for the agent if they aren't already present."""
    assert client is not None
    existing = await client.agents.tools.list(agent_id=agent_id)
    existing_map = {t.name: t.id for t in existing}
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        if name in existing_map:
            continue
        if name in tool_ids:
            tool_id = tool_ids[name]
        else:
            source = f"def {name}(**kwargs):\n    raise NotImplementedError('external tool')\n"
            created = await client.tools.create(
                name=name,
                source_code=source,
                description=fn.get("description"),
                json_schema=fn.get("parameters"),
            )
            tool_ids[name] = created.id
            tool_id = created.id
        await client.agents.tools.attach(agent_id=agent_id, tool_id=tool_id)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    if body.model not in agent_map:
        raise HTTPException(status_code=404, detail="Unknown model")
    agent_id = agent_map[body.model]
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages required")
    message = await _prepare_message(body.messages[-1])
    if body.tools:
        await _ensure_tools(agent_id, body.tools)

    if body.stream:
        async def event_stream():
            assert client is not None
            async for event in client.agents.messages.create_stream(agent_id=agent_id, messages=[message]):
                if isinstance(event, ToolCallMessage):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": event.tool_call.tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": event.tool_call.name,
                                                "arguments": event.tool_call.arguments,
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if isinstance(event, AssistantMessage):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": event.content},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    assert client is not None
    resp = await client.agents.messages.create(agent_id=agent_id, messages=[message])
    assistant: Optional[AssistantMessage] = None
    tool_calls: List[ToolCallMessage] = []
    tool_returns: List[ToolReturnMessage] = []
    for m in resp.messages:
        if isinstance(m, AssistantMessage):
            assistant = m
        elif isinstance(m, ToolCallMessage):
            tool_calls.append(m)
        elif isinstance(m, ToolReturnMessage):
            tool_returns.append(m)

    response_message: Dict[str, Any]
    finish_reason = "stop"
    if tool_calls and not tool_returns and assistant is None:
        response_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc.tool_call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_call.name,
                        "arguments": tc.tool_call.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        finish_reason = "tool_calls"
    else:
        content = assistant.content if assistant else ""
        response_message = {"role": "assistant", "content": content}

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
