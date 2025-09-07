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
)
from letta_client.types import Tool as LettaTool

LETTA_BASE_URL = "https://jetson-letta.resonancegroupusa.com"

app = FastAPI()

client: Optional[AsyncLetta] = None
agent_map: Dict[str, str] = {}
tool_ids: Dict[str, str] = {}


def _calc(expression: str, **_: Any) -> str:
    return str(eval(expression))


LOCAL_TOOL_FUNCS = {"calculator": _calc}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
async def startup_event() -> None:
    global client, agent_map
    # Allow extra time for Letta server-side tool initialization
    client = AsyncLetta(base_url=LETTA_BASE_URL, timeout=120)
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
        content = last.get("content", "")
        tool_call_id = last.get("tool_call_id")
        if not tool_call_id:
            raise HTTPException(status_code=400, detail="tool_call_id required for tool messages")
        return MessageCreate(
            role="assistant",
            content="",
            tool_returns=[{"tool_call_id": tool_call_id, "tool_return": content, "status": "success"}],
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
            # Only register a stub; actual execution happens client-side
            source = f"def {name}(**kwargs):\n    pass\n"
            schema = {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
            created = await client.tools.create(
                source_code=source,
                description=fn.get("description"),
                json_schema=schema,
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

    assert client is not None

    if body.stream:
        async def event_stream():
            messages_to_send = [message]
            while True:
                async for event in client.agents.messages.create_stream(agent_id=agent_id, messages=messages_to_send):
                    if isinstance(event, ToolCallMessage):
                        func = LOCAL_TOOL_FUNCS.get(event.tool_call.name)
                        args = json.loads(event.tool_call.arguments or "{}")
                        result = func(**args) if func else ""
                        tool_return = MessageCreate(
                            role="assistant",
                            content="",
                            tool_returns=[{
                                "tool_call_id": event.tool_call.tool_call_id,
                                "tool_return": str(result),
                                "status": "success",
                            }],
                        )
                        messages_to_send = [tool_return]
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
                        break
                    elif isinstance(event, AssistantMessage):
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
                        return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # non-streaming: loop to handle tool calls
    messages_to_send = [message]
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    assistant: Optional[AssistantMessage] = None
    last_tool_calls: List[ToolCallMessage] = []

    while True:
        resp = await client.agents.messages.create(agent_id=agent_id, messages=messages_to_send)
        usage = resp.usage
        total_prompt += usage.prompt_tokens
        total_completion += usage.completion_tokens
        total_tokens += usage.total_tokens

        messages_to_send = []
        assistant = None
        last_tool_calls = []

        for m in resp.messages:
            if isinstance(m, AssistantMessage):
                assistant = m
            elif isinstance(m, ToolCallMessage):
                last_tool_calls.append(m)

        if last_tool_calls and assistant is None:
            tc = last_tool_calls[0]
            func = LOCAL_TOOL_FUNCS.get(tc.tool_call.name)
            args = json.loads(tc.tool_call.arguments or "{}")
            result = func(**args) if func else ""
            tool_return = MessageCreate(
                role="assistant",
                content="",
                tool_returns=[{
                    "tool_call_id": tc.tool_call.tool_call_id,
                    "tool_return": str(result),
                    "status": "success",
                }],
            )
            messages_to_send = [tool_return]
            continue
        break

    response_message: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant.content if assistant else "",
    }
    if last_tool_calls:
        response_message["tool_calls"] = [
            {
                "id": tc.tool_call.tool_call_id,
                "type": "function",
                "function": {
                    "name": tc.tool_call.name,
                    "arguments": tc.tool_call.arguments,
                },
            }
            for tc in last_tool_calls
        ]

    openai_resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        },
    }
    return JSONResponse(openai_resp)
