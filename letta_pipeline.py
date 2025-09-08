"""
Letta OpenWebUI Pipeline
------------------------
This pipeline bridges OpenWebUI to a Letta agent server.
"""

from typing import List, Dict, Iterator, Generator, Union, Optional

from pydantic import BaseModel, Field, SecretStr

from letta_client import Letta
from letta_client.types import (
    MessageCreate,
    TextContent,
    ReasoningMessage,
    HiddenReasoningMessage,
    ToolCallMessage,
    ToolReturnMessage,
    AssistantMessage,
    LettaPing,
    LettaStopReason,
    LettaUsageStatistics,
)


class Pipeline:
    class Valves(BaseModel):
        letta_server_url: str = Field(
            default="https://jetson-letta.resonancegroupusa.com",
            description="Base URL of the Letta server",
        )
        agent_name: str = Field(
            default="Milo", description="Name of the Letta agent to use"
        )
        api_key: Optional[SecretStr] = Field(
            default=None, description="API key for authenticating with Letta"
        )
        request_timeout: int = Field(
            default=120,
            description="Request timeout in seconds",
        )

    def __init__(self) -> None:
        self.type = "manifold"
        self.name = "Letta"
        self.valves = self.Valves()

    # Manifold pipeline listing
    def pipelines(self):
        return [{"id": self.valves.agent_name, "name": self.valves.agent_name}]

    # internal helper to create a Letta client based on valves
    def _get_client(self) -> Letta:
        token = (
            self.valves.api_key.get_secret_value()
            if isinstance(self.valves.api_key, SecretStr)
            else None
        )
        client = Letta(
            base_url=self.valves.letta_server_url,
            token=token,
            timeout=self.valves.request_timeout,
        )
        return client

    # core pipe function
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict],
        body: Dict,
    ) -> Union[str, Generator, Iterator]:
        """Forward the latest user message to Letta and stream the response."""

        agent_name = model_id or self.valves.agent_name
        client = self._get_client()

        try:
            agents = client.agents.list(name=agent_name)
            if not agents:
                raise ValueError(f"Agent '{agent_name}' not found on Letta server")
            agent_id = agents[0].id

            request = MessageCreate(
                role="user", content=[TextContent(text=user_message)]
            )

            stream = client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[request],
                stream_tokens=True,
                include_pings=True,
            )

            for event in stream:
                # Ping messages for keep-alive
                if isinstance(event, LettaPing):
                    yield {"type": "status", "status": "ping"}
                # Model reasoning (visible)
                elif isinstance(event, ReasoningMessage):
                    yield {
                        "type": "status",
                        "status": "thinking",
                        "message": event.reasoning,
                    }
                # Hidden reasoning (redacted/omitted)
                elif isinstance(event, HiddenReasoningMessage):
                    yield {
                        "type": "status",
                        "status": "hidden_reasoning",
                        "state": event.state,
                    }
                # Tool call details
                elif isinstance(event, ToolCallMessage):
                    tool_name = getattr(event.tool_call, "name", None)
                    args = getattr(event.tool_call, "arguments", None)
                    yield {
                        "type": "status",
                        "status": "tool_call",
                        "tool_name": tool_name,
                        "args": args,
                    }
                # Tool return result
                elif isinstance(event, ToolReturnMessage):
                    yield {
                        "type": "status",
                        "status": "tool_result",
                        "tool_call_id": event.tool_call_id,
                        "result": event.tool_return,
                        "completion": event.status,
                    }
                # Assistant output
                elif isinstance(event, AssistantMessage):
                    content = event.content
                    if isinstance(content, list):
                        for item in content:
                            if hasattr(item, "text"):
                                yield item.text
                    else:
                        yield content
                # Stop reason
                elif isinstance(event, LettaStopReason):
                    yield {
                        "type": "status",
                        "status": "stop",
                        "reason": event.stop_reason,
                    }
                # Usage statistics
                elif isinstance(event, LettaUsageStatistics):
                    yield {
                        "type": "status",
                        "status": "usage",
                        "data": event.model_dump(),
                    }
        except Exception as e:
            yield {"type": "error", "error": str(e)}
