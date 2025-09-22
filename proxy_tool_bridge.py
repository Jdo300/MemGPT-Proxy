"""
Letta Proxy Tool Bridge

This module implements the "proxy tool" pattern to enable OpenAI-compatible
tool calling with Letta agents. Instead of trying to make Letta execute tools
selectively, we create proxy tools that format tool calls for downstream
consumption by clients like Roo Code.

Architecture:
1. Convert OpenAI tool definitions to Letta proxy tools
2. Sync agent tool registry to match request exactly
3. Proxy tools format calls for downstream execution
4. Results are forwarded back to Letta as if executed server-side
"""

import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Set
from letta_client import AsyncLetta

logger = logging.getLogger(__name__)


class ProxyToolBridge:
    """
    Bridge between OpenAI tool definitions and Letta proxy tools.

    This class manages the lifecycle of proxy tools that format tool calls
    for downstream consumption rather than executing them server-side.
    """

    def __init__(self, letta_client: AsyncLetta):
        """Initialize the proxy tool bridge."""
        self.client = letta_client
        self.tool_mapping: Dict[str, str] = {}  # OpenAI name -> Letta tool ID

    async def sync_agent_tools(self, agent_id: str, openai_tools: List[Dict[str, Any]]) -> None:
        """
        Sync agent tools to match exactly what's in the OpenAI request.

        Args:
            agent_id: The Letta agent ID
            openai_tools: List of OpenAI tool definitions
        """
        logger.info(f"Syncing tools for agent {agent_id}")

        # Get current agent tools
        current_tools = await self.client.agents.tools.list(agent_id)
        current_tool_names = {tool.name for tool in current_tools}

        # Calculate requested tools (handle None case)
        requested_tool_names = {tool['function']['name'] for tool in openai_tools} if openai_tools else set()

        # Remove tools not in request (or all tools if no tools requested)
        to_remove = current_tool_names - requested_tool_names
        for tool_name in to_remove:
            tool_id = self._find_tool_id_by_name(current_tools, tool_name)
            if tool_id:
                await self.client.agents.tools.detach(agent_id, tool_id)
                logger.info(f"Removed tool {tool_name} from agent {agent_id}")

        # If no tools requested, we're done after cleanup
        if not openai_tools:
            self.tool_mapping.clear()
            logger.info("No tools requested, cleaned up all proxy tools")
            return

        # Add tools in request but not on agent
        to_add = requested_tool_names - current_tool_names
        for openai_tool in openai_tools:
            if openai_tool['function']['name'] in to_add:
                proxy_tool = await self._create_proxy_tool(openai_tool)
                await self.client.agents.tools.attach(agent_id, proxy_tool.id)
                self.tool_mapping[openai_tool['function']['name']] = proxy_tool.id
                logger.info(f"Added proxy tool {openai_tool['function']['name']} to agent {agent_id}")

        # Update mapping for existing tools
        for openai_tool in openai_tools:
            if openai_tool['function']['name'] not in to_add:
                # Tool already exists, update mapping
                existing_tool_id = self._find_tool_id_by_name(current_tools, openai_tool['function']['name'])
                if existing_tool_id:
                    self.tool_mapping[openai_tool['function']['name']] = existing_tool_id

        logger.info(f"Tool sync complete. Current tools: {list(requested_tool_names)}")

    async def _create_proxy_tool(self, openai_tool: Dict[str, Any]) -> Any:
        """
        Create a proxy tool that formats calls for downstream consumption.

        Args:
            openai_tool: OpenAI tool definition

        Returns:
            Letta tool object
        """
        function_name = openai_tool['function']['name']
        function_args = self._generate_function_args(openai_tool)

        # Create source code for proxy tool
        required_params = self._get_required_params(openai_tool)
        source_code = f"""
def {function_name}({function_args}):
    # This is a proxy tool that formats calls for downstream execution
    # Return the tool call in a format that can be processed by the proxy
    import json

    # Get the actual arguments passed to this function
    import inspect
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    actual_args = {{arg: values[arg] for arg in args if arg != 'self'}}
    
    # Remove empty/default values to match OpenAI behavior
    required_params = {list(required_params)}
    actual_args = {{k: v for k, v in actual_args.items() if v not in [None, '', 0, [], {{}}] or k in required_params}}

    return {{
        "type": "proxy_tool_call",
        "tool_call_id": "{self._generate_tool_call_id()}",
        "function": {{
            "name": "{function_name}",
            "arguments": json.dumps(actual_args)
        }}
    }}
"""

        # Create the tool using Letta's upsert API
        proxy_tool = await self.client.tools.upsert(
            source_code=source_code,
            description=f"Proxy tool for {function_name}",
            json_schema=openai_tool['function']
        )

        return proxy_tool

    def _generate_function_args(self, openai_tool: Dict[str, Any]) -> str:
        """Generate function arguments from OpenAI tool parameters."""
        parameters = openai_tool['function'].get('parameters', {})
        properties = parameters.get('properties', {})

        args = []
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'str')
            if param_type == 'string':
                args.append(f"{param_name}: str = ''")
            elif param_type == 'number' or param_type == 'integer':
                args.append(f"{param_name}: float = 0")
            elif param_type == 'boolean':
                args.append(f"{param_name}: bool = False")
            elif param_type == 'array':
                args.append(f"{param_name}: list = []")
            elif param_type == 'object':
                args.append(f"{param_name}: dict = {{}}")
            else:
                args.append(f"{param_name}: str = ''")

        return ", ".join(args)

    def _get_required_params(self, openai_tool: Dict[str, Any]) -> Set[str]:
        """Get required parameter names from OpenAI tool definition."""
        parameters = openai_tool['function'].get('parameters', {})
        required = parameters.get('required', [])
        return set(required)

    def _find_tool_id_by_name(self, tools: List[Any], name: str) -> Optional[str]:
        """Find tool ID by name from list of tools."""
        for tool in tools:
            if tool.name == name:
                return tool.id
        return None

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:8]}"

    def is_proxy_tool_call(self, tool_call_id: str) -> bool:
        """Check if a tool call ID belongs to a proxy tool."""
        return tool_call_id in self.tool_mapping.values()

    def get_proxy_tool_name(self, tool_call_id: str) -> Optional[str]:
        """Get the OpenAI tool name for a proxy tool call ID."""
        for name, call_id in self.tool_mapping.items():
            if call_id == tool_call_id:
                return name
        return None

    async def cleanup(self, agent_id: str) -> None:
        """Clean up all proxy tools for an agent."""
        logger.info(f"Cleaning up proxy tools for agent {agent_id}")
        self.tool_mapping.clear()


# Global instance
proxy_bridge: Optional[ProxyToolBridge] = None


def get_proxy_bridge() -> ProxyToolBridge:
    """Get or create the global proxy tool bridge instance."""
    global proxy_bridge
    if proxy_bridge is None:
        raise RuntimeError("Proxy tool bridge not initialized")
    return proxy_bridge


def initialize_proxy_bridge(letta_client: AsyncLetta) -> None:
    """Initialize the global proxy tool bridge."""
    global proxy_bridge
    proxy_bridge = ProxyToolBridge(letta_client)
    logger.info("Proxy tool bridge initialized")