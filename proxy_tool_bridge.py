# FILE: proxy_tool_bridge.py
"""
Letta Proxy Tool Bridge

This module implements the "proxy tool" pattern to enable OpenAI-compatible
tool calling with Letta agents. Instead of trying to make Letta execute tools
selectively, we create proxy tools that format tool calls for downstream
consumption by clients like Roo Code.

The proxy tool pattern works by:
1. **Tool Registration**: Converting OpenAI tool definitions to Letta proxy tools
2. **Exact Synchronization**: Syncing agent tool registry to match request exactly
3. **Call Formatting**: Proxy tools format calls for downstream execution
4. **Result Forwarding**: Results are forwarded back to Letta as if executed server-side

Key Features:
- **Dynamic Tool Management**: Tools are added/removed based on OpenAI requests
- **Conflict Avoidance**: Uses 'proxy_' prefix to avoid conflicts with Letta built-ins
- **Bidirectional Mapping**: Maintains mappings between OpenAI names and Letta tool IDs
- **Automatic Cleanup**: Removes unused proxy tools to keep agent registry clean
- **Parameter Handling**: Converts OpenAI parameter types to Python function signatures
- **Call ID Generation**: Generates unique tool call IDs for tracking

Usage:
    bridge = ProxyToolBridge(letta_client)
    await bridge.sync_agent_tools(agent_id, openai_tools)

    # Get original OpenAI name from Letta tool ID
    original_name = bridge.get_proxy_tool_name(letta_tool_id)

Author: Jason Owens
Version: 1.0.0
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
        self.letta_name_mapping: Dict[str, str] = {}  # Letta tool ID -> Letta name

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

        # Requested Letta-side names (prefixed), handle None case
        requested_letta_names: Set[str] = (
            {f'proxy_{tool["function"]["name"]}' for tool in openai_tools}
            if openai_tools else set()
        )

        # Remove only our proxy_ tools that are not requested; leave built-ins intact
        to_remove = {
            name for name in current_tool_names
            if name.startswith("proxy_") and name not in requested_letta_names
        }
        for tool_name in to_remove:
            tool_id = self._find_tool_id_by_name(current_tools, tool_name)
            if tool_id:
                await self.client.agents.tools.detach(agent_id, tool_id)
                self._remove_tool_from_mappings(tool_id)
                logger.info(f"Removed proxy tool {tool_name} from agent {agent_id}")

        # If no tools requested, we're done after cleanup
        if not openai_tools:
            self.tool_mapping.clear()
            self.letta_name_mapping.clear()
            logger.info("No tools requested, cleaned up all proxy tools")
            return

        # Add proxy_ tools that are requested but not present
        to_add_names = requested_letta_names - current_tool_names
        for openai_tool in openai_tools:
            prefixed = f'proxy_{openai_tool["function"]["name"]}'
            if prefixed in to_add_names:
                proxy_tool = await self._create_proxy_tool(openai_tool)
                await self.client.agents.tools.attach(agent_id, proxy_tool.id)
                self.tool_mapping[openai_tool['function']['name']] = proxy_tool.id
                self.letta_name_mapping[proxy_tool.id] = proxy_tool.name
                logger.info(f"Added proxy tool {proxy_tool.name} to agent {agent_id}")

        # Map existing proxy_ tools
        for openai_tool in openai_tools:
            prefixed = f'proxy_{openai_tool["function"]["name"]}'
            existing_tool_id = self._find_tool_id_by_name(current_tools, prefixed)
            if existing_tool_id:
                self.tool_mapping[openai_tool['function']['name']] = existing_tool_id
                self.letta_name_mapping[existing_tool_id] = prefixed

        logger.info(f"Tool sync complete. Current tools: {list(requested_letta_names)}")

    async def _create_proxy_tool(self, openai_tool: Dict[str, Any]) -> Any:
        """
        Create a proxy tool with prefixed name to avoid conflicts with Letta built-in tools.

        This method generates Python source code for a proxy tool that:
        1. Accepts the same parameters as the original OpenAI tool
        2. Captures the actual arguments passed to the function
        3. Formats them as a proxy_tool_call for downstream processing
        4. Returns the formatted call instead of executing the tool server-side

        The generated proxy tool acts as an intermediary that preserves the
        original OpenAI tool interface while routing calls through the proxy
        system for external execution.

        Args:
            openai_tool: OpenAI tool definition with function schema and parameters

        Returns:
            Letta tool object with generated source code and metadata
        """
        openai_function_name = openai_tool['function']['name']
        # Use prefixed name for Letta tool to avoid conflicts with built-in tools
        letta_function_name = f"proxy_{openai_function_name}"
        function_args = self._generate_function_args(openai_tool)

        # Create source code for proxy tool
        required_params = self._get_required_params(openai_tool)
        source_code = f"""
def {letta_function_name}({function_args}):
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
            "name": "{openai_function_name}",  # Return original OpenAI name
            "arguments": json.dumps(actual_args)
        }}
    }}
"""

        # Create the tool using Letta's upsert API with prefixed name
        proxy_tool = await self.client.tools.upsert(
            source_code=source_code,
            description=f"Proxy tool for {openai_function_name}",
            json_schema=openai_tool['function'],
            name=letta_function_name  # Use prefixed name
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
            elif param_type == 'number':
                args.append(f"{param_name}: float = 0")
            elif param_type == 'integer':
                args.append(f"{param_name}: int = 0")
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
        """
        Get required parameter names from OpenAI tool definition.

        This utility method extracts the list of required parameters from
        an OpenAI tool's function schema. Required parameters are those
        that must be provided when calling the tool and cannot be omitted.

        Args:
            openai_tool: OpenAI tool definition with function schema

        Returns:
            Set of parameter names that are required (cannot be None/empty)

        Note:
            - Returns empty set if no parameters are defined
            - Returns empty set if no 'required' field is specified
            - Used by proxy tools to filter out optional parameters with default values
        """
        parameters = openai_tool['function'].get('parameters', {})
        required = parameters.get('required', [])
        return set(required)

    def _find_tool_id_by_name(self, tools: List[Any], name: str) -> Optional[str]:
        """
        Find tool ID by name from list of tools.

        This utility method searches through a list of Letta tool objects
        to find the tool ID for a given tool name. Used during tool
        synchronization to map existing tools.

        Args:
            tools: List of Letta tool objects from the API
            name: Name of the tool to find

        Returns:
            Tool ID if found, None if not found

        Note:
            - Performs linear search through the tools list
            - Used during sync to map existing proxy tools to their IDs
            - Essential for maintaining accurate tool mappings during cleanup
        """
        for tool in tools:
            if tool.name == name:
                return tool.id
        return None

    def _remove_tool_from_mappings(self, tool_id: str) -> None:
        """
        Remove a tool from both internal mappings.

        This method cleans up the bidirectional mappings when a proxy tool
        is removed from an agent. It removes entries from both:
        - tool_mapping: OpenAI name -> Letta tool ID
        - letta_name_mapping: Letta tool ID -> Letta tool name

        Args:
            tool_id: The Letta tool ID to remove from mappings

        Note:
            - Safely handles missing keys without errors
            - Used during tool cleanup to maintain mapping consistency
            - Critical for preventing stale references after tool removal
        """
        # Remove from OpenAI name -> Letta ID mapping
        openai_names_to_remove = [name for name, tid in self.tool_mapping.items() if tid == tool_id]
        for name in openai_names_to_remove:
            del self.tool_mapping[name]

        # Remove from Letta ID -> Letta name mapping
        if tool_id in self.letta_name_mapping:
            del self.letta_name_mapping[tool_id]

    def get_letta_tool_name(self, tool_id: str) -> Optional[str]:
        """
        Get the Letta tool name for a Letta tool ID.

        This method retrieves the internal Letta name (with proxy_ prefix)
        for a given tool ID. Used when processing tool results to map
        back to the correct proxy tool name.

        Args:
            tool_id: Letta tool ID to look up

        Returns:
            Letta tool name (e.g., "proxy_my_function") if found, None otherwise

        Note:
            - Returns the prefixed Letta name, not the original OpenAI name
            - Used internally when processing tool execution results
            - Essential for maintaining correct tool identity during proxy operations
        """
        return self.letta_name_mapping.get(tool_id)

    def _generate_tool_call_id(self) -> str:
        """
        Generate a unique tool call ID.

        Creates a unique identifier for tracking tool calls through the
        proxy system. The ID format includes a "call_" prefix followed
        by a short hex string from a UUID.

        Returns:
            Unique tool call ID string (e.g., "call_a1b2c3d4")

        Note:
            - Uses first 8 characters of UUID hex for uniqueness
            - Short enough to be readable but unique enough to avoid collisions
            - Used to track individual tool calls in the proxy system
        """
        return f"call_{uuid.uuid4().hex[:8]}"

    def is_proxy_tool_call(self, tool_id: str) -> bool:
        """
        Check if a Letta tool ID belongs to a proxy tool.

        This method determines whether a given tool ID corresponds to
        one of the proxy tools managed by this bridge instance.

        Args:
            tool_id: Letta tool ID to check

        Returns:
            True if the tool ID belongs to a proxy tool, False otherwise

        Note:
            - Used to identify proxy tool calls during result processing
            - Only returns True for tools managed by this bridge instance
            - Essential for routing proxy tool results correctly
        """
        return tool_id in self.tool_mapping.values()

    def get_proxy_tool_name(self, tool_id: str) -> Optional[str]:
        """
        Get the OpenAI tool name for a proxy tool Letta ID.

        This method performs a reverse lookup to find the original OpenAI
        tool name given a Letta tool ID. This is the inverse operation
        of the tool_mapping dictionary.

        Args:
            tool_id: Letta tool ID to look up

        Returns:
            Original OpenAI tool name if found, None otherwise

        Note:
            - Returns the original OpenAI name without the proxy_ prefix
            - Used when forwarding tool results back to the client
            - Critical for maintaining correct tool identity in responses
        """
        for name, tid in self.tool_mapping.items():
            if tid == tool_id:
                return name
        return None

    async def cleanup(self, agent_id: str) -> None:
        """
        Clean up all proxy tools for an agent.

        This method performs a complete cleanup of all proxy tools associated
        with a specific agent. It clears the internal mappings but does not
        actually remove the tools from the agent - that's handled by the
        sync process when called with an empty tool list.

        Args:
            agent_id: The Letta agent ID to clean up

        Note:
            - Only clears internal mappings, doesn't detach tools from agent
            - Use sync_agent_tools() with empty list to actually remove tools
            - Called during agent reset or when proxy functionality is disabled
            - Helps maintain clean state between tool synchronization operations
        """
        logger.info(f"Cleaning up proxy tools for agent {agent_id}")
        self.tool_mapping.clear()
        self.letta_name_mapping.clear()


# Global instance for singleton pattern
proxy_bridge: Optional[ProxyToolBridge] = None


def get_proxy_bridge() -> ProxyToolBridge:
    """
    Get the global proxy tool bridge instance.

    This function implements a singleton pattern to provide access to
    the global ProxyToolBridge instance. The bridge must be initialized
    using initialize_proxy_bridge() before it can be retrieved.

    Returns:
        The global ProxyToolBridge instance

    Raises:
        RuntimeError: If the bridge has not been initialized

    Note:
        - Used by main.py to access the bridge for tool operations
        - Ensures only one bridge instance exists per application
        - Must be called after initialize_proxy_bridge()
    """
    global proxy_bridge
    if proxy_bridge is None:
        raise RuntimeError("Proxy tool bridge not initialized")
    return proxy_bridge


def initialize_proxy_bridge(letta_client: AsyncLetta) -> None:
    """
    Initialize the global proxy tool bridge.

    This function creates and sets the global ProxyToolBridge instance
    using the provided Letta client. This should be called during
    application startup before any tool operations are performed.

    Args:
        letta_client: Configured AsyncLetta client instance

    Note:
        - Should be called once during application initialization
        - Sets up the global singleton instance used throughout the app
        - Logs initialization for debugging and monitoring purposes
    """
    global proxy_bridge
    proxy_bridge = ProxyToolBridge(letta_client)
    logger.info("Proxy tool bridge initialized")
