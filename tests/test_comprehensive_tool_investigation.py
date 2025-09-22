#!/usr/bin/env python3
"""
Comprehensive investigation of Letta tool calling capabilities
Testing all possible approaches for dynamic tool definition
"""
import os
import asyncio
import json
from letta_client import AsyncLetta

# Configuration from environment variables
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")
LETTA_PROJECT = os.getenv("LETTA_PROJECT", "default-project")

class LettaToolInvestigator:
    def __init__(self):
        self.client_kwargs = {"base_url": LETTA_BASE_URL}
        if LETTA_API_KEY:
            if "letta.com" in LETTA_BASE_URL:
                self.client_kwargs.update({
                    "token": LETTA_API_KEY,
                    "project": LETTA_PROJECT
                })
            else:
                self.client_kwargs["token"] = LETTA_API_KEY

    async def get_agent(self):
        """Get the companion agent"""
        async with AsyncLetta(**self.client_kwargs) as client:
            agents = await client.agents.list()
            for agent in agents:
                if agent.name == "companion-agent-1758429513525":
                    return agent
            return None

    async def investigate_approach_1_create_and_attach_tool(self):
        """Approach 1: Create a tool and attach it to the agent"""
        print("🔧 APPROACH 1: Create and attach tool to agent")
        print("=" * 60)

        async with AsyncLetta(**self.client_kwargs) as client:
            try:
                # Get agent
                agent = await self.get_agent()
                if not agent:
                    print("❌ Agent not found")
                    return False

                print(f"✅ Found agent: {agent.name}")

                # Create a simple calculator tool
                calculator_source = '''
def calculate(expression: str) -> str:
    """Calculate a mathematical expression and return the result"""
    try:
        # Basic calculator functionality
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"
'''

                print("📝 Creating calculator tool...")
                tool = await client.tools.create(
                    source_code=calculator_source,
                    name="dynamic_calculator",
                    description="A simple calculator for basic arithmetic"
                )

                print(f"✅ Created tool: {tool.name}")

                # Try to attach tool to agent (this might not work)
                try:
                    # Check if agent has an update method
                    if hasattr(client.agents, 'update'):
                        print("🔄 Attempting to update agent with new tool...")
                        # This might fail - we're testing if it's possible
                        updated_agent = await client.agents.update(
                            agent_id=agent.id,
                            tools=[tool.id]  # Assuming this is how tools are attached
                        )
                        print("✅ Successfully attached tool to agent!")
                        return True
                    else:
                        print("⚠️  Agent update method not available")
                        return False
                except Exception as e:
                    print(f"⚠️  Could not attach tool to agent: {e}")
                    return False

            except Exception as e:
                print(f"❌ Approach 1 failed: {e}")
                return False

    async def investigate_approach_2_runtime_tool_loading(self):
        """Approach 2: Runtime tool loading with run_tool_from_source"""
        print("\n🔧 APPROACH 2: Runtime tool loading")
        print("=" * 60)

        async with AsyncLetta(**self.client_kwargs) as client:
            try:
                # Test runtime tool execution
                calculator_code = '''
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
'''

                print("⚡ Testing runtime tool execution...")
                result = await client.tools.run_tool_from_source(
                    source_code=calculator_code,
                    args={"expression": "15 * 23"}
                )

                print(f"✅ Runtime tool execution result: {result}")
                return True

            except Exception as e:
                print(f"❌ Approach 2 failed: {e}")
                return False

    async def investigate_approach_3_mcp_server_tools(self):
        """Approach 3: MCP server tool integration"""
        print("\n🔧 APPROACH 3: MCP Server integration")
        print("=" * 60)

        async with AsyncLetta(**self.client_kwargs) as client:
            try:
                # Try to connect to a simple MCP server
                # This might require an actual MCP server running
                print("⚡ Testing MCP server connection...")

                # This is a test - it might fail if no MCP server is available
                try:
                    response = await client.tools.connect_mcp_server(
                        request={
                            "server_name": "test_calculator",
                            "command": "echo",
                            "args": ["No MCP server available for testing"]
                        }
                    )
                    print(f"✅ MCP server response: {response}")
                    return True
                except Exception as mcp_error:
                    print(f"⚠️  MCP server approach failed (expected): {mcp_error}")
                    print("💡 This suggests MCP server integration exists but needs a running server")
                    return False

            except Exception as e:
                print(f"❌ Approach 3 failed: {e}")
                return False

    async def investigate_approach_4_tool_rules_investigation(self):
        """Approach 4: Investigate tool rules and agent configuration"""
        print("\n🔧 APPROACH 4: Tool rules and agent configuration")
        print("=" * 60)

        async with AsyncLetta(**self.client_kwargs) as client:
            try:
                agent = await self.get_agent()
                if not agent:
                    print("❌ Agent not found")
                    return False

                # Get detailed agent information
                agent_details = await client.agents.retrieve(agent.id)

                print("📊 Agent configuration details:")
                print(f"   System prompt: {agent_details.system[:200]}...")
                print(f"   Has tools: {hasattr(agent_details, 'tools') and bool(agent_details.tools)}")
                print(f"   Has tool_rules: {hasattr(agent_details, 'tool_rules') and bool(agent_details.tool_rules)}")

                if hasattr(agent_details, 'tools') and agent_details.tools:
                    print(f"   Tools: {len(agent_details.tools)}")
                    for tool in agent_details.tools:
                        print(f"     - {tool.name}: {tool.description}")

                if hasattr(agent_details, 'tool_rules') and agent_details.tool_rules:
                    print(f"   Tool rules: {len(agent_details.tool_rules)}")
                    for rule in agent_details.tool_rules:
                        print(f"     - {rule}")

                # Check if there are methods to modify tool rules
                print(f"   Available agent methods: {[m for m in dir(client.agents) if not m.startswith('_')]}")

                return True

            except Exception as e:
                print(f"❌ Approach 4 failed: {e}")
                return False

    async def investigate_approach_5_context_aware_tools(self):
        """Approach 5: Test if tools can be discovered from context/prompts"""
        print("\n🔧 APPROACH 5: Context-aware tool discovery")
        print("=" * 60)

        async with AsyncLetta(**self.client_kwargs) as client:
            try:
                agent = await self.get_agent()
                if not agent:
                    print("❌ Agent not found")
                    return False

                # Test with explicit tool request in message
                print("⚡ Testing explicit tool request...")
                response = await client.agents.messages.create(
                    agent_id=agent.id,
                    messages=[{
                        "role": "user",
                        "content": "I need to calculate 15 * 23. Please use a calculator tool if available."
                    }]
                )

                print("📊 Response analysis:")
                for i, message in enumerate(response.messages):
                    print(f"   Message {i+1}: {message.role}")
                    if hasattr(message, 'content') and message.content:
                        content = str(message.content)
                        print(f"     Content: {content[:300]}{'...' if len(content) > 300 else ''}")
                        # Check if response mentions tools
                        if any(keyword in content.lower() for keyword in ['tool', 'calculator', 'calculate', 'function']):
                            print("     💡 Response mentions tools/calculation!")

                return True

            except Exception as e:
                print(f"❌ Approach 5 failed: {e}")
                return False

    async def run_all_investigations(self):
        """Run all investigation approaches"""
        print("🔍 COMPREHENSIVE LETTA TOOL INVESTIGATION")
        print("Testing all possible approaches for dynamic tool definition\n")

        results = {}

        # Run all approaches
        results['approach_1'] = await self.investigate_approach_1_create_and_attach_tool()
        results['approach_2'] = await self.investigate_approach_2_runtime_tool_loading()
        results['approach_3'] = await self.investigate_approach_3_mcp_server_tools()
        results['approach_4'] = await self.investigate_approach_4_tool_rules_investigation()
        results['approach_5'] = await self.investigate_approach_5_context_aware_tools()

        # Summary
        print("\n" + "=" * 80)
        print("📋 INVESTIGATION RESULTS SUMMARY")
        print("=" * 80)

        successful_approaches = sum(results.values())
        total_approaches = len(results)

        for approach, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {approach.replace('_', ' ').title()}: {status}")

        print(f"\n🎯 Overall: {successful_approaches}/{total_approaches} approaches successful")

        if successful_approaches > 0:
            print("🎉 GOOD NEWS: Some dynamic tool approaches work!")
        else:
            print("💥 CONCLUSION: No dynamic tool approaches discovered")

        return results

async def main():
    investigator = LettaToolInvestigator()
    await investigator.run_all_investigations()

if __name__ == "__main__":
    asyncio.run(main())