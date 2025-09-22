#!/usr/bin/env python3
"""
Test Letta agent tools and tool calling
Investigate what tools are available to the agent
"""
import os
import asyncio
from letta_client import AsyncLetta

# Configuration from environment variables
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")

async def investigate_agent_tools():
    """Investigate what tools are available to the agent"""

    print("🔍 Investigating Letta agent tools...\n")

    # Configure client
    client_kwargs = {"base_url": LETTA_BASE_URL}
    if LETTA_API_KEY:
        if "letta.com" in LETTA_BASE_URL:
            client_kwargs.update({
                "token": LETTA_API_KEY,
                "project": os.getenv("LETTA_PROJECT", "default-project")
            })
        else:
            client_kwargs["token"] = LETTA_API_KEY

    client = AsyncLetta(**client_kwargs)

    try:
        # Get the agent
        agents = await client.agents.list()
        agent = None
        for a in agents:
            if a.name == "companion-agent-1758429513525":
                agent = a
                break

        if not agent:
            print("❌ Agent 'companion-agent-1758429513525' not found")
            return

        print(f"✅ Found agent: {agent.name} (ID: {agent.id})")

        # Get agent details
        agent_details = await client.agents.retrieve(agent.id)
        print(f"✅ Agent details retrieved")

        # Check if agent has tools
        if hasattr(agent_details, 'tools') and agent_details.tools:
            print(f"✅ Agent has {len(agent_details.tools)} tools:")
            for tool in agent_details.tools:
                print(f"   - {tool.name}: {tool.description}")
        else:
            print("⚠️  Agent has no tools configured")

        # Check if agent has tool rules
        if hasattr(agent_details, 'tool_rules') and agent_details.tool_rules:
            print(f"✅ Agent has {len(agent_details.tool_rules)} tool rules:")
            for rule in agent_details.tool_rules:
                print(f"   - {rule}")
        else:
            print("⚠️  Agent has no tool rules configured")

    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        return False

async def test_with_simple_calculator():
    """Test with a simple prompt to see if the agent can use any built-in tools"""

    print("\n🧮 Testing with simple calculator prompt...\n")

    async with AsyncLetta(base_url=LETTA_BASE_URL, token=LETTA_API_KEY) as client:
        try:
            # Get the agent
            agents = await client.agents.list()
            agent = None
            for a in agents:
                if a.name == "companion-agent-1758429513525":
                    agent = a
                    break

            if not agent:
                print("❌ Agent not found")
                return False

            # Try a simple message to see if agent has any calculation capabilities
            response = await client.agents.messages.create(
                agent_id=agent.id,
                messages=[{
                    "role": "user",
                    "content": "What is 15 + 10? Can you calculate this for me?"
                }]
            )

            print("✅ Response received:")
            for message in response.messages:
                if hasattr(message, 'content') and message.content:
                    print(f"   {message.role}: {message.content}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

if __name__ == "__main__":
    print("Investigating Letta agent tool capabilities...")

    # Investigate agent tools
    asyncio.run(investigate_agent_tools())

    # Test simple calculation
    asyncio.run(test_with_simple_calculator())