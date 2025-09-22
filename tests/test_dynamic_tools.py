#!/usr/bin/env python3
"""
Test dynamic tool definition with Letta SDK directly
"""
import os
import asyncio
from letta_client import AsyncLetta

# Configuration from environment variables
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")

async def test_dynamic_tools_directly():
    """Test if Letta SDK supports dynamic tool definition"""

    print("🔧 Testing dynamic tool definition with Letta SDK directly...\n")

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
            print("❌ Agent not found")
            return False

        print(f"✅ Found agent: {agent.name}")

        # Test with use_assistant_message=False to see raw tool calls
        print("\n🧮 Testing with use_assistant_message=False...")

        response = await client.agents.messages.create(
            agent_id=agent.id,
            messages=[{
                "role": "user",
                "content": "What is 15 times 23? Please calculate this for me."
            }],
            use_assistant_message=False  # This should return raw tool calls
        )

        print("📊 Response details:")
        print(f"   Number of messages: {len(response.messages)}")

        for i, message in enumerate(response.messages):
            print(f"   Message {i+1}:")
            print(f"     Role: {message.role}")
            print(f"     Type: {type(message)}")
            print(f"     Has content: {hasattr(message, 'content') and message.content is not None}")
            print(f"     Has tool_calls: {hasattr(message, 'tool_calls') and message.tool_calls is not None}")

            if hasattr(message, 'content') and message.content:
                print(f"     Content: {message.content[:200]}{'...' if len(str(message.content)) > 200 else ''}")

            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"     Tool calls: {len(message.tool_calls)}")
                for j, tool_call in enumerate(message.tool_calls):
                    print(f"       Tool call {j+1}: {tool_call.name}({tool_call.arguments})")

        # Check if we got tool calls
        tool_calls_found = []
        for message in response.messages:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_found.extend(message.tool_calls)

        if tool_calls_found:
            print("✅ SUCCESS: Tool calls found!")
            for i, tool_call in enumerate(tool_calls_found):
                print(f"   Tool call {i+1}: {tool_call.name}({tool_call.arguments})")
            return True
        else:
            print("⚠️  No tool calls found")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_with_system_prompt():
    """Test if we can influence tool usage through system prompts"""

    print("\n🔧 Testing with system prompt to encourage tool usage...\n")

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

            # Get current system prompt
            agent_details = await client.agents.retrieve(agent.id)
            print(f"✅ Current system prompt: {agent_details.system[:100]}...")

            # Try to modify system prompt to include tool awareness
            # Note: This might not work if we can't modify the agent
            print("⚠️  System prompt modification may not be allowed")

            # Test with a direct message encouraging tool usage
            response = await client.agents.messages.create(
                agent_id=agent.id,
                messages=[{
                    "role": "user",
                    "content": "Please use the available tools to calculate 15 × 23. Show me the tool call."
                }]
            )

            print("📊 Response:")
            for message in response.messages:
                if hasattr(message, 'content') and message.content:
                    print(f"   {message.role}: {message.content}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

if __name__ == "__main__":
    print("Testing dynamic tool definition with Letta SDK...")

    # Test direct SDK usage
    success1 = asyncio.run(test_dynamic_tools_directly())

    # Test with system prompts
    success2 = asyncio.run(test_with_system_prompt())

    print("\n📋 RESULTS:")
    print(f"   Direct SDK test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   System prompt test: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1:
        print("\n🎉 Dynamic tool definition works!")
    else:
        print("\n💥 Dynamic tool definition needs more investigation.")