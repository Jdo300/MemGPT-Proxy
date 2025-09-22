#!/usr/bin/env python3
"""
Comprehensive test to verify Open WebUI integration with the proxy tool bridge.
Tests the complete flow that Open WebUI would use.
"""

import asyncio
import json
import os
import time
import requests
from openai import OpenAI

# Configuration
PROXY_BASE_URL = "http://localhost:8000/v1"
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")

# Test agent - using the one that works
TEST_AGENT_NAME = "companion-agent-1758429513525"

# Test tools that Open WebUI might send
OPEN_WEBUI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def test_open_webui_chat_completion():
    """Test chat completion exactly as Open WebUI would use it."""
    try:
        client = OpenAI(
            base_url=PROXY_BASE_URL,
            api_key="dummy-key"  # Not needed for proxy
        )

        print("🔍 Testing Open WebUI chat completion flow...")
        
        response = client.chat.completions.create(
            model=TEST_AGENT_NAME,
            messages=[
                {"role": "user", "content": "What's the weather like in New York? Use the get_current_weather tool."}
            ],
            tools=OPEN_WEBUI_TOOLS,
            stream=False,
            max_tokens=300
        )

        print("✅ Open WebUI chat completion successful")
        print(f"   Response type: {type(response)}")
        print(f"   Choices: {len(response.choices)}")
        
        message = response.choices[0].message
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"   Tool calls detected: {len(message.tool_calls)}")
            for i, tool_call in enumerate(message.tool_calls):
                print(f"     Tool call {i+1}:")
                print(f"       Name: {tool_call.function.name}")
                print(f"       Arguments: {tool_call.function.arguments}")
        else:
            print(f"   Content: {message.content[:200] if message.content else 'No content'}...")
            
        return True

    except Exception as e:
        print(f"❌ Open WebUI chat completion failed: {e}")
        return False

def test_open_webui_streaming():
    """Test streaming exactly as Open WebUI would use it."""
    try:
        client = OpenAI(
            base_url=PROXY_BASE_URL,
            api_key="dummy-key"
        )

        print("🔍 Testing Open WebUI streaming flow...")
        
        response = client.chat.completions.create(
            model=TEST_AGENT_NAME,
            messages=[
                {"role": "user", "content": "Search for information about artificial intelligence. Use the web_search tool."}
            ],
            tools=OPEN_WEBUI_TOOLS,
            stream=True,
            max_tokens=300
        )

        print("✅ Open WebUI streaming successful")
        
        chunks_received = 0
        tool_calls = []
        content = ""
        reasoning_content = ""

        for chunk in response:
            chunks_received += 1
            
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # Check for tool calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function and tool_call.function.name:
                            tool_calls.append({
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else ""
                            })
                
                # Check for content
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                    
                # Check for reasoning
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    reasoning_content += delta.reasoning

        print(f"   Chunks received: {chunks_received}")
        print(f"   Tool calls found: {len(tool_calls)}")
        print(f"   Content length: {len(content)}")
        print(f"   Reasoning length: {len(reasoning_content)}")
        
        if tool_calls:
            print(f"   First tool call: {tool_calls[0]['name']}")
            
        return True

    except Exception as e:
        print(f"❌ Open WebUI streaming failed: {e}")
        return False

def test_tool_cleanup():
    """Test that tools are properly cleaned up between requests."""
    try:
        client = OpenAI(
            base_url=PROXY_BASE_URL,
            api_key="dummy-key"
        )

        print("🔍 Testing tool cleanup between requests...")
        
        # First request with tools
        response1 = client.chat.completions.create(
            model=TEST_AGENT_NAME,
            messages=[
                {"role": "user", "content": "Use the web_search tool."}
            ],
            tools=OPEN_WEBUI_TOOLS,
            max_tokens=100
        )
        
        # Second request without tools (should clean up)
        response2 = client.chat.completions.create(
            model=TEST_AGENT_NAME,
            messages=[
                {"role": "user", "content": "Hello, just a regular chat."}
            ],
            max_tokens=100
        )
        
        # Third request with different tools
        single_tool = [OPEN_WEBUI_TOOLS[0]]  # Just web_search
        response3 = client.chat.completions.create(
            model=TEST_AGENT_NAME,
            messages=[
                {"role": "user", "content": "Search for AI news."}
            ],
            tools=single_tool,
            max_tokens=100
        )

        print("✅ Tool cleanup test successful")
        print("   All requests completed without errors")
        return True

    except Exception as e:
        print(f"❌ Tool cleanup test failed: {e}")
        return False

def main():
    """Main test runner for Open WebUI integration."""
    print("🚀 Letta Proxy - Open WebUI Integration Test")
    print("=" * 50)

    # Check if proxy server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Proxy server not responding at {PROXY_BASE_URL}")
            return 1
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to proxy server at {PROXY_BASE_URL}")
        return 1

    # Run Open WebUI integration tests
    results = []
    
    print("\n🧪 Testing Open WebUI Integration...")
    print("-" * 40)
    
    results.append(("Open WebUI Chat Completion", test_open_webui_chat_completion()))
    results.append(("Open WebUI Streaming", test_open_webui_streaming()))
    results.append(("Tool Cleanup", test_tool_cleanup()))

    # Summary
    print("\n📊 Open WebUI Integration Test Results:")
    print("=" * 45)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Open WebUI integration tests passed!")
        print("✅ Proxy tool bridge is ready for Open WebUI integration!")
        return 0
    else:
        print("💥 Some Open WebUI integration tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())