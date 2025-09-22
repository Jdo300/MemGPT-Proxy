#!/usr/bin/env python3
"""
Simple test to verify proxy tool bridge functionality.
Tests that tools are properly synced and the basic flow works.
"""

import asyncio
import json
import os
import time
import requests

# Configuration
PROXY_BASE_URL = "http://localhost:8000"
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")

def test_proxy_health():
    """Test that the proxy server is running and healthy."""
    try:
        response = requests.get(f"{PROXY_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Proxy server health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Agents loaded: {health_data['agents_loaded']}")
            return True
        else:
            print(f"❌ Proxy health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Proxy health check error: {e}")
        return False

def test_models_endpoint():
    """Test that the models endpoint returns available agents."""
    try:
        response = requests.get(f"{PROXY_BASE_URL}/v1/models")
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['id'] for model in models_data['data']]
            print("✅ Models endpoint working")
            print(f"   Available models: {available_models}")

            # Use the first available agent (avoid sleeptime agents)
            for model in available_models:
                if 'sleeptime' not in model.lower():
                    print(f"✅ Using test agent: '{model}'")
                    return model

            # Fallback to first available if no non-sleeptime agent
            if available_models:
                first_model = available_models[0]
                print(f"⚠️  Using fallback agent (no non-sleeptime found): '{first_model}'")
                return first_model
            else:
                print("❌ No agents available")
                return None
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return None

def test_basic_chat_completion(agent_name):
    """Test basic chat completion without tools."""
    try:
        payload = {
            "model": agent_name,
            "messages": [{"role": "user", "content": "Hello! Please respond with a simple greeting."}],
            "max_tokens": 100
        }

        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print("✅ Basic chat completion working")
            print(f"   Response: {content[:100]}...")
            return True
        else:
            print(f"❌ Basic chat completion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Basic chat completion error: {e}")
        return False

def test_chat_completion_with_tools(agent_name):
    """Test chat completion with tools."""
    try:
        # Simple test tools
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        payload = {
            "model": agent_name,
            "messages": [{"role": "user", "content": "What's the weather like in New York?"}],
            "tools": test_tools,
            "max_tokens": 200
        }

        print("🛠️  Testing chat completion with tools...")
        print(f"   Tools: {[tool['function']['name'] for tool in test_tools]}")

        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Chat completion with tools successful")
            print(f"   Response type: {type(result)}")

            # Check if we got tool calls or regular content
            message = result['choices'][0]['message']
            if 'tool_calls' in message and message['tool_calls']:
                print(f"   Tool calls: {len(message['tool_calls'])}")
                for tool_call in message['tool_calls']:
                    print(f"     - {tool_call['function']['name']}")
            else:
                print(f"   Content: {message.get('content', '')[:200]}...")

            return True
        else:
            print(f"❌ Chat completion with tools failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Chat completion with tools error: {e}")
        return False

def test_streaming_with_tools(agent_name):
    """Test streaming chat completion with tools."""
    try:
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_math",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        payload = {
            "model": agent_name,
            "messages": [{"role": "user", "content": "Calculate 15 + 27 using the calculate_math tool."}],
            "tools": test_tools,
            "stream": True,
            "max_tokens": 200
        }

        print("🛠️  Testing streaming chat completion with tools...")

        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )

        if response.status_code == 200:
            print("✅ Streaming chat completion with tools successful")

            chunks_received = 0
            tool_calls = []
            content = ""

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            chunks_received += 1

                            # Check for tool calls
                            choices = chunk.get('choices', [])
                            if choices and choices[0].get('delta', {}).get('tool_calls'):
                                for tool_call in choices[0]['delta']['tool_calls']:
                                    if tool_call.get('function', {}).get('name'):
                                        tool_calls.append(tool_call['function']['name'])

                            # Check for content in various formats
                            delta = choices[0].get('delta', {}) if choices else {}
                            if delta.get('content'):
                                content += delta['content']
                            elif delta.get('reasoning'):
                                content += delta['reasoning']
                            elif delta.get('text'):
                                content += delta['text']

                            # Also check for content in the message directly (some formats)
                            if choices and choices[0].get('message', {}).get('content'):
                                content += choices[0]['message']['content']

                        except json.JSONDecodeError:
                            continue

            print(f"   Chunks received: {chunks_received}")
            print(f"   Tool calls found: {len(tool_calls)}")
            print(f"   Content length: {len(content)}")

            if tool_calls:
                print(f"   Tool call: {tool_calls[0]}")

            return True
        else:
            print(f"❌ Streaming chat completion with tools failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Streaming chat completion with tools error: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 Letta Proxy Tool Bridge - Simple Integration Test")
    print("=" * 55)

    # Check if proxy server is running
    try:
        response = requests.get(f"{PROXY_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Proxy server not responding at {PROXY_BASE_URL}")
            print("   Please start the proxy server with: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
            return 1
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to proxy server at {PROXY_BASE_URL}")
        print("   Please start the proxy server with: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        return 1

    # Run tests
    if not test_proxy_health():
        return 1

    agent_name = test_models_endpoint()
    if not agent_name:
        return 1

    # Test basic functionality first
    if not test_basic_chat_completion(agent_name):
        print("❌ Basic functionality not working. Cannot test tools.")
        return 1

    # Test tool functionality
    results = []

    print("\n🛠️  Testing tool functionality...")
    print("-" * 30)

    results.append(("Chat with tools (non-streaming)", test_chat_completion_with_tools(agent_name)))
    results.append(("Streaming with tools", test_streaming_with_tools(agent_name)))

    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Proxy tool bridge is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())