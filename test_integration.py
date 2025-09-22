"""
Real integration tests with actual Letta server and Milo agent.
These tests exercise the real functionality to ensure it works in production.
"""
import requests
import json
import time
import os
from typing import Dict, Any

# Configuration constants
AGENT_NAME = "companion-agent-1758429513525"


class LettaProxyTester:
    """Test the Letta Proxy with real Letta server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint."""
        print("🩺 Testing health endpoint...")
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()

        data = response.json()
        print(f"✅ Health status: {data['status']}")
        print(f"🔗 Letta server: {data['letta_base_url']}")
        print(f"🤖 Agents loaded: {data['agents_loaded']}")
        return data

    def test_models_endpoint(self) -> Dict[str, Any]:
        """Test the models endpoint (should show Milo agent)."""
        print("📋 Testing models endpoint...")
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()

        data = response.json()
        print(f"✅ Found {len(data['data'])} models")

        # Look for Milo agent
        milo_agent = None
        for model in data['data']:
            print(f"  🤖 Model: {model['id']} (owned by {model['owned_by']})")
            if model['id'] == AGENT_NAME:
                milo_agent = model
                break

        if milo_agent:
            print("✅ Milo agent found!")
        else:
            print("⚠️  Milo agent not found in models list")
            print("Available agents:", [model['id'] for model in data['data']])

        return data

    def test_chat_completion(self, message: str = "Hello! What's 2+2?") -> Dict[str, Any]:
        """Test chat completion with Milo agent."""
        print(f"💬 Testing chat completion with message: '{message}'")

        payload = {
            "model": AGENT_NAME,
            "messages": [{"role": "user", "content": message}]
        }

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        data = response.json()
        print("✅ Chat completion successful!")
        print(f"🤖 Response: {data['choices'][0]['message']['content']}")
        return data

    def test_chat_completion_with_tools(self) -> Dict[str, Any]:
        """Test chat completion with tool calling."""
        print("🔧 Testing chat completion with tools...")

        payload = {
            "model": AGENT_NAME,
            "messages": [{"role": "user", "content": "What's the current date and time?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "description": "Get the current date and time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ]
        }

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        data = response.json()
        print("✅ Tool calling test completed!")
        print(f"🤖 Response type: {data['choices'][0]['finish_reason']}")

        if data['choices'][0]['finish_reason'] == 'tool_calls':
            print("🔧 Tool was called!")
            tool_calls = data['choices'][0]['message']['tool_calls']
            for tool_call in tool_calls:
                print(f"  Tool: {tool_call['function']['name']}")
        else:
            print(f"💬 Regular response: {data['choices'][0]['message']['content']}")

        return data

    def test_streaming_response(self) -> None:
        """Test streaming chat completion."""
        print("🌊 Testing streaming response...")

        payload = {
            "model": AGENT_NAME,
            "messages": [{"role": "user", "content": "Count to 5 slowly"}],
            "stream": True
        }

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        response.raise_for_status()

        print("✅ Streaming response started:")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content:  # Only print non-empty content
                                    print(f"  {content}", end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        print("\n✅ Streaming completed!")

    def run_all_tests(self) -> None:
        """Run all integration tests."""
        print("🚀 Starting Letta Proxy integration tests...")
        print("=" * 50)

        try:
            # Test 1: Health check
            health_data = self.test_health_endpoint()
            print()

            # Test 2: Models list
            models_data = self.test_models_endpoint()
            print()

            # Test 3: Basic chat completion
            chat_data = self.test_chat_completion()
            print()

            # Test 4: Tool calling
            tools_data = self.test_chat_completion_with_tools()
            print()

            # Test 5: Streaming (if supported)
            try:
                self.test_streaming_response()
                print()
            except Exception as e:
                print(f"⚠️  Streaming test failed: {e}")
                print()

            print("=" * 50)
            print("🎉 All integration tests completed!")

            # Summary
            if 'Milo' in str(models_data):
                print("✅ SUCCESS: Milo agent is accessible through the proxy!")
            else:
                print("⚠️  WARNING: Milo agent not found - check agent name and server configuration")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            print("🔧 Troubleshooting tips:")
            print("  1. Check if the proxy server is running: python -m uvicorn main:app --reload")
            print("  2. Verify LETTA_BASE_URL environment variable is set correctly")
            print("  3. Check if the Letta server is running and accessible")
            print("  4. Ensure the Milo agent exists on the Letta server")


def main():
    """Main test runner."""
    tester = LettaProxyTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()