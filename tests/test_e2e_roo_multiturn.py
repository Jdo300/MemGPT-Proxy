"""
End-to-end tests for multi-turn conversations via Roo Code patterns against LIVE Letta server.
These tests require LETTA_E2E=1 environment variable to run.
"""
import os
import json
import pytest
import httpx
from typing import Dict, Any, List


# Skip all tests unless LETTA_E2E=1 is set
pytestmark = pytest.mark.skipif(
    os.getenv("LETTA_E2E") != "1",
    reason="LETTA_E2E environment variable not set to 1. Set LETTA_E2E=1 to run live server tests."
)


PROXY_BASE_URL = "http://localhost:8000"
TEST_AGENT = "Milo"


class TestRooMultiTurnE2E:
    """End-to-end tests for Roo Code multi-turn conversation patterns"""

    @pytest.fixture
    def httpx_client(self):
        """HTTP client for making requests"""
        return httpx.Client(timeout=30.0)

    def send_chat_request(self, httpx_client, messages: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
        """Send a chat completion request and return the response"""
        payload = {
            "model": TEST_AGENT,
            "messages": messages,
            "stream": stream
        }
        
        if stream:
            with httpx_client.stream(
                "POST",
                f"{PROXY_BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status_code == 200, f"Streaming request failed: {response.status_code}"
                
                full_content = ""
                for line in response.iter_lines():
                    if line.strip():
                        if line.startswith("data: "):
                            data_part = line[6:]
                            if data_part == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_part)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                if "content" in delta:
                                    full_content += delta["content"]
                            except json.JSONDecodeError:
                                continue
                
                # Return a mock response structure with the streamed content
                return {
                    "choices": [{"message": {"role": "assistant", "content": full_content}}]
                }
        else:
            response = httpx_client.post(
                f"{PROXY_BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 200, f"Request failed: {response.status_code} {response.text}"
            return response.json()

    def test_three_turn_conversation_non_stream(self, httpx_client):
        """Test a 3-turn conversation (non-streaming)"""
        # Turn 1: Initial greeting
        messages = [
            {"role": "system", "content": "You are Milo, a helpful AI assistant for coding tasks."},
            {"role": "user", "content": "Hello! I need help with Python programming."}
        ]
        
        response1 = self.send_chat_request(httpx_client, messages)
        assistant_response1 = response1["choices"][0]["message"]["content"]
        
        assert len(assistant_response1) > 10, "First response too short"
        assert "python" in assistant_response1.lower() or "help" in assistant_response1.lower()
        
        # Turn 2: Ask about a specific topic, including conversation history
        messages.extend([
            {"role": "assistant", "content": assistant_response1},
            {"role": "user", "content": "Can you show me how to create a dictionary in Python?"}
        ])
        
        response2 = self.send_chat_request(httpx_client, messages)
        assistant_response2 = response2["choices"][0]["message"]["content"]
        
        assert len(assistant_response2) > 10, "Second response too short"
        assert "dict" in assistant_response2.lower() or "{" in assistant_response2
        
        # Turn 3: Follow up with reference to previous response
        messages.extend([
            {"role": "assistant", "content": assistant_response2},
            {"role": "user", "content": "Thanks! Can you also show me how to iterate over the dictionary you just showed me?"}
        ])
        
        response3 = self.send_chat_request(httpx_client, messages)
        assistant_response3 = response3["choices"][0]["message"]["content"]
        
        assert len(assistant_response3) > 10, "Third response too short"
        assert "for" in assistant_response3.lower() or "iterate" in assistant_response3.lower()

    def test_three_turn_conversation_streaming(self, httpx_client):
        """Test a 3-turn conversation (streaming)"""
        # Turn 1: Initial coding question
        messages = [
            {"role": "system", "content": "You are Milo, a helpful AI assistant for coding tasks."},
            {"role": "user", "content": "Hi! How do I read a file in Python?"}
        ]
        
        response1 = self.send_chat_request(httpx_client, messages, stream=True)
        assistant_response1 = response1["choices"][0]["message"]["content"]
        
        assert len(assistant_response1) > 10, "First streaming response too short"
        assert "open" in assistant_response1.lower() or "file" in assistant_response1.lower()
        
        # Turn 2: Follow up about error handling
        messages.extend([
            {"role": "assistant", "content": assistant_response1},
            {"role": "user", "content": "What if the file doesn't exist? How do I handle errors?"}
        ])
        
        response2 = self.send_chat_request(httpx_client, messages, stream=True)
        assistant_response2 = response2["choices"][0]["message"]["content"]
        
        assert len(assistant_response2) > 10, "Second streaming response too short"
        assert "try" in assistant_response2.lower() or "except" in assistant_response2.lower() or "error" in assistant_response2.lower()
        
        # Turn 3: Ask for complete example
        messages.extend([
            {"role": "assistant", "content": assistant_response2},
            {"role": "user", "content": "Can you put it all together in a complete example?"}
        ])
        
        response3 = self.send_chat_request(httpx_client, messages, stream=True)
        assistant_response3 = response3["choices"][0]["message"]["content"]
        
        assert len(assistant_response3) > 20, "Third streaming response too short"

    def test_conversation_with_code_context(self, httpx_client):
        """Test conversation that builds up code context"""
        # Start with system message that provides coding context
        messages = [
            {
                "role": "system", 
                "content": "You are Milo, an AI coding assistant. You help write clean, efficient Python code."
            },
            {
                "role": "user", 
                "content": "I need to create a function that calculates the factorial of a number. Can you help?"
            }
        ]
        
        # First turn - get the basic function
        response1 = self.send_chat_request(httpx_client, messages)
        assistant_response1 = response1["choices"][0]["message"]["content"]
        
        assert "factorial" in assistant_response1.lower()
        assert ("def " in assistant_response1 or "function" in assistant_response1.lower())
        
        # Second turn - ask for optimization
        messages.extend([
            {"role": "assistant", "content": assistant_response1},
            {"role": "user", "content": "That's great! Can you make it more efficient using memoization?"}
        ])
        
        response2 = self.send_chat_request(httpx_client, messages)
        assistant_response2 = response2["choices"][0]["message"]["content"]
        
        assert len(assistant_response2) > 20
        assert ("memo" in assistant_response2.lower() or "cache" in assistant_response2.lower())
        
        # Third turn - ask for tests
        messages.extend([
            {"role": "assistant", "content": assistant_response2},
            {"role": "user", "content": "Perfect! Now can you write some test cases for this function?"}
        ])
        
        response3 = self.send_chat_request(httpx_client, messages)
        assistant_response3 = response3["choices"][0]["message"]["content"]
        
        assert len(assistant_response3) > 20
        assert ("test" in assistant_response3.lower() or "assert" in assistant_response3.lower())

    def test_error_recovery_in_conversation(self, httpx_client):
        """Test that conversations can recover from potentially problematic inputs"""
        # Start with a system message and user message
        messages = [
            {"role": "system", "content": "You are Milo, a helpful coding assistant."},
            {"role": "user", "content": "Hi there!"}
        ]
        
        response1 = self.send_chat_request(httpx_client, messages)
        assistant_response1 = response1["choices"][0]["message"]["content"]
        
        assert len(assistant_response1) > 5
        
        # Add a potentially tricky message with special characters
        messages.extend([
            {"role": "assistant", "content": assistant_response1},
            {"role": "user", "content": "Can you help me with this weird string: 'hello<world>\"test\"&amp;'?"}
        ])
        
        response2 = self.send_chat_request(httpx_client, messages)
        assistant_response2 = response2["choices"][0]["message"]["content"]
        
        # Should handle special characters gracefully
        assert len(assistant_response2) > 5