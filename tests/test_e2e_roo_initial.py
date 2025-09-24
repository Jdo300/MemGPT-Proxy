"""
End-to-end tests for Roo Code initial request handling against LIVE Letta server.
These tests require LETTA_E2E=1 environment variable to run.
"""
import os
import json
import pytest
import httpx
from typing import Dict, Any


# Skip all tests unless LETTA_E2E=1 is set
pytestmark = pytest.mark.skipif(
    os.getenv("LETTA_E2E") != "1",
    reason="LETTA_E2E environment variable not set to 1. Set LETTA_E2E=1 to run live server tests."
)


PROXY_BASE_URL = "http://localhost:8000"
TEST_AGENT = "Milo"


class TestRooInitialE2E:
    """End-to-end tests for Roo Code initial request patterns"""

    @pytest.fixture
    def httpx_client(self):
        """HTTP client for making requests"""
        return httpx.Client(timeout=30.0)

    def load_fixture(self, filename: str) -> Dict[str, Any]:
        """Load test fixture from JSON file"""
        with open(f"tests/fixtures/{filename}", "r") as f:
            return json.load(f)

    def test_models_endpoint_includes_milo(self, httpx_client):
        """Test that the models endpoint includes the Milo agent"""
        response = httpx_client.get(f"{PROXY_BASE_URL}/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        
        model_ids = [model["id"] for model in data["data"]]
        assert TEST_AGENT in model_ids, f"Agent '{TEST_AGENT}' not found in models: {model_ids}"

    def test_roo_initial_request_non_stream(self, httpx_client):
        """Test Roo Code's initial request pattern (non-streaming)"""
        payload = self.load_fixture("roo_initial.json")
        
        response = httpx_client.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200, f"Request failed: {response.status_code} {response.text}"
        
        data = response.json()
        
        # Validate OpenAI-compatible response structure
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert data["model"] == TEST_AGENT
        assert "choices" in data
        assert len(data["choices"]) == 1
        
        choice = data["choices"][0]
        assert "index" in choice
        assert choice["index"] == 0
        assert "message" in choice
        assert "finish_reason" in choice
        
        message = choice["message"]
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message
        assert len(message["content"]) > 0  # Should have some response content
        
        # Validate usage information
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_roo_initial_request_streaming(self, httpx_client):
        """Test Roo Code's initial request pattern (streaming)"""
        payload = self.load_fixture("roo_initial_stream.json")
        
        with httpx_client.stream(
            "POST",
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            assert response.status_code == 200, f"Streaming request failed: {response.status_code}"
            
            chunks = []
            for line in response.iter_lines():
                if line.strip():
                    if line.startswith("data: "):
                        data_part = line[6:]  # Remove "data: " prefix
                        if data_part == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_part)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
            
            assert len(chunks) > 0, "No streaming chunks received"
            
            # Validate first chunk (should be the role primer)
            first_chunk = chunks[0]
            assert "id" in first_chunk
            assert "object" in first_chunk
            assert first_chunk["object"] == "chat.completion.chunk"
            assert "choices" in first_chunk
            assert len(first_chunk["choices"]) == 1
            
            choice = first_chunk["choices"][0]
            assert "delta" in choice
            assert choice["delta"]["role"] == "assistant"
            
            # Validate we received content chunks
            content_chunks = [
                chunk for chunk in chunks 
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            ]
            assert len(content_chunks) > 0, "No content chunks received"
            
            # Validate final chunk has finish_reason
            final_chunks = [
                chunk for chunk in chunks 
                if chunk.get("choices", [{}])[0].get("finish_reason") is not None
            ]
            assert len(final_chunks) > 0, "No final chunk with finish_reason found"

    def test_roo_multiturn_conversation(self, httpx_client):
        """Test multi-turn conversation pattern"""
        payload = self.load_fixture("roo_multiturn.json")
        
        response = httpx_client.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200, f"Multi-turn request failed: {response.status_code} {response.text}"
        
        data = response.json()
        
        # Validate response structure
        assert "choices" in data
        assert len(data["choices"]) == 1
        
        choice = data["choices"][0]
        assert "message" in choice
        message = choice["message"]
        assert "content" in message
        
        # Should have meaningful response about Fibonacci
        content = message["content"].lower()
        assert len(content) > 20, "Response content too short for fibonacci question"

    def test_health_endpoint(self, httpx_client):
        """Test health endpoint shows connection to live server"""
        response = httpx_client.get(f"{PROXY_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["letta_connected"] is True
        assert data["agents_loaded"] > 0