#!/usr/bin/env python3
"""
Smoke test script to reproduce Roo Code initial request issue and verify fix.
This script sends the exact patterns Roo Code would send to test the proxy.
"""
import json
import requests
import sys
import os


PROXY_BASE_URL = "http://localhost:8000"
TEST_AGENT = "Milo"


def load_fixture(filename: str) -> dict:
    """Load test fixture from JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_path = os.path.join(script_dir, "..", "tests", "fixtures", filename)
    with open(fixture_path, "r") as f:
        return json.load(f)


def test_models_endpoint():
    """Test that models endpoint works and includes Milo"""
    print("🔍 Testing /v1/models endpoint...")
    try:
        response = requests.get(f"{PROXY_BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        model_ids = [model["id"] for model in data.get("data", [])]
        
        print(f"✅ Models endpoint working. Found agents: {model_ids}")
        
        if TEST_AGENT in model_ids:
            print(f"✅ Target agent '{TEST_AGENT}' is available")
            return True
        else:
            print(f"❌ Target agent '{TEST_AGENT}' not found")
            return False
            
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False


def test_roo_initial_non_stream():
    """Test Roo's initial request pattern (non-streaming)"""
    print("\n🔍 Testing Roo initial request (non-streaming)...")
    
    try:
        payload = load_fixture("roo_initial.json")
        print(f"📤 Sending payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        data = response.json()
        print("✅ Non-streaming request successful!")
        
        # Validate response structure
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            print(f"🤖 Assistant response: {content[:200]}...")
            
            if len(content) > 10:
                print("✅ Received meaningful response content")
                return True
            else:
                print("❌ Response content too short")
                return False
        else:
            print("❌ Invalid response structure")
            return False
            
    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")
        return False


def test_roo_initial_streaming():
    """Test Roo's initial request pattern (streaming)"""
    print("\n🔍 Testing Roo initial request (streaming)...")
    
    try:
        payload = load_fixture("roo_initial_stream.json")
        print(f"📤 Sending streaming payload")
        
        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Streaming request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        print("✅ Streaming request started successfully!")
        
        # Parse streaming response
        chunks = []
        full_content = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_part = line[6:]
                    if data_part == "[DONE]":
                        print("✅ Received [DONE] terminator")
                        break
                    try:
                        chunk = json.loads(data_part)
                        chunks.append(chunk)
                        
                        # Extract content from delta
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]
                                
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ Received {len(chunks)} streaming chunks")
        
        if len(full_content) > 10:
            print(f"🤖 Streamed content: {full_content[:200]}...")
            print("✅ Streaming completed successfully with meaningful content")
            return True
        else:
            print("❌ Streaming content too short or empty")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


def test_multiturn_conversation():
    """Test multi-turn conversation"""
    print("\n🔍 Testing multi-turn conversation...")
    
    try:
        payload = load_fixture("roo_multiturn.json")  
        print(f"📤 Sending multi-turn payload with {len(payload['messages'])} messages")
        
        # Log the message roles for debugging
        roles = [msg["role"] for msg in payload["messages"]]
        print(f"Message roles: {roles}")
        
        response = requests.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Multi-turn request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        data = response.json()
        print("✅ Multi-turn request successful!")
        
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            print(f"🤖 Multi-turn response: {content[:200]}...")
            
            if len(content) > 10:
                print("✅ Multi-turn conversation working")
                return True
            else:
                print("❌ Multi-turn response too short")
                return False
        else:
            print("❌ Invalid multi-turn response structure")
            return False
            
    except Exception as e:
        print(f"❌ Multi-turn test failed: {e}")
        return False


def main():
    """Run all smoke tests"""
    print("🚀 Roo Code Smoke Test Suite")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{PROXY_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Proxy server not running or not healthy")
            print(f"Start the proxy with: uvicorn main:app --host 0.0.0.0 --port 8000")
            sys.exit(1)
        print("✅ Proxy server is running")
    except Exception as e:
        print(f"❌ Cannot connect to proxy server: {e}")
        print(f"Start the proxy with: uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run all tests
    tests = [
        ("Models Endpoint", test_models_endpoint),
        ("Roo Initial (Non-Stream)", test_roo_initial_non_stream),  
        ("Roo Initial (Streaming)", test_roo_initial_streaming),
        ("Multi-turn Conversation", test_multiturn_conversation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Roo Code compatibility verified.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()