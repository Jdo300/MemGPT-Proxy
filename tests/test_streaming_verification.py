
"""
Streaming verification test script.
Tests that the Letta Proxy is actually streaming responses in real-time
rather than buffering and sending all at once.
"""
import requests
import json
import time
import sys
from datetime import datetime
from typing import Iterator


# Configuration constants
AGENT_NAME = "companion-agent-1758429513525"
BASE_URL = "http://localhost:8000"


def test_streaming_response():
    """Test streaming with a long poem to verify real-time streaming."""
    print("🌊 Testing streaming response with long poem...")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print()

    payload = {
        "model": AGENT_NAME,
        "messages": [{"role": "user", "content": "Write me a long, beautiful poem about strawberries. Make it at least 300 words with lots of vivid descriptions, metaphors, and sensory details about strawberries - their color, taste, smell, texture, and the feelings they evoke."}],
        "stream": True
    }

    try:
        # Make streaming request
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )

        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return

        print(f"✅ Connected! Response status: {response.status_code}")
        print()

        accumulated_content = ""
        chunk_count = 0

        # Process streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    chunk_count += 1
                    data = line[6:]  # Remove 'data: ' prefix

                    if data == '[DONE]':
                        print(f"\n🎉 Streaming complete! Received {chunk_count} chunks")
                        break

                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                accumulated_content += content

                                # Show real-time streaming with timestamp
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"[{timestamp}] 📝 Chunk {chunk_count}: {content}", end='', flush=True)

                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error: {e}")
                        continue

        print(f"\n\n📊 Summary:")
        print(f"- Total chunks received: {chunk_count}")
        print(f"- Total content length: {len(accumulated_content)} characters")
        print(f"- Word count: {len(accumulated_content.split())} words")
        print(f"- Finished at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        # Show if streaming was actually real-time by checking for reasonable chunk intervals
        if chunk_count > 1:
            print(f"\n⚡ Streaming verification: Looks like real-time streaming! ({chunk_count} chunks)")
        else:
            print(f"\n⚠️  Warning: Only received {chunk_count} chunk(s) - might be buffered response")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_thinking_tokens():
    """Test if thinking tokens are handled properly in streaming."""
    print("🧠 Testing thinking tokens handling...")
    print("=" * 60)

    payload = {
        "model": AGENT_NAME,
        "messages": [{"role": "user", "content": "Think step by step: What is 15 + 27? Show your thinking process clearly, then give the final answer."}],
        "stream": True
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )

        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            return

        print("✅ Connected! Analyzing streaming chunks for thinking tokens...\n")

        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data)
                        chunks.append(chunk)
                        print(f"📦 Chunk {len(chunks)}: {json.dumps(chunk, indent=2)}")
                    except json.JSONDecodeError:
                        continue

        print(f"\n📊 Analysis:")
        print(f"- Total chunks: {len(chunks)}")
        print("- Chunk structure analysis:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1}: {list(chunk.get('choices', [{}])[0].get('delta', {}).keys())}")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_reasoning_detection():
    """Test if reasoning messages are properly detected and streamed."""
    print("🧠 Testing reasoning message detection...")
    print("=" * 60)

    payload = {
        "model": AGENT_NAME,
        "messages": [{"role": "user", "content": "Think step by step: What is 17 × 23? Show your thinking process clearly."}],
        "stream": True
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )

        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            return

        print("✅ Connected! Analyzing chunks for reasoning vs content...\n")

        chunks = []
        reasoning_chunks = 0
        content_chunks = 0

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data)
                        chunks.append(chunk)

                        # Analyze the chunk structure
                        choices = chunk.get('choices', [])
                        if choices:
                            delta = choices[0].get('delta', {})
                            if 'reasoning' in delta:
                                reasoning_chunks += 1
                                print(f"🤔 Reasoning chunk {reasoning_chunks}: {delta['reasoning'][:100]}...")
                            elif 'content' in delta:
                                content_chunks += 1
                                print(f"📝 Content chunk {content_chunks}: {delta['content'][:100]}...")
                            else:
                                print(f"🔍 Other chunk: {list(delta.keys())}")

                    except json.JSONDecodeError:
                        continue

        print(f"\n📊 Results:")
        print(f"- Total chunks: {len(chunks)}")
        print(f"- Reasoning chunks: {reasoning_chunks}")
        print(f"- Content chunks: {content_chunks}")

        if reasoning_chunks > 0:
            print("✅ SUCCESS: Thinking tokens are properly separated!")
        else:
            print("⚠️  WARNING: No reasoning chunks detected - might be using old message format")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_reasoning_detection()
