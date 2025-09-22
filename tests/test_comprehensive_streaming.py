#!/usr/bin/env python3
"""
Comprehensive streaming test for Letta Proxy
Tests both reasoning/thinking AND content streaming together
"""
import json
import asyncio
import httpx
import time

async def test_comprehensive_streaming():
    """Test that both reasoning and content stream properly together"""

    async with httpx.AsyncClient() as client:
        print("🚀 Testing comprehensive streaming (reasoning + content)...\n")

        # Test data
        payload = {
            "model": "companion-agent-1758429513525",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain step by step how to solve 25 × 15 and provide the final answer."
                }
            ],
            "stream": True
        }

        reasoning_chunks = []
        content_chunks = []
        tool_call_chunks = []
        total_chunks = 0
        start_time = time.time()

        try:
            async with client.stream("POST", "http://localhost:8000/v1/chat/completions", json=payload) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            total_chunks += 1

                            # Check what type of chunk this is
                            delta = chunk.get("choices", [{}])[0].get("delta", {})

                            if "reasoning" in delta:
                                reasoning_chunks.append(delta["reasoning"])
                                print(f"🤔 Reasoning chunk: {delta['reasoning']}", end="", flush=True)

                            elif "content" in delta:
                                content_chunks.append(delta["content"])
                                print(f"💬 Content chunk: {delta['content']}", end="", flush=True)

                            elif "tool_calls" in delta:
                                tool_call_chunks.append(delta["tool_calls"])
                                print(f"🔧 Tool call chunk: {delta['tool_calls']}")

                            elif chunk.get("choices", [{}])[0].get("finish_reason"):
                                print(f"\n🏁 Finished: {chunk['choices'][0]['finish_reason']}")

                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse chunk: {data}")
                            print(f"Error: {e}")

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

        end_time = time.time()

        # Print results
        print("\n📊 COMPREHENSIVE STREAMING TEST RESULTS:")
        print(f"   Total chunks received: {total_chunks}")
        print(f"   Reasoning chunks: {len(reasoning_chunks)}")
        print(f"   Content chunks: {len(content_chunks)}")
        print(f"   Tool call chunks: {len(tool_call_chunks)}")
        print(f"   Total time: {end_time - start_time:.2f}s")

        # Verify we got both reasoning and content
        if reasoning_chunks and content_chunks:
            print("✅ SUCCESS: Both reasoning and content streamed properly!")
            print(f"   Reasoning content: {''.join(reasoning_chunks)}")
            print(f"   Final content: {''.join(content_chunks)}")
            return True
        elif reasoning_chunks:
            print("⚠️  PARTIAL: Only reasoning chunks received")
            print(f"   Reasoning content: {''.join(reasoning_chunks)}")
            return False
        elif content_chunks:
            print("⚠️  PARTIAL: Only content chunks received")
            print(f"   Content: {''.join(content_chunks)}")
            return False
        else:
            print("❌ FAILURE: No content received")
            return False

if __name__ == "__main__":
    print("Testing comprehensive streaming with both reasoning and content...")
    success = asyncio.run(test_comprehensive_streaming())

    if success:
        print("\n🎉 All tests passed! Both reasoning and content stream properly.")
    else:
        print("\n💥 Some tests failed. Check the implementation.")