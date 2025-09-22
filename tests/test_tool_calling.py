#!/usr/bin/env python3
"""
Tool calling test for Letta Proxy
Tests streaming tool calls with a calculator tool
"""
import json
import asyncio
import httpx
import time

async def test_tool_calling():
    """Test streaming tool calls with a calculator function"""

    async with httpx.AsyncClient() as client:
        print("🛠️ Testing tool calling with streaming...\n")

        # Define a calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        # Test data with tool prompt
        payload = {
            "model": "companion-agent-1758429513525",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 15 times 23? Please use the calculator tool to compute this."
                }
            ],
            "tools": tools,
            "stream": True
        }

        reasoning_chunks = []
        content_chunks = []
        tool_call_chunks = []
        total_chunks = 0
        tool_calls_found = []
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
                                print(f"🤔 Reasoning: {delta['reasoning']}", end="", flush=True)

                            elif "content" in delta:
                                content_chunks.append(delta["content"])
                                print(f"💬 Content: {delta['content']}", end="", flush=True)

                            elif "tool_calls" in delta:
                                tool_call_chunks.append(delta["tool_calls"])
                                print(f"🔧 Tool call: {delta['tool_calls']}")

                                # Collect tool call information
                                for tool_call in delta["tool_calls"]:
                                    if "function" in tool_call:
                                        tool_calls_found.append({
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"].get("arguments", "")
                                        })

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
        print("\n📊 TOOL CALLING TEST RESULTS:")
        print(f"   Total chunks received: {total_chunks}")
        print(f"   Reasoning chunks: {len(reasoning_chunks)}")
        print(f"   Content chunks: {len(content_chunks)}")
        print(f"   Tool call chunks: {len(tool_call_chunks)}")
        print(f"   Tool calls found: {len(tool_calls_found)}")
        print(f"   Total time: {end_time - start_time:.2f}s")

        # Verify results
        success = True

        # Check if we got tool calls
        if tool_calls_found:
            print("✅ SUCCESS: Tool calls were made!")
            for i, tool_call in enumerate(tool_calls_found):
                print(f"   Tool call {i+1}: {tool_call['name']}({tool_call['arguments']})")
        else:
            print("⚠️  WARNING: No tool calls were found")
            success = False

        # Check if reasoning worked
        if reasoning_chunks:
            print("✅ SUCCESS: Reasoning chunks received")
            reasoning_text = ''.join(reasoning_chunks)
            print(f"   Reasoning: {reasoning_text[:100]}{'...' if len(reasoning_text) > 100 else ''}")
        else:
            print("⚠️  WARNING: No reasoning chunks")

        # Check if content worked
        if content_chunks:
            print("✅ SUCCESS: Content chunks received")
            content_text = ''.join(content_chunks)
            print(f"   Content: {content_text[:100]}{'...' if len(content_text) > 100 else ''}")
        else:
            print("⚠️  WARNING: No content chunks")

        # Overall assessment
        if tool_calls_found and len(tool_calls_found) > 0:
            print("🎉 Tool calling test PASSED!")
            return True
        else:
            print("💥 Tool calling test FAILED!")
            return False

async def test_tool_calling_non_streaming():
    """Test tool calls in non-streaming mode for comparison"""

    async with httpx.AsyncClient() as client:
        print("\n🛠️ Testing tool calling (non-streaming)...\n")

        # Define a calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        # Test data with tool prompt
        payload = {
            "model": "companion-agent-1758429513525",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 15 times 23? Please use the calculator tool to compute this."
                }
            ],
            "tools": tools,
            "stream": False
        }

        try:
            response = await client.post("http://localhost:8000/v1/chat/completions", json=payload)
            result = response.json()

            print("📊 NON-STREAMING TOOL CALLING RESULTS:")
            print(f"   Response time: {response.elapsed.total_seconds():.2f}s")

            # Extract tool calls from response
            message = result.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            print(f"   Tool calls found: {len(tool_calls)}")

            if tool_calls:
                print("✅ SUCCESS: Tool calls found in non-streaming mode!")
                for i, tool_call in enumerate(tool_calls):
                    print(f"   Tool call {i+1}: {tool_call['function']['name']}({tool_call['function']['arguments']})")
                return True
            else:
                print("⚠️  WARNING: No tool calls found in non-streaming mode")
                return False

        except Exception as e:
            print(f"❌ Non-streaming request failed: {e}")
            return False

if __name__ == "__main__":
    print("Testing tool calling functionality...")

    # Test streaming tool calls
    streaming_success = asyncio.run(test_tool_calling())

    # Test non-streaming tool calls for comparison
    non_streaming_success = asyncio.run(test_tool_calling_non_streaming())

    print("\n📋 FINAL RESULTS:")
    print(f"   Streaming tool calls: {'✅ PASS' if streaming_success else '❌ FAIL'}")
    print(f"   Non-streaming tool calls: {'✅ PASS' if non_streaming_success else '❌ FAIL'}")

    if streaming_success:
        print("\n🎉 Tool calling is working perfectly in streaming mode!")
    else:
        print("\n💥 Tool calling needs debugging.")