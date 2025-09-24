"""
Tests specifically for Roo Code compatibility scenarios.
These tests demonstrate that the key issue has been resolved.
"""
import pytest
import json
from main import _prepare_messages


@pytest.mark.asyncio
async def test_roo_code_initial_request_compatibility():
    """
    Test that demonstrates the core issue is fixed.
    
    BEFORE: This would fail because:
    1. Only the last message was processed (ignoring system message)
    2. System messages caused HTTPException
    
    AFTER: This now works because:
    1. All messages are processed in order
    2. System messages are converted to user messages with prefix
    """
    # This is the exact pattern Roo Code sends on initial request
    roo_messages = [
        {
            "role": "system",
            "content": "You are Roo Code, an AI programming assistant. You help users write, debug, and improve code."
        },
        {
            "role": "user", 
            "content": "hello"
        }
    ]
    
    # This should NOT raise an exception (it would have before the fix)
    result = await _prepare_messages(roo_messages)
    
    # Verify we get both messages processed
    assert len(result) == 2
    
    # Verify system message is converted properly
    system_msg = result[0]
    assert system_msg.role == "user"
    assert "[System]:" in system_msg.content[0].text
    assert "Roo Code" in system_msg.content[0].text
    assert "programming assistant" in system_msg.content[0].text
    
    # Verify user message passes through unchanged
    user_msg = result[1]
    assert user_msg.role == "user"
    assert user_msg.content[0].text == "hello"


@pytest.mark.asyncio 
async def test_old_behavior_vs_new_behavior():
    """
    Demonstrate the difference between old behavior (broken) and new behavior (fixed).
    """
    # Multi-turn conversation that would have failed before
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "What's 2+2?"}
    ]
    
    # OLD BEHAVIOR (simulated - would have failed):
    # 1. Only processes messages[-1] = {"role": "user", "content": "What's 2+2?"}
    # 2. System message and conversation history would be ignored
    # 3. Assistant message would cause HTTPException if it was the last message
    
    # NEW BEHAVIOR (actual - now works):
    result = await _prepare_messages(messages)
    
    # All 4 messages should be processed
    assert len(result) == 4
    
    # System message converted
    assert result[0].role == "user"
    assert "[System]:" in result[0].content[0].text
    
    # First user message preserved
    assert result[1].role == "user"
    assert result[1].content[0].text == "Hi"
    
    # Assistant message converted to preserve conversation history
    assert result[2].role == "user"
    assert "[Assistant Previous]:" in result[2].content[0].text
    assert "Hello! How can I help?" in result[2].content[0].text
    
    # Final user message preserved
    assert result[3].role == "user"
    assert result[3].content[0].text == "What's 2+2?"


@pytest.mark.asyncio
async def test_edge_cases_that_would_break_old_system():
    """
    Test edge cases that would have broken the old system but work with the new one.
    """
    
    # Case 1: Conversation ending with assistant message (would have failed before)
    case1 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    result1 = await _prepare_messages(case1)
    assert len(result1) == 2
    assert "[Assistant Previous]:" in result1[1].content[0].text
    
    # Case 2: Only system message (would have failed before)
    case2 = [
        {"role": "system", "content": "You are an expert coder"}
    ]
    
    result2 = await _prepare_messages(case2)
    assert len(result2) == 1
    assert "[System]:" in result2[0].content[0].text
    
    # Case 3: Mixed roles in complex order
    case3 = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "Previous response"},
        {"role": "tool", "content": "Tool output", "tool_call_id": "call_123"},
        {"role": "user", "content": "New question"}
    ]
    
    result3 = await _prepare_messages(case3)
    assert len(result3) == 4
    assert "[System]:" in result3[0].content[0].text
    assert "[Assistant Previous]:" in result3[1].content[0].text
    assert "[Tool Result call_123]:" in result3[2].content[0].text
    assert result3[3].content[0].text == "New question"


@pytest.mark.asyncio
async def test_maintains_message_order():
    """
    Verify that message order is preserved, which is critical for conversation context.
    """
    messages = [
        {"role": "system", "content": "First"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "Third"}, 
        {"role": "user", "content": "Fourth"}
    ]
    
    result = await _prepare_messages(messages)
    
    # Order should be maintained
    assert "[System]: First" in result[0].content[0].text
    assert "Second" == result[1].content[0].text
    assert "[Assistant Previous]: Third" in result[2].content[0].text
    assert "Fourth" == result[3].content[0].text


@pytest.mark.asyncio
async def test_error_resilience():
    """
    Test that the system gracefully handles malformed or edge case inputs.
    """
    
    # Empty content
    result1 = await _prepare_messages([{"role": "user", "content": ""}])
    assert len(result1) == 1
    assert result1[0].content[0].text == ""
    
    # Missing content field
    result2 = await _prepare_messages([{"role": "user"}])
    assert len(result2) == 1
    assert result2[0].content[0].text == ""
    
    # Unknown role (should not crash, should log warning)
    result3 = await _prepare_messages([{"role": "mystery_role", "content": "test"}])
    assert len(result3) == 1
    assert "[Unknown Role mystery_role]:" in result3[0].content[0].text