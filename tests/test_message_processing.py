"""
Tests for message processing functionality
"""
import pytest
import json
from main import _prepare_messages
from letta_client import MessageCreate
from letta_client.types import TextContent


@pytest.mark.asyncio
async def test_prepare_messages_user_only():
    """Test basic user message processing"""
    messages = [{"role": "user", "content": "Hello"}]
    result = await _prepare_messages(messages)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content[0].text == "Hello"


@pytest.mark.asyncio
async def test_prepare_messages_system_converted():
    """Test system message gets converted to user with prefix"""
    messages = [{"role": "system", "content": "You are helpful"}]
    result = await _prepare_messages(messages)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content[0].text == "[System]: You are helpful"


@pytest.mark.asyncio
async def test_prepare_messages_assistant_converted():
    """Test assistant message gets converted to user with prefix"""
    messages = [{"role": "assistant", "content": "I can help you"}]
    result = await _prepare_messages(messages)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content[0].text == "[Assistant Previous]: I can help you"


@pytest.mark.asyncio
async def test_prepare_messages_roo_initial_format():
    """Test processing Roo Code's typical initial request"""
    with open("tests/fixtures/roo_initial.json", "r") as f:
        roo_request = json.load(f)
    
    result = await _prepare_messages(roo_request["messages"])
    
    assert len(result) == 2
    # System message converted to user with prefix
    assert result[0].role == "user"
    assert "[System]:" in result[0].content[0].text
    assert "Roo Code" in result[0].content[0].text
    
    # User message preserved
    assert result[1].role == "user"
    assert result[1].content[0].text == "hello"


@pytest.mark.asyncio
async def test_prepare_messages_multiturn():
    """Test processing multi-turn conversation"""
    with open("tests/fixtures/roo_multiturn.json", "r") as f:
        multiturn_request = json.load(f)
    
    result = await _prepare_messages(multiturn_request["messages"])
    
    assert len(result) == 4
    
    # System message
    assert result[0].role == "user"
    assert "[System]:" in result[0].content[0].text
    
    # First user message
    assert result[1].role == "user"
    assert result[1].content[0].text == "Hello, I need help with Python."
    
    # Assistant response (converted to user with prefix)
    assert result[2].role == "user"
    assert "[Assistant Previous]:" in result[2].content[0].text
    
    # Second user message 
    assert result[3].role == "user"
    assert "fibonacci" in result[3].content[0].text


@pytest.mark.asyncio 
async def test_prepare_messages_tool_role():
    """Test tool message processing"""
    messages = [{"role": "tool", "content": '{"result": "success"}', "tool_call_id": "call_123"}]
    result = await _prepare_messages(messages)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert "[Tool Result call_123]:" in result[0].content[0].text


@pytest.mark.asyncio
async def test_prepare_messages_unknown_role():
    """Test unknown role handling with warning"""
    messages = [{"role": "unknown_role", "content": "test content"}]
    result = await _prepare_messages(messages)
    
    assert len(result) == 1
    assert result[0].role == "user"
    assert "[Unknown Role unknown_role]:" in result[0].content[0].text


@pytest.mark.asyncio
async def test_prepare_messages_empty_list():
    """Test empty message list"""
    result = await _prepare_messages([])
    assert len(result) == 0