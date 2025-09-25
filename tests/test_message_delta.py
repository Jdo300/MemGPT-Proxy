import pytest

from main import (
    _collect_system_content,
    _extract_latest_user_message,
    _extract_trailing_tool_messages,
    _normalize_content,
)


def test_collect_system_content_multiple_blocks():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Follow safety guidelines."},
                {"type": "text", "text": "Be concise."},
            ],
        },
    ]

    combined = _collect_system_content(messages)
    assert "You are helpful." in combined
    assert "Follow safety guidelines." in combined
    assert combined.count("\n\n") == 1


def test_collect_system_content_ignores_empty():
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
    ]

    assert _collect_system_content(messages) is None


def test_extract_latest_user_message_prefers_last():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Ack"},
        {"role": "user", "content": [
            {"type": "text", "text": "Second"},
            " line",
        ]},
    ]

    assert _extract_latest_user_message(messages) == "Second line"


def test_extract_trailing_tool_messages_after_last_assistant():
    messages = [
        {"role": "user", "content": "Ask"},
        {"role": "assistant", "content": "Calling tool"},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"value\": 1}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "done"},
        {"role": "user", "content": "Thanks"},
    ]

    trailing = _extract_trailing_tool_messages(messages)
    assert [msg["tool_call_id"] for msg in trailing] == ["call_1", "call_2"]


def test_extract_trailing_tool_messages_handles_no_assistant():
    messages = [
        {"role": "tool", "tool_call_id": "call_3", "content": "value"},
        {"role": "user", "content": "Proceed"},
    ]

    trailing = _extract_trailing_tool_messages(messages)
    assert len(trailing) == 1
    assert trailing[0]["tool_call_id"] == "call_3"


def test_normalize_content_handles_nested_list():
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": ", world"},
        "!",
    ]
    assert _normalize_content(content) == "Hello, world!"
