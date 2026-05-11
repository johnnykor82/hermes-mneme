import pytest
from parser import parse_message_to_event

def test_parse_user_message():
    """Test parsing a simple user message."""
    msg = {"role": "user", "content": "Hello, world!"}
    events = parse_message_to_event(msg, "sess1", "seg_1")
    
    assert len(events) == 1
    assert events[0]["type"] == "user_message"
    assert events[0]["role"] == "user"
    assert events[0]["content"] == "Hello, world!"
    assert events[0]["session_id"] == "sess1"

def test_parse_assistant_message():
    """Test parsing a simple assistant message."""
    msg = {"role": "assistant", "content": "Hi there!"}
    events = parse_message_to_event(msg, "sess1", "seg_1")
    
    assert len(events) == 1
    assert events[0]["type"] == "assistant_message"  # Correct spelling
    assert events[0]["content"] == "Hi there!"

def test_parse_tool_call():
    """Test parsing a message with tool calls."""
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "web_search",
                    "arguments": "{\"query\": \"test\"}"
                }
            }
        ]
    }
    events = parse_message_to_event(msg, "sess1", "seg_1")
    
    # Should produce one event for the tool call
    assert len(events) == 1
    assert events[0]["type"] == "tool_call"
    assert events[0]["tool_name"] == "web_search"

def test_parse_tool_output():
    """Test parsing a tool output message."""
    msg = {"role": "tool", "content": "Search results here", "name": "web_search"}
    events = parse_message_to_event(msg, "sess1", "seg_1")
    
    assert len(events) == 1
    assert events[0]["type"] == "tool_output"
    assert events[0]["tool_name"] == "web_search"
