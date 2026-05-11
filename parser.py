import json
import uuid
from typing import List, Dict, Any

def parse_message_to_event(msg: Dict[str, Any], session_id: str, segment_id: str) -> List[Dict[str, Any]]:
    """
    Converts a single message dict (OpenAI format) into one or more events for the store.
    Returns a list because one message (e.g. assistant with tool_calls) can generate multiple events.
    """
    events = []
    role = msg.get("role")
    content = msg.get("content", "")
    
    # Calculate a rough token estimate (chars / 4)
    token_estimate = len(str(content)) // 4 if content else 0

    if role == "user":
        events.append({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "segment_id": segment_id,
            "type": "user_message",
            "role": "user",
            "content": content,
            "tool_name": None,
            "tool_input": None,
            "token_estimate": token_estimate
        })
    
    elif role == "assistant":
        # Check for tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                # Event for the tool call itself
                events.append({
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "segment_id": segment_id,
                    "type": "tool_call",
                    "role": "assistant",
                    "content": json.dumps(tc.get("function", {})), # Store function details
                    "tool_name": tc.get("function", {}).get("name"),
                    "tool_input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    "token_estimate": token_estimate // max(len(tool_calls), 1)
                })
        else:
            # Regular assistant message
            events.append({
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "segment_id": segment_id,
                "type": "assistant_message",
                "role": "assistant",
                "content": content,
                "tool_name": None,
                "tool_input": None,
                "token_estimate": token_estimate
            })
            
    elif role == "tool":
        # Tool output
        events.append({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "segment_id": segment_id,
            "type": "tool_output",
            "role": "tool",
            "content": content,
            "tool_name": msg.get("name"), # Tool name might be in 'name' or derived from context
            "tool_input": None,
            "token_estimate": token_estimate
        })

    return events
