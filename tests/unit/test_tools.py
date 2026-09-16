"""Tests for Mneme agent memory tools."""

from __future__ import annotations

import json

from hermes_mneme.store import ContextStore
from hermes_mneme.tools import ContextTools


class _GraphStub:
    def get_neighbors(self, event_id, depth=1):
        return []


class _EngineStub:
    session_id = "tool-session"


def test_fetch_event_accepts_unique_event_id_prefix(tmp_path):
    store = ContextStore(str(tmp_path / "plugin.db"))
    store.create_session("tool-session", "tui")
    event_id, inserted = store.add_event(
        session_id="tool-session",
        segment_id="seg1",
        event_type="user_message",
        role="user",
        content="prefix lookup payload",
        tool_name=None,
        tool_input=None,
        token_estimate=3,
    )
    assert inserted is True

    tools = ContextTools(store, None, None, _GraphStub(), _EngineStub())
    result = json.loads(tools._fetch_event({"event_id": event_id[:12]}))

    assert result["id"] == event_id
    assert result["content"] == "prefix lookup payload"
