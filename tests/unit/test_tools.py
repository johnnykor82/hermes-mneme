"""Tests for Mneme agent memory tools."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from hermes_mneme.store import ContextStore
from hermes_mneme.tools import ContextTools


class _GraphStub:
    def get_neighbors(self, event_id, depth=1):
        return []


class _EngineStub:
    session_id = "tool-session"


@pytest.mark.parametrize("k", ["30", -1, True, False, 3.5, [], {}])
def test_context_search_invalid_k_names_public_argument(k):
    index = Mock()
    index.search_global.return_value = []
    tools = ContextTools(None, index, None, None)
    result = json.loads(tools._context_search(
        {"query": "memory", "session": "all", "segment": "all", "k": k}, "A"
    ))
    assert isinstance(result, dict), "Invalid k must be rejected before searching"
    assert "k must be a non-negative integer or null" in result["error"]
    assert '{"k": 30}' in result["error"]
    index.search_global.assert_not_called()


@pytest.mark.parametrize("args", [{"top_k": 30}, {"top_k": "30", "k": 30}])
def test_context_search_rejects_internal_top_k_argument(args):
    index = Mock()
    index.search_global.return_value = []
    tools = ContextTools(None, index, None, None)
    result = json.loads(tools._context_search(
        {"query": "memory", "session": "all", "segment": "all", **args}, "A"
    ))
    assert isinstance(result, dict), "Unknown top_k must not silently select default k"
    assert "Use k, not top_k" in result["error"]
    index.search_global.assert_not_called()


@pytest.mark.parametrize("args, expected", [({}, 5), ({"k": 30}, 30), ({"k": 0}, 0), ({"k": None}, None)])
def test_context_search_preserves_valid_k_contract(args, expected):
    index = Mock()
    index.search_global.return_value = []
    tools = ContextTools(None, index, None, None)
    assert json.loads(tools._context_search(
        {"query": "memory", "session": "all", "segment": "all", **args}, "A"
    )) == []
    index.search_global.assert_called_once_with("memory", top_k=expected)


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
