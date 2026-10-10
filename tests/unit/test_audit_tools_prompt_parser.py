"""Regression cases for audit F06, F12 and F14; no external services."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_mneme.parser import parse_message_to_event
from hermes_mneme.prompt_builder import PromptBuilder
from hermes_mneme.prompt_builder import PromptBudgetExceeded
from hermes_mneme.store import ContextStore
from hermes_mneme.tools import ContextTools


def test_current_segment_search_uses_engine_topic(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    store.create_session("A", "test")
    index = Mock()
    index.search.return_value = []
    tools = ContextTools(store, index, None, None,
                         SimpleNamespace(session_id="A", current_segment_id="seg_A_2"))
    assert json.loads(tools._context_search({"query": "topic"}, "A")) == []
    index.search.assert_called_once_with("topic", "A", segment_id="seg_A_2", top_k=5)


def test_current_segment_rejects_global_scope(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    index = Mock()
    index.search_global.return_value = []
    tools = ContextTools(store, index, None, None,
                         SimpleNamespace(session_id="A", current_segment_id="seg_A_2"))
    result = json.loads(tools._context_search({"query": "topic", "session": "all"}, "A"))
    assert "error" in result
    index.search_global.assert_not_called()


def test_current_segment_explicit_session_uses_stored_topic(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    store.create_session("B", "test")
    store.add_event("B", "seg_B_3", "user_message", "user", "other topic")
    index = Mock()
    index.search.return_value = []
    tools = ContextTools(store, index, None, None,
                         SimpleNamespace(session_id="A", current_segment_id="seg_A_2"))
    tools._context_search({"query": "topic", "session": "B"}, "A")
    index.search.assert_called_once_with("topic", "B", segment_id="seg_B_3", top_k=5)


@pytest.mark.parametrize("kwargs", [
    {"current_user_message": "x" * 1000},
    {"current_user_message": "question", "memory_access_hint": "x" * 1000},
    {"current_user_message": "question", "system_prompt_override": "x" * 1000},
    {"current_user_message": "question", "checkpoint_block": "x" * 1000},
    {"current_user_message": "question", "request_prefix": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call1", "function": {"name": "tool", "arguments": "x" * 1000}}
        ]}
    ]},
])
def test_mandatory_prompt_overflow_is_explicit(kwargs):
    builder = PromptBuilder(total_budget=100)
    builder.tokenizer = len  # deterministic independent of tokenizer/download
    with pytest.raises(ValueError, match="budget"):
        builder.build({}, [], [], **kwargs)


def test_mixed_assistant_preserves_text_and_call_indices():
    calls = [{"id": "first", "function": {"name": "a", "arguments": "{}"}},
             {"id": "second", "function": {"name": "b", "arguments": "{}"}}]
    events = parse_message_to_event({"role": "assistant", "content": "Running both", "tool_calls": calls}, "A", "seg_A_1")
    assert [e["type"] for e in events] == ["tool_call", "tool_call", "assistant_message"]
    assert events[2]["content"] == "Running both"
    assert [e["tool_call_id"] for e in events[:2]] == ["first", "second"]


def test_tool_output_preserves_original_call_id():
    event = parse_message_to_event({"role": "tool", "content": "result", "tool_call_id": "first"}, "A", "seg_A_1")[0]
    assert event["tool_call_id"] == "first"


def test_all_segments_keeps_lineage_scope(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    store.get_lineage_chain = lambda sid: ["A0", "A1"]
    index = Mock()
    index.search_sessions.return_value = []
    tools = ContextTools(store, index, None, None,
                         SimpleNamespace(session_id="A1", current_segment_id="seg_A1_2"))
    tools._context_search({"query": "topic", "segment": "all"}, "A1")
    index.search_sessions.assert_called_once_with("topic", ["A0", "A1"], segment_id=None, top_k=5)


def test_current_segment_filters_lineage_including_legacy_id(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    store.get_lineage_chain = lambda sid: ["A0", "A1"]
    index = Mock()
    index.search_sessions.return_value = []
    tools = ContextTools(store, index, None, None,
                         SimpleNamespace(session_id="A1", current_segment_id="seg_1"))
    tools._context_search({"query": "topic"}, "A1")
    index.search_sessions.assert_called_once_with("topic", ["A0", "A1"], segment_id="seg_1", top_k=5)


def test_current_segment_rejects_ambiguous_multiple_sessions(tmp_path):
    store = ContextStore(str(tmp_path / "memory.db"))
    index = Mock()
    tools = ContextTools(store, index, None, None)
    result = json.loads(tools._context_search({"query": "topic", "session": ["A", "B"]}, "A"))
    assert "error" in result
    index.search_sessions.assert_not_called()


def test_prompt_accounts_reserved_overhead_without_double_count():
    builder = PromptBuilder(total_budget=100)
    builder.tokenizer = len
    builder.set_system_prompt_tokens(50)
    # 40 original system characters + 10 external tokens, plus Mneme state,
    # wrappers and user message: the system itself must not be charged twice.
    messages = builder.build({}, [], [], "question", system_prompt_override="x" * 40)
    assert sum(builder.count_message_tokens(m) for m in messages) + 10 <= 100
    builder.total_budget = 60
    with pytest.raises(PromptBudgetExceeded) as error:
        builder.build({}, [], [], "question", system_prompt_override="x" * 40)
    assert error.value.actual_tokens > error.value.budget_tokens == 60


def test_final_wrapper_check_drops_retrieval_before_tail():
    builder = PromptBuilder(total_budget=150, retrieved_budget_ratio=0.7,
                            protected_tail_ratio=0.1)
    builder.tokenizer = len
    retrieved = [SimpleNamespace(content="r" * 85, type="assistant_message")]
    recent = [{"role": "assistant", "content": "old"},
              {"role": "user", "content": "previous"}]
    messages = builder.build({}, retrieved, recent, "current")
    assert sum(builder.count_message_tokens(m) for m in messages) <= 150
    assert not any("[RETRIEVED CONTEXT]" in m["content"] for m in messages)
    assert messages[-3:] == recent + [{"role": "user", "content": "current"}]


def test_mandatory_recent_tool_payload_overflows_explicitly():
    builder = PromptBuilder(total_budget=100)
    builder.tokenizer = len
    recent = [{"role": "tool", "content": "huge " * 1000}]
    with pytest.raises(PromptBudgetExceeded):
        builder.build({}, [], recent, "current")


def test_mixed_replay_keeps_original_tool_event_hashes():
    calls = [{"id": "first", "function": {"name": "a", "arguments": "{}"}}]
    old = parse_message_to_event({"role": "assistant", "tool_calls": calls}, "A", "seg_A_1")
    mixed = parse_message_to_event({"role": "assistant", "content": "why", "tool_calls": calls}, "A", "seg_A_1")
    ids = lambda rows: [ContextStore.make_event_id("A", 7, e["role"], e["content"], i) for i, e in enumerate(rows)]
    assert ids(old) == ids(mixed)[:1]


def test_budget_trimming_keeps_tool_protocol_atomic():
    builder = PromptBuilder(total_budget=100, protected_tail_turns=6)
    builder.tokenizer = len
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "a", "arguments": "{}"}}]}
    recent = [call, {"role": "tool", "tool_call_id": "c1", "content": "one"},
              {"role": "assistant", "content": "answer"}]
    # The group cannot be dropped while keeping the existing two-message
    # recent-tail floor. Returning only its output would violate the API.
    with pytest.raises(PromptBudgetExceeded):
        builder.build({}, [], recent, "next")


def test_floor_selection_includes_call_for_parallel_outputs():
    builder = PromptBuilder(total_budget=400, protected_tail_turns=2)
    builder.tokenizer = len
    calls = {"role": "assistant", "content": "c" * 130, "tool_calls": [
        {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
        {"id": "c2", "function": {"name": "b", "arguments": "{}"}}]}
    recent = [{"role": "user", "content": "old " * 1000}, calls,
              {"role": "tool", "tool_call_id": "c2", "content": "second"},
              {"role": "tool", "tool_call_id": "c1", "content": "first"}]
    messages = builder.build({}, [], recent, "next")
    assert calls in messages
    assert [m.get("tool_call_id") for m in messages if m["role"] == "tool"] == ["c2", "c1"]


def test_atomic_group_removed_when_remaining_tail_satisfies_floor():
    builder = PromptBuilder(total_budget=120, protected_tail_turns=10,
                            protected_tail_ratio=1)
    builder.tokenizer = len
    recent = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "a", "arguments": "x" * 150}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "content": "answer"}]
    messages = builder.build({}, [], recent, "next")
    assert messages[1:] == recent[-2:] + [{"role": "user", "content": "next"}]


def test_tool_continuation_cannot_drop_current_user_request():
    builder = PromptBuilder(total_budget=200, protected_tail_turns=2)
    builder.tokenizer = len
    question = "mandatory request " * 100
    recent = [{"role": "user", "content": question},
              {"role": "assistant", "content": "", "tool_calls": [
                  {"id": "c1", "function": {"name": "a", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "result"}]
    with pytest.raises(PromptBudgetExceeded):
        builder.build({}, [], recent, question, append_current_user_message=False)
