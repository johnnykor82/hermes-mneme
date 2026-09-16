"""Request selection retries must retain one-shot context until completion."""

import copy
from unittest.mock import Mock

import pytest

from hermes_mneme import index, observability
from test_native_hooks import _engine, _shutdown


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(index.EmbeddingIndex, "add_embeddings_batch", lambda *a: 0)
    monkeypatch.setattr(index.EmbeddingIndex, "get_embedding", lambda *a: None)
    monkeypatch.setattr(observability.Observability, "log_turn", lambda *a, **kw: None)
    eng = _engine(tmp_path)
    yield eng
    _shutdown(eng)


def test_retry_preserves_resume_and_checkpoint_without_reassembly(engine, monkeypatch):
    engine._resume_context_fill_pending = True
    engine._pending_checkpoint = "[CHECKPOINT] return to the current task"
    messages = [{"role": "user", "content": "continue previous work"}]
    pipeline = Mock(wraps=engine._process_context_messages)
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    first = engine.select_context(messages, conversation_messages=messages)
    assert first is not None
    assert "[CHECKPOINT]" in first[0]["content"]
    expected = copy.deepcopy(first)
    first[0]["content"] = "provider sanitizer mutation"
    retry = engine.select_context(copy.deepcopy(messages), conversation_messages=copy.deepcopy(messages))
    assert retry == expected
    retry[0]["content"] = "second mutation"
    assert engine.select_context(messages, conversation_messages=messages) == expected
    assert pipeline.call_count == 1
    assert engine.store.count_events("native-session") == 1


@pytest.mark.parametrize("changed", ["request", "conversation", "incoming", "budget", "session", "host_session"])
def test_changed_inputs_invalidate_cached_selection(engine, monkeypatch, changed):
    pipeline = Mock(return_value=[{"role": "system", "content": "selected"}])
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    request = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    kwargs = dict(conversation_messages=copy.deepcopy(request), incoming_message=copy.deepcopy(request[0]), budget_tokens=100)
    engine.select_context(request, **kwargs)
    engine.select_context(copy.deepcopy(request), **copy.deepcopy(kwargs))
    assert pipeline.call_count == 1
    if changed == "request":
        request[0]["content"][0]["text"] = "changed"
    elif changed == "conversation":
        kwargs["conversation_messages"][0]["content"][0]["text"] = "changed"
    elif changed == "incoming":
        kwargs["incoming_message"]["content"][0]["text"] = "changed"
    elif changed == "budget":
        kwargs["budget_tokens"] = 200
    elif changed == "session":
        engine.session_id = "other-session"
    else:
        from hermes_logging import _session_context
        monkeypatch.setattr(_session_context, "session_id", "other-host-session", raising=False)
    engine.select_context(request, **kwargs)
    assert pipeline.call_count == 2


def test_completion_invalidates_retry_cache(engine, monkeypatch):
    engine._resume_context_fill_pending = True
    messages = [{"role": "user", "content": "continue previous work"}]
    first = engine.select_context(messages, conversation_messages=messages)
    assert first is not None
    pipeline = Mock(wraps=engine._process_context_messages)
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    engine.on_turn_complete(messages)
    assert engine.select_context(messages, conversation_messages=messages) is None
    assert pipeline.call_count == 2


def test_same_session_restart_invalidates_retry_cache(engine, monkeypatch):
    pipeline = Mock(return_value=[{"role": "system", "content": "selected"}])
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    messages = [{"role": "user", "content": "hello"}]
    engine.select_context(messages, conversation_messages=messages)
    engine.on_session_start("native-session")
    engine.select_context(messages, conversation_messages=messages)
    assert pipeline.call_count == 2


def test_cache_retains_only_latest_request(engine, monkeypatch):
    pipeline = Mock(return_value=[{"role": "system", "content": "selected"}])
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    first = [{"role": "user", "content": "first"}]
    second = [{"role": "user", "content": "second"}]
    for messages in (first, second, first):
        engine.select_context(messages, conversation_messages=messages)
    assert pipeline.call_count == 3


@pytest.mark.parametrize("boundary", ["reset", "end", "model"])
def test_lifecycle_clears_cache(engine, monkeypatch, boundary):
    pipeline = Mock(return_value=[{"role": "system", "content": "selected"}])
    monkeypatch.setattr(engine, "_process_context_messages", pipeline)
    messages = [{"role": "user", "content": "hello"}]
    engine.select_context(messages, conversation_messages=messages)
    if boundary == "reset":
        engine.on_session_reset()
    elif boundary == "end":
        engine.on_session_end("native-session", messages)
    else:
        engine.update_model("test-model", 100_000)
    assert engine._native_request_cache is None
