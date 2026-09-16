"""Native Hermes ContextEngine hook tests."""

from __future__ import annotations

import sqlite3

from agent.context_engine import ContextEngine
from hermes_mneme.config import PluginConfig
from hermes_mneme.engine import CustomRouterContextEngine


def _engine(tmp_path, *, budget=100_000):
    test_config = {
        "active_window_tokens": budget,
        "pass_through_overhead_initial": 0,
        "llm_enrichment_enabled": False,
        "reranker_enabled": False,
    }
    cfg = PluginConfig(test_config)
    cfg._config.update(test_config)
    eng = CustomRouterContextEngine(
        db_path=str(tmp_path / "plugin.db"),
        config=cfg,
    )
    eng.on_session_start("native-session")
    return eng


def _shutdown(engine):
    for attr in ("_embed_executor", "_enrichment_executor"):
        executor = getattr(engine, attr, None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def test_native_hooks_detect_current_hermes_select_context(tmp_path):
    eng = _engine(tmp_path)
    try:
        assert callable(getattr(ContextEngine, "select_context", None))
        assert eng._native_request_hook_name() == "select_context"
        assert eng._native_hooks_available() is True
        assert eng.should_compress(prompt_tokens=999_999) is False
    finally:
        _shutdown(eng)


def test_legacy_host_keeps_compression_fallback(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    try:
        monkeypatch.setattr(ContextEngine, "select_context", None)
        monkeypatch.setattr(ContextEngine, "prepare_request_messages", None, raising=False)
        assert eng.should_compress() is True
        messages = [{"role": "user", "content": "legacy memory"}]
        assert eng.select_context(messages) is None
        eng.compress(messages)
        assert eng.store.count_events("native-session") == 1
    finally:
        _shutdown(eng)


def test_native_retries_and_completion_are_idempotent_without_compress(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    try:
        def unexpected_compress(*args, **kwargs):
            raise AssertionError("native hooks must not call compress")

        monkeypatch.setattr(eng, "compress", unexpected_compress)
        messages = [{"role": "user", "content": "remember this"}]
        for _ in range(2):
            eng.select_context(messages, conversation_messages=messages)
        completed = messages + [{"role": "assistant", "content": "remembered"}]
        for _ in range(2):
            eng.on_turn_complete(completed, usage=None)
        assert eng.store.count_events("native-session") == 2
        assert eng.should_compress() is False
    finally:
        _shutdown(eng)


def test_select_context_ingests_current_user_without_replacing_small_request(tmp_path):
    eng = _engine(tmp_path)
    try:
        conversation = [{"role": "user", "content": "remember the blue key"}]
        request = [{"role": "user", "content": "remember the blue key"}]

        selected = eng.select_context(
            request,
            conversation_messages=conversation,
            incoming_message=conversation[-1],
            budget_tokens=100_000,
            session_id="native-session",
        )

        assert selected is None
        assert eng.store.count_events("native-session") == 1

        eng.on_turn_complete(
            conversation + [{"role": "assistant", "content": "Noted."}],
            usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            session_id="native-session",
        )

        assert eng.store.count_events("native-session") == 2
    finally:
        _shutdown(eng)


def test_select_context_returns_request_only_context_when_assembly_needed(tmp_path):
    eng = _engine(tmp_path, budget=80)
    try:
        conversation = [
            {"role": "user", "content": "alpha " * 200},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "what did I say?"},
        ]

        selected = eng.select_context(
            list(conversation),
            conversation_messages=list(conversation),
            incoming_message=conversation[-1],
            budget_tokens=80,
            session_id="native-session",
        )

        assert selected is not None
        assert selected is not conversation
        assert selected[0]["role"] == "system"
        assert "[EXECUTION STATE]" in selected[0]["content"]
        assert selected[-1]["role"] == "user"
        assert selected[-1]["content"] == "what did I say?"
        assert conversation[0]["content"].startswith("alpha ")
    finally:
        _shutdown(eng)


def test_select_context_preserves_request_system_prompt_without_ingesting_it(tmp_path):
    eng = _engine(tmp_path, budget=80)
    try:
        system_text = "ORIGINAL HERMES SYSTEM PROMPT"
        conversation = [
            {"role": "user", "content": "alpha " * 200},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "what did I say?"},
        ]
        request = [{"role": "system", "content": system_text}, *conversation]

        selected = eng.select_context(
            request,
            conversation_messages=list(conversation),
            incoming_message=conversation[-1],
            budget_tokens=80,
            session_id="native-session",
        )

        assert selected is not None
        assert selected[0]["role"] == "system"
        assert system_text in selected[0]["content"]
        assert "[EXECUTION STATE]" in selected[0]["content"]
        assert eng.prompt_builder.system_prompt_tokens > 0
        assert eng.store.count_events("native-session") == 3
        conn = sqlite3.connect(tmp_path / "plugin.db")
        try:
            stored = conn.execute(
                "SELECT COUNT(*) FROM events WHERE content LIKE ?",
                (f"%{system_text}%",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert stored == 0
    finally:
        _shutdown(eng)


def test_select_context_does_not_reappend_user_after_tool_outputs(tmp_path):
    eng = _engine(tmp_path, budget=120)
    try:
        search_prompt = (
            'Сделай context_search отдельно по трём запросам: '
            '"документация YouTube противоречия", '
            '"популярные skills Claude Code Codex", '
            '"проверка сжатия контекста Mneme"'
        )
        conversation = [
            {"role": "user", "content": "alpha " * 200},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": search_prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "context_search",
                            "arguments": '{"query":"документация YouTube противоречия"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "context_search",
                "content": '[{"snippet":"found prior YouTube docs"}]',
            },
        ]

        selected = eng.select_context(
            list(conversation),
            conversation_messages=list(conversation),
            incoming_message=conversation[-1],
            budget_tokens=120,
            session_id="native-session",
        )

        assert selected is not None
        assert selected[-1]["role"] == "tool"
        assert selected[-1]["content"] == '[{"snippet":"found prior YouTube docs"}]'
    finally:
        _shutdown(eng)


def test_select_context_preserves_ephemeral_prefill_without_ingesting_it(tmp_path):
    eng = _engine(tmp_path, budget=80)
    try:
        conversation = [
            {"role": "user", "content": "alpha " * 200},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "continue"},
        ]
        prefill = {"role": "assistant", "content": "EPHEMERAL PREFILL"}
        selected = eng.select_context(
            [{"role": "system", "content": "system"}, prefill, *conversation],
            conversation_messages=conversation,
        )
        assert selected[1] == prefill
        assert selected[-1] == conversation[-1]
        assert eng.store.count_events("native-session") == len(conversation)
        assert conversation[-1]["role"] == "user"
    finally:
        _shutdown(eng)
