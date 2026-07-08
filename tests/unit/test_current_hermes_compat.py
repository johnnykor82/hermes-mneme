import copy

from hermes_mneme import config as config_module
from hermes_mneme.engine import CustomRouterContextEngine


def _build_engine(tmp_path, **overrides):
    cfg = {
        "active_window_tokens": 0,
        "context_window_usage_percent": 0.50,
        "llm_enrichment_enabled": False,
        "reranker_enabled": False,
        "debug_mode": False,
        "memory_access_hint_enabled": False,
        "reindex_on_model_change": False,
    }
    cfg.update(overrides)
    return CustomRouterContextEngine(
        db_path=str(tmp_path / "plugin.db"),
        config=config_module.PluginConfig(cfg),
    )


def _shutdown(engine):
    for attr in ("_embed_executor", "_enrichment_executor"):
        executor = getattr(engine, attr, None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def test_deepcopy_creates_isolated_engine_with_model_budget_state(tmp_path):
    engine = _build_engine(tmp_path)
    clone = None
    try:
        engine.update_model(
            model="compat-model",
            context_length=2000,
            base_url="http://example.invalid/v1",
            api_key="test-key",
            provider="compat-provider",
        )
        engine.last_prompt_tokens = 111
        engine.last_completion_tokens = 22
        engine.last_total_tokens = 133
        engine._observed_prompt_overhead = 444
        engine._last_content_estimate = 555

        clone = copy.deepcopy(engine)

        assert clone is not engine
        assert clone.config is not engine.config
        assert clone.config.as_dict == engine.config.as_dict
        assert clone.store is not engine.store
        assert clone.indexer is not engine.indexer
        assert clone.store.db_path == engine.store.db_path
        assert clone.tools.engine is clone
        assert clone._embed_executor is not engine._embed_executor
        assert clone._enrichment_executor is not engine._enrichment_executor

        assert clone.context_length == engine.context_length
        assert clone.threshold_tokens == engine.threshold_tokens
        assert clone._budget_tokens == engine._budget_tokens
        assert clone.prompt_builder.total_budget == engine.prompt_builder.total_budget
        assert clone._hermes_llm == engine._hermes_llm
        assert clone._hermes_llm is not engine._hermes_llm
        assert clone.last_prompt_tokens == 111
        assert clone.last_completion_tokens == 22
        assert clone.last_total_tokens == 133
        assert clone._observed_prompt_overhead == 444
        assert clone._last_content_estimate == 555
    finally:
        _shutdown(engine)
        if clone is not None:
            _shutdown(clone)


def test_deepcopy_does_not_inherit_active_session_state(tmp_path):
    engine = _build_engine(tmp_path)
    clone = None
    try:
        engine.session_id = "source-session"
        engine.current_segment_id = "seg_source-session_1"
        engine._pending_session_id = "pending-session"
        engine._processed_msg_count = 7
        engine._current_state["session_id"] = "source-session"
        engine._current_state["segment_id"] = "seg_source-session_1"

        clone = copy.deepcopy(engine)

        assert clone.session_id is None
        assert clone.current_segment_id == "seg_1"
        assert clone._pending_session_id is None
        assert clone._processed_msg_count == 0
        assert clone._current_state["session_id"] is None
        assert clone._current_state["segment_id"] is None
    finally:
        _shutdown(engine)
        if clone is not None:
            _shutdown(clone)


def test_compress_accepts_current_hermes_force_keyword(tmp_path):
    engine = _build_engine(tmp_path)
    try:
        messages = [{"role": "user", "content": "hello"}]

        assert engine.compress(
            messages,
            current_tokens=1,
            focus_topic="compat smoke",
            force=True,
        ) == messages
    finally:
        _shutdown(engine)
