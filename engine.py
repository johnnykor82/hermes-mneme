import json
import logging
import os
import threading
from typing import Dict, List, Any, Optional

MEMORY_ACCESS_HINT_SHORT = """[MEMORY ACCESS]
Memory tools доступны: list_segments, context_search(query, segment='all', \
session='all'), expand_context, fetch_event(id), get_goal_history, \
get_execution_state, recall_recent(turns). Зови их, когда нужен контекст \
из прошлого, которого нет в активном окне."""

MEMORY_ACCESS_HINT = """[MEMORY ACCESS]
Эта сессия — длинная. Большая часть прошлых шагов осталась в базе и в
активном окне их нет. У тебя есть инструменты, чтобы достать любое
прошлое событие. Используй их в следующих случаях.

ОБЯЗАТЕЛЬНО сходи в память ПРЕЖДЕ ЧЕМ:
  1. Переспрашивать пользователя «мы это обсуждали?», «как ты говорил
     назвать X?», «какой путь к Y?» — сначала проверь, не было ли уже
     ответа в сессии.
  2. Запускать поиск/чтение/команду, которая может повторить уже
     сделанную работу: «найти все упоминания X», «прочитать файл Y»,
     «проверить, есть ли Z». Если что-то из этого делалось ранее в сессии —
     результат уже в памяти, не повторяй.
  3. Делать вывод о коде/архитектуре, который требует знания решений,
     принятых раньше в этой же сессии («мы вроде договаривались...»,
     «по-моему, мы выбирали подход A»). Сверься с памятью, не с догадкой.
  4. Начинать новую задачу, у которой может быть продолжение в прошлом
     сегменте (пользователь говорит «давай вернёмся к X», «продолжим то,
     что начали с Y», «доделай Z»).
  5. Объявлять что-то «новым» или «свежим», если по контексту это могло
     уже всплывать (имя файла, переменная, термин из домена пользователя).

ИНСТРУМЕНТЫ:
  • list_segments() — оглавление сессии: какие темы были и в каких сегментах.
    Дёшево, начинай с него если не знаешь, куда копать.
  • context_search(query, segment='all', session='current') — семантический
    поиск по всей сессии.
  • expand_context(seed_event_id, mode='segment') — костяк выбранного
    сегмента (ключевые user-сообщения, решения, последние tool-выводы).
  • fetch_event(event_id) — полный текст конкретного события (если retrieved
    или search-сниппет обрезан).
  • get_goal_history(20) — последние 20 целей сессии. Зови, если
    [GOAL TRAIL] выше не отвечает на вопрос «что я делал 20 ходов назад».
  • get_execution_state() — текущие goal / open_loops / active_entities.

ПОРЯДОК ПОИСКА (рекомендуемый):
  1. Сначала глянь [EXECUTION STATE] и [GOAL TRAIL] выше — часто ответа уже хватает.
  2. Если нет — list_segments() для карты, потом context_search или
     expand_context в нужный сегмент.
  3. fetch_event только когда нужен полный оригинал.

ЕСЛИ ПЕТЛЯ. Не делай больше 3–4 memory-вызовов подряд без полезного
действия. Если 3 поиска не дали ответа — остановись и сформулируй
пользователю, что именно ищешь, вместо ещё одного слепого поиска."""

from agent.context_engine import ContextEngine

from . import parser
from . import index
from . import router
from . import prompt_builder
from . import observability
from . import config as config_module
from . import store
from . import tools
from . import segmenter
from . import graph
from . import compressor
from . import enrichment

logger = logging.getLogger(__name__)

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(PLUGIN_DIR, "db", "plugin.db")

# Per-session write locks (spec: Component 3 — per-session mutex).
# `_session_locks_dict_lock` (A2) protects insertions into the dict so that
# two threads acquiring locks for the same fresh session_id at the same time
# can't race on `if session_id not in _session_locks` and end up with two
# distinct Lock objects (defeating the mutex).
_session_locks: Dict[str, threading.Lock] = {}
_session_locks_dict_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_dict_lock:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


class CustomRouterContextEngine(ContextEngine):
    def __init__(self, db_path: str = DEFAULT_DB_PATH, config: config_module.PluginConfig = None):
        super().__init__()
        self.session_id = None
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if config is None:
            config = config_module.PluginConfig()
        self.config = config

        # Initialize components
        self.store = store.ContextStore(db_path)
        self.indexer = index.EmbeddingIndex(db_path)
        self.current_segment_id = "seg_1"
        self.segmenter = segmenter.SessionSegmenter(self.store, self.indexer, self.current_segment_id, config=self.config)
        self.context_router = router.ContextRouter(self.store, self.indexer, config=self.config, segmenter=self.segmenter)
        self.prompt_builder = prompt_builder.PromptBuilder(
            total_budget=self.config.get("active_window_tokens", 16000),
            tokenizer_model=self.config.get("tokenizer_model", "cl100k_base"),
            protected_tail_turns=self.config.get("protected_tail_turns", 20),
            state_budget_ratio=self.config.get("state_budget_ratio", 0.05),
            retrieved_budget_ratio=self.config.get("retrieved_budget_ratio", 0.45),
            protected_tail_ratio=self.config.get("protected_tail_ratio", None),
        )
        self.observability = observability.Observability()
        self.graph = graph.ExecutionGraph(db_path)
        self.tools = tools.ContextTools(self.store, self.indexer, self.context_router, self.graph, engine=self)

        self._processed_msg_count = 0
        # A5: cached session_id from on_session_start, used by compress() if
        # hermes_logging._session_context is unavailable.
        self._pending_session_id: Optional[str] = None
        # G: prompt-overhead bookkeeping for the pass-through guard.
        # `_observed_prompt_overhead` is the gap we've measured between
        # `content_tokens` (what the plugin's pass-through check sees) and
        # `prompt_tokens` (what Hermes actually sent to the LLM). Starts
        # at a configured estimate (system prompt + tool schemas + reasoning
        # wrappers — empirically ~16k in real Hermes installs); updated on
        # every update_from_response with the real observed delta.
        # `_last_content_estimate` is the content estimate from the most
        # recent compress() pass — used to compute the real overhead when
        # update_from_response lands.
        self._observed_prompt_overhead: int = int(
            self.config.get("pass_through_overhead_initial", 16000) or 0
        )
        self._last_content_estimate: int = 0
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        # H: resume context-fill flag. Set on RESUME paths in on_session_start
        # and in compress() drift detection. While True, the pass-through
        # guard refuses to short-circuit — the first turn after a process
        # restart must always exercise the assembly path so the prompt
        # builder pulls retrieved context from the DB. Cleared once we've
        # gone through assembly at least once.
        self._resume_context_fill_pending: bool = False
        # B5: cross-session bootstrap probe results. Populated once per
        # session at the first ingested user_message (if enabled) and
        # rendered into the prompt as a [CROSS-SESSION CANDIDATES] block.
        # Cleared on session reset / fresh bind.
        self._cross_session_candidates: List[Dict[str, Any]] = []
        self._bootstrap_probe_done: bool = False
        self._current_state = {
            "schema_version": 1,
            "session_id": None,
            "segment_id": None,
            "goal": "",
            "current_step": "",
            "open_loops": [],
            "last_tool": None,
            "last_tool_output_summary": None,
            "decision_stack": [],
            "active_entities": [],
            "turn_count": 0,
            "enrichment": {"decision_summary": None, "intent_label": None, "topic_tags": []}
        }
        # Context window budget (recalculated on update_model)
        self._budget_tokens = self._calculate_budget()
        # Hermes LLM params, populated by update_model() — used as enricher fallback.
        self._hermes_llm: Dict[str, str] = {}
        self.enricher = enrichment.LLMEnricher(self.config, hermes_llm=self._hermes_llm)
        # Memory-tool consecutive-call counter — drives [CHECKPOINT] injection.
        # Reset to 0 by any "useful" tool call (Edit/Write/Bash with side effects)
        # or by an assistant text response longer than ~200 tokens.
        self._consecutive_memory_calls = 0
        self._pending_checkpoint: Optional[str] = None
        self._last_enrichment_turn = -1
        logger.info(f"CustomRouterContextEngine initialized. DB: {db_path}")

        # One-shot phantom-session cleanup (A0): drop legacy `sessions` rows
        # that have no events, no parent, and were last touched >24h ago.
        # Catches phantoms left over by older plugin versions that did eager
        # create_session() on FRESH/drift paths. Safe: 24h window protects
        # any in-flight session.
        try:
            removed = self.store.delete_empty_sessions(min_age_seconds=86400)
            if removed:
                logger.info(f"Startup cleanup: removed {removed} empty phantom session rows")
        except Exception as e:
            logger.warning(f"Startup cleanup of empty sessions failed: {e}")

        # Reindex check (Stage 4): if embedding model in DB differs from config —
        # either re-embed everything or warn, depending on reindex_on_model_change.
        self._check_and_reindex_embeddings()

    def _check_and_reindex_embeddings(self) -> None:
        """Stage 4: detect embedding model change in DB vs config.

        - If DB has only the configured model (or is empty) → no-op.
        - If DB has a different model AND reindex_on_model_change=True → drop old
          embeddings (embedding_index + vec_items + vec_items_meta) for stale
          models and re-embed every event with non-empty content under the
          configured model. Progress logged every 100 events.
        - Otherwise → log a warning that retrieval may degrade until the user
          either reindexes or reverts the model.
        """
        from .index import JINA_MODEL
        configured_model = JINA_MODEL
        try:
            conn = self.store._get_connection()
            rows = conn.execute(
                "SELECT DISTINCT embedding_model_id FROM embedding_index"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"Reindex check: failed to read embedding_index: {e}")
            return

        existing_models = {r[0] for r in rows if r and r[0]}
        if not existing_models:
            logger.info("Reindex check: embedding_index empty, nothing to do.")
            return
        stale_models = existing_models - {configured_model}
        if not stale_models:
            logger.info(f"Reindex check: all embeddings under configured model {configured_model}.")
            return

        if not self.config.get("reindex_on_model_change", False):
            logger.warning(
                f"Embedding model changed: DB has {sorted(existing_models)}, "
                f"config wants {configured_model}. retrieval may degrade. "
                f"Set reindex_on_model_change=true to rebuild."
            )
            return

        logger.info(
            f"Reindex starting: dropping stale models {sorted(stale_models)}, "
            f"target {configured_model}"
        )
        try:
            self._reindex_under_model(configured_model, stale_models)
        except Exception as e:
            logger.error(f"Reindex failed: {e}")

    def _reindex_under_model(self, target_model: str, stale_models: set) -> None:
        """Drop embeddings for stale models and re-embed all events under target_model."""
        from .index import JINA_MODEL

        conn = self.store._get_connection()
        try:
            placeholders = ",".join("?" for _ in stale_models)
            stale_list = list(stale_models)

            cur = conn.execute(
                f"SELECT event_id FROM embedding_index WHERE embedding_model_id IN ({placeholders})",
                stale_list,
            )
            stale_event_ids = [r[0] for r in cur.fetchall()]

            if stale_event_ids:
                cur = conn.execute(
                    f"SELECT rowid FROM vec_items_meta WHERE embedding_model_id IN ({placeholders})",
                    stale_list,
                )
                stale_rowids = [r[0] for r in cur.fetchall()]
                if stale_rowids:
                    rid_placeholders = ",".join("?" for _ in stale_rowids)
                    try:
                        conn.execute(
                            f"DELETE FROM vec_items WHERE rowid IN ({rid_placeholders})",
                            stale_rowids,
                        )
                    except Exception as e:
                        logger.warning(f"Reindex: vec_items delete failed: {e}")
                    conn.execute(
                        f"DELETE FROM vec_items_meta WHERE embedding_model_id IN ({placeholders})",
                        stale_list,
                    )
                conn.execute(
                    f"DELETE FROM embedding_index WHERE embedding_model_id IN ({placeholders})",
                    stale_list,
                )
                conn.commit()
                logger.info(f"Reindex: dropped {len(stale_event_ids)} stale embeddings.")

            cur = conn.execute(
                "SELECT id, segment_id, content, type, token_estimate FROM events "
                "WHERE content IS NOT NULL AND length(trim(content)) > 5 "
                "ORDER BY timestamp ASC"
            )
            events_to_embed = cur.fetchall()
        finally:
            conn.close()

        total = len(events_to_embed)
        if total == 0:
            logger.info("Reindex: no events to re-embed.")
            return

        logger.info(f"Reindex: re-embedding {total} events under {target_model}")
        embedded = 0
        skipped = 0
        for i, row in enumerate(events_to_embed, 1):
            event_id, segment_id, content, ev_type, token_estimate = row
            text_for_embedding = content
            if ev_type == "tool_output" and text_for_embedding:
                text_for_embedding = compressor.summarize_for_embedding(
                    text_for_embedding,
                    threshold_tokens=self.config.get("tool_output_compress_threshold_tokens", 500),
                    summary_tokens=self.config.get("tool_output_summary_tokens", 100),
                )
            try:
                self.indexer.add_embedding(
                    event_id, segment_id, text_for_embedding,
                    target_model, token_estimate or 0, ev_type,
                )
                embedded += 1
            except Exception as e:
                skipped += 1
                logger.warning(f"Reindex: embed failed for {event_id}: {e}")

            if i % 100 == 0:
                logger.info(f"Reindex progress: {i}/{total} events processed ({embedded} ok, {skipped} skipped)")

        logger.info(f"Reindex done: {embedded}/{total} events re-embedded, {skipped} skipped.")

    @property
    def name(self) -> str:
        return "hermes-mneme"

    def _maybe_enrich(self, segment_changed: bool) -> None:
        """Hybrid trigger: every N turns OR on segment boundary.

        On failure or disabled flag — silently keeps existing enrichment.
        Never blocks the turn: any exception is logged and swallowed.
        """
        if not self.config.get("llm_enrichment_enabled", False):
            logger.info("Enricher: disabled in config (llm_enrichment_enabled=False).")
            return
        ready = self.enricher.is_ready()
        resolved = self.enricher._resolve_endpoint() if hasattr(self.enricher, "_resolve_endpoint") else {}
        if not ready:
            logger.info(
                f"Enricher: NOT ready. endpoint={resolved.get('endpoint')!r} "
                f"model={resolved.get('model')!r} hermes_llm_keys={list(self._hermes_llm.keys())}"
            )
            return

        every_n = int(self.config.get("enricher_every_n_turns", 5) or 5)
        on_boundary = bool(self.config.get("enricher_on_segment_boundary", True))
        turn = self._current_state.get("turn_count", 0)

        boundary_trigger = on_boundary and segment_changed
        every_n_trigger = every_n > 0 and (turn - max(self._last_enrichment_turn, 0)) >= every_n
        if turn < 1:
            logger.info(f"Enricher: skip — turn_count={turn} (<1, nothing to enrich yet)")
            return
        if not (boundary_trigger or every_n_trigger):
            logger.info(
                f"Enricher: skip — turn={turn}, last_run={self._last_enrichment_turn}, "
                f"every_n={every_n}, segment_changed={segment_changed}"
            )
            return
        logger.info(
            f"Enricher: TRIGGERING — endpoint={resolved.get('endpoint')!r} "
            f"model={resolved.get('model')!r} turn={turn} "
            f"trigger={'boundary' if boundary_trigger else f'every_{every_n}'}"
        )

        max_turns = int(self.config.get("enricher_max_history_turns", 10) or 10)
        try:
            recent = self.store.get_recent_events(self.session_id, limit=max_turns * 2)
        except Exception as e:
            logger.warning(f"Enricher: failed to load recent events: {e}")
            return

        try:
            result = self.enricher.enrich(recent)
        except Exception as e:
            logger.warning(f"Enricher: enrich() raised: {e}")
            return

        if result is None:
            return

        # Merge into _current_state.enrichment without nuking unrelated fields.
        enrichment_block = self._current_state.get("enrichment") or {}
        if result.intent_label:
            enrichment_block["intent_label"] = result.intent_label
        if result.topic_tags:
            enrichment_block["topic_tags"] = result.topic_tags
        if result.decisions:
            # Keep last 5 decisions in decision_stack (FIFO).
            stack = self._current_state.get("decision_stack") or []
            for d in result.decisions:
                if d not in stack:
                    stack.append(d)
            self._current_state["decision_stack"] = stack[-5:]
            enrichment_block["decision_summary"] = result.decisions[-1].get("decision")
        self._current_state["enrichment"] = enrichment_block
        self._last_enrichment_turn = turn
        logger.info(
            f"Enricher: state updated (intent={bool(result.intent_label)}, "
            f"tags={len(result.topic_tags)}, decisions={len(result.decisions)}, "
            f"trigger={'boundary' if boundary_trigger else f'every_{every_n}'})"
        )

    def _calculate_budget(self) -> int:
        """Calculate context window budget from config.
        Priority: absolute active_window_tokens > percentage of model context.
        """
        absolute = self.config.get("active_window_tokens", 0)
        if absolute and absolute > 0:
            return absolute
        # Percentage mode
        percent = self.config.get("context_window_usage_percent", 0.70)
        context_len = getattr(self, 'context_length', 128000)
        return int(context_len * percent)

    def update_model(self, model: str, context_length: int, base_url: str = "",
                     api_key: str = "", provider: str = "", **kwargs) -> None:
        """Called by Hermes when model switches. Recalculate budget and
        capture LLM endpoint params for use by enricher fallback."""
        self.context_length = context_length
        self._budget_tokens = self._calculate_budget()
        # Save Hermes LLM params; enricher reads from this dict at call-time.
        self._hermes_llm.clear()
        self._hermes_llm.update({
            "model": model or "",
            "base_url": base_url or "",
            "api_key": api_key or "",
            "provider": provider or "",
        })
        absolute = self.config.get("active_window_tokens", 0)
        if absolute and absolute > 0:
            mode = f"absolute={absolute}"
        else:
            pct = self.config.get("context_window_usage_percent", 0.70)
            mode = f"percent={pct:.0%} of {context_length:,}"
        logger.info(f"Context budget: {self._budget_tokens:,} tokens ({mode})")
        # Re-initialize prompt builder with new budget
        self.prompt_builder = prompt_builder.PromptBuilder(
            total_budget=self._budget_tokens,
            tokenizer_model=self.config.get("tokenizer_model", "cl100k_base"),
            protected_tail_turns=self.config.get("protected_tail_turns", 20),
            state_budget_ratio=self.config.get("state_budget_ratio", 0.05),
            retrieved_budget_ratio=self.config.get("retrieved_budget_ratio", 0.45),
            protected_tail_ratio=self.config.get("protected_tail_ratio", None),
        )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from API response (called after every LLM call)."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

        # NB: turn_count is no longer incremented here. It is owned by
        # _update_state_from_event() so the count tracks ingested
        # user_message events deterministically — independent of whether
        # Hermes happens to call this hook.
        self._current_state["session_id"] = self.session_id
        self._current_state["segment_id"] = self.current_segment_id

        # G: recalibrate prompt-overhead estimate from this round-trip.
        # `_last_content_estimate` is the content_tokens that compress()
        # saw in its pass-through check. Anything in `prompt_tokens` above
        # that is system prompt + tool schemas + tool_call/tool_output JSON
        # framing — the bits the plugin's content-only sum misses. We
        # widen the estimate monotonically (always take the max observed)
        # so pass-through stays conservative; otherwise a single small
        # turn would shrink the estimate and let the next big turn slip
        # back over budget. Reset is via /reset or session boundary.
        if self.last_prompt_tokens and self._last_content_estimate:
            observed = max(0, self.last_prompt_tokens - self._last_content_estimate)
            if observed > self._observed_prompt_overhead:
                logger.info(
                    f"update_from_response: prompt overhead grew "
                    f"{self._observed_prompt_overhead} -> {observed} "
                    f"(prompt_tokens={self.last_prompt_tokens}, "
                    f"content_est={self._last_content_estimate})"
                )
                self._observed_prompt_overhead = observed

        logger.info(
            f"update_from_response: prompt_tokens={self.last_prompt_tokens}, "
            f"total={self.last_total_tokens}, budget={self._budget_tokens}, "
            f"observed_overhead={self._observed_prompt_overhead}"
        )

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Always returns True — this engine replaces the default compressor
        and must be called on every turn to assemble context.
        The plugin manages its own token budget inside compress().
        """
        return True

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> List[Dict[str, Any]]:
        """
        Main entry point called by Hermes each turn.
        1. Ingest new messages into event store
        2. Update execution state
        3. Index embeddings
        4. Assemble context package within token budget
        """
        logger.info(f"compress called: messages={len(messages)}, current_tokens={current_tokens}, budget={self._budget_tokens}")
        # Auto-rebind to Hermes' current session_id if it drifted out from
        # under us. Hermes rotates self.session_id on /new and on every
        # compression boundary; the plugin learns about compression boundaries
        # via on_session_start(boundary_reason="compression", ...) and about
        # /new via on_session_reset(). But there is at least one path where
        # Hermes assigns a brand-new session_id without firing either hook
        # (observed in agent.log: "Session reset" → next event under a new
        # id with no on_session_start). The thread-local logging context is
        # the authoritative source — set in run_agent.py:10614 right at the
        # top of every conversation turn.
        # A5: fallback chain for the authoritative session_id.
        #   1. hermes_logging._session_context (preferred — set at top of every turn)
        #   2. self._pending_session_id (cached from on_session_start)
        #   3. agent.session_id attribute (last-resort introspection)
        # Without a fallback, drift detection silently does nothing if Hermes
        # forgot to set the thread-local — and we lose state.
        hermes_sid = None
        try:
            from hermes_logging import _session_context as _sctx
            hermes_sid = getattr(_sctx, "session_id", None)
        except Exception:
            hermes_sid = None
        if not hermes_sid:
            hermes_sid = getattr(self, "_pending_session_id", None)
        if not hermes_sid:
            agent = getattr(self, "agent", None)
            hermes_sid = getattr(agent, "session_id", None) if agent else None
        if hermes_sid and hermes_sid != self.session_id:
            previous = self.session_id
            # Distinguish RESUME (existing session_id, e.g. after gateway
            # restart) from /new-style fresh binding. Resume must keep the
            # full messages buffer as legitimate tail and rejoin existing
            # events; fresh must anchor at the last user message.
            existing_status = None
            try:
                existing_status = self.store.session_exists(hermes_sid)
            except Exception as e:
                logger.warning(f"compress: session_exists check failed: {e}")

            if existing_status is not None and existing_status == "active":
                logger.info(
                    f"compress: session drift detected (RESUME). plugin={previous!r} "
                    f"hermes={hermes_sid!r} status={existing_status!r} — rejoining existing session."
                )
                self.session_id = hermes_sid
                # H: same logic as the on_session_start RESUME path —
                # force assembly on this turn so the LLM gets retrieved
                # context from the existing events.
                try:
                    if self.store.count_events(hermes_sid) > 0:
                        self._resume_context_fill_pending = True
                except Exception:
                    self._resume_context_fill_pending = True
                try:
                    self.store.reactivate_session(hermes_sid)
                except Exception as e:
                    logger.warning(f"compress: reactivate failed: {e}")
                last_seg = None
                try:
                    last_seg = self.store.latest_segment_id(hermes_sid)
                except Exception:
                    pass
                self.current_segment_id = last_seg or f"seg_{hermes_sid}_1"
                saved_state = None
                try:
                    saved_state = self.store.load_state(hermes_sid)
                except Exception:
                    pass
                if saved_state:
                    self._current_state = saved_state
                    self._current_state["session_id"] = hermes_sid
                    self._current_state["segment_id"] = self.current_segment_id
                else:
                    # A3: try state_history before falling back to default.
                    recovered = None
                    if last_seg is not None:
                        recovered = self._recover_state_from_history(hermes_sid, self.current_segment_id)
                    self._current_state = recovered or self._make_default_state(
                        hermes_sid, self.current_segment_id
                    )
                self._processed_msg_count = 0
                self._binding_origin_idx = 0
                self._last_enrichment_turn = -1
                if previous and previous != hermes_sid:
                    try:
                        conn = self.store._get_connection()
                        conn.execute(
                            "UPDATE sessions SET status='closed' WHERE session_id=? AND status='active'",
                            (previous,),
                        )
                        conn.commit(); conn.close()
                    except Exception as e:
                        logger.warning(f"compress: drift-resume close-previous failed: {e}")
            elif existing_status is not None:
                # Session exists but is not active (compressed/closed) — treat
                # as FRESH rebind. Rejoining a dead session causes the entire
                # message buffer to be re-ingested as "new" on every turn.
                logger.info(
                    f"compress: session drift detected (DEAD→FRESH). plugin={previous!r} "
                    f"hermes={hermes_sid!r} status={existing_status!r} — rebinding as fresh."
                )
                # Drift-with-events: if `previous` actually accumulated events
                # under us, treat the drift like an implicit compression
                # boundary and reassign them to the new session id so they
                # don't orphan. This is the second (silent) producer of
                # orphan sessions (see reproduce_lineage_loss.py scenario D).
                if previous and previous != hermes_sid:
                    try:
                        prev_event_count = self.store.count_events(previous)
                    except Exception:
                        prev_event_count = 0
                    if prev_event_count > 0:
                        try:
                            moved = self.store.reassign_session(previous, hermes_sid)
                            logger.info(
                                f"compress: drift carry-over (DEAD→FRESH) reassigned "
                                f"{moved} rows {previous} -> {hermes_sid}"
                            )
                        except Exception as e:
                            logger.warning(f"compress: drift reassign failed: {e}")
                self.session_id = hermes_sid
                self.current_segment_id = f"seg_{hermes_sid}_1"
                self._processed_msg_count = -1
                self._binding_origin_idx = -1
                self._current_state = self._make_default_state(
                    hermes_sid, self.current_segment_id
                )
                self._last_enrichment_turn = -1
            else:
                logger.info(
                    f"compress: session drift detected (FRESH). plugin={previous!r} "
                    f"hermes={hermes_sid!r} — rebinding."
                )
                # Drift-with-events: same logic as DEAD→FRESH branch above.
                # If `previous` has accumulated events, reassign them so we
                # don't orphan the work the user did before the drift. This
                # closes the third producer of orphan sessions (gateway
                # restart / silent thread-local rotation; see
                # reproduce_lineage_loss.py scenarios C and D).
                #
                # Cold-start recovery: when the plugin process just started,
                # `previous` is None — there's no in-memory binding to learn
                # from. Fall back to the most-recently-active session in the
                # DB and treat it as previous. This catches the gateway
                # restart case where Hermes hands us a fresh session id with
                # no on_session_start and our engine instance has no
                # in-memory state.
                effective_previous = previous
                if not effective_previous:
                    try:
                        effective_previous = self.store.find_last_active_session(
                            exclude_session_id=hermes_sid
                        )
                        if effective_previous:
                            logger.info(
                                f"compress: cold-start recovery — using DB's last "
                                f"active session {effective_previous!r} as previous"
                            )
                    except Exception as e:
                        logger.warning(f"compress: find_last_active_session failed: {e}")
                carried_over = False
                if effective_previous and effective_previous != hermes_sid:
                    try:
                        prev_event_count = self.store.count_events(effective_previous)
                    except Exception:
                        prev_event_count = 0
                    if prev_event_count > 0:
                        try:
                            moved = self.store.reassign_session(effective_previous, hermes_sid)
                            logger.info(
                                f"compress: drift carry-over (FRESH) reassigned "
                                f"{moved} rows {effective_previous} -> {hermes_sid}"
                            )
                            carried_over = True
                        except Exception as e:
                            logger.warning(f"compress: drift reassign failed: {e}")
                self.session_id = hermes_sid
                self.current_segment_id = f"seg_{hermes_sid}_1"
                # Sentinel: anchor binding origin to last user message on
                # next compress() pass so we don't ingest pre-/new content.
                self._processed_msg_count = -1
                self._binding_origin_idx = -1
                self._current_state = self._make_default_state(
                    hermes_sid, self.current_segment_id
                )
                self._last_enrichment_turn = -1
                # Lazy session row (A0): do not create a sessions row here
                # unless we just reassigned events into it (in which case
                # reassign_session already created the row atomically and
                # set previous to status='compressed').
                if not carried_over and effective_previous:
                    try:
                        conn = self.store._get_connection()
                        conn.execute(
                            "UPDATE sessions SET status='closed' WHERE session_id=? AND status='active'",
                            (effective_previous,),
                        )
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        logger.warning(f"compress: drift rebind close-previous failed: {e}")
        if not self.session_id:
            logger.warning("compress called but no session_id.")
            return messages

        # Resolve sentinel: decide between RESUME and FRESH binding.
        #
        # On /new or /reset Hermes hands us a buffer with a single fresh user
        # message (no assistant/tool turns yet). On gateway resume of a session
        # the plugin never saw, Hermes hands us the full prior conversation —
        # which always contains at least one assistant or tool message.
        #
        # If we see assistant/tool turns in the buffer on a fresh-bind, this is
        # actually a RESUME of a session whose id our DB doesn't know yet
        # (on_session_start treated it as fresh because session_exists() was
        # None). Treat the entire buffer as legitimate tail (anchor=0) so the
        # protected_tail isn't gutted.
        if getattr(self, "_binding_origin_idx", 0) == -1 or self._processed_msg_count == -1:
            has_prior_turns = any(
                (m or {}).get("role") in ("assistant", "tool") for m in messages
            )
            if has_prior_turns:
                anchor = 0
                logger.info(
                    f"binding RESUME-detected: buffer has assistant/tool turns "
                    f"(len={len(messages)}) — anchoring at 0, ingesting full tail."
                )
            else:
                anchor = 0
                for i in range(len(messages) - 1, -1, -1):
                    if (messages[i] or {}).get("role") == "user":
                        anchor = i
                        break
                logger.info(
                    f"binding anchored at messages_idx={anchor} "
                    f"(treating messages[0:{anchor}] as pre-binding history)"
                )
            self._binding_origin_idx = anchor
            self._processed_msg_count = anchor
        # NB: we do NOT mark _pending_compression here. The plugin calls compress()
        # on every turn purely to ingest messages and assemble a retrieval tail —
        # this is NOT a session boundary. The real compaction signal comes from
        # Hermes via on_session_start(boundary_reason="compression", old_session_id=...).

        # Per-session mutex for write ordering (spec: Component 3)
        lock = _get_session_lock(self.session_id)
        with lock:
            # ---
            # 1. Ingest new messages (write sequence: events → state → embeddings)
            # ---
            # If Hermes truncated history (e.g. inline compaction), our cursor is
            # past the end — restart from scratch on this batch.
            if len(messages) < self._processed_msg_count:
                logger.info(f"messages shrank ({len(messages)} < {self._processed_msg_count}) — resetting cursor")
                self._processed_msg_count = 0
            new_messages = messages[self._processed_msg_count:]
            base_idx = self._processed_msg_count
            segment_changed_this_turn = False
            if new_messages:
                logger.info(f"Processing {len(new_messages)} new messages")
                for offset, msg in enumerate(new_messages):
                    msg_idx = base_idx + offset
                    # Check segment boundary on user messages
                    if msg.get("role") == "user":
                        prev_seg = self.current_segment_id
                        self.current_segment_id = self.segmenter.handle_new_message(
                            self.session_id,
                            str(msg.get("content", "")),
                            self._current_state
                        )
                        if self.current_segment_id != prev_seg:
                            segment_changed_this_turn = True

                    events_data = parser.parse_message_to_event(msg, self.session_id, self.current_segment_id)
                    for sub_idx, ev in enumerate(events_data):
                        # Deterministic event_id: re-ingest of the same logical
                        # message produces the same id, so INSERT OR IGNORE
                        # silently dedupes (critical on session resume after
                        # gateway restart).
                        det_id = self.store.make_event_id(
                            ev["session_id"], msg_idx, ev["role"], ev["content"], sub_idx
                        )
                        # Step 1: Write to raw event store
                        event_id, inserted = self.store.add_event(
                            session_id=ev["session_id"], segment_id=ev["segment_id"],
                            event_type=ev["type"], role=ev["role"], content=ev["content"],
                            tool_name=ev["tool_name"], tool_input=ev["tool_input"],
                            token_estimate=ev["token_estimate"], event_id=det_id,
                        )
                        if not inserted:
                            # Already in DB from a prior process — skip state
                            # mutation, embedding, and graph-edge writes.
                            # Embedding (INSERT OR REPLACE) and graph edges
                            # (INSERT OR IGNORE) are individually idempotent,
                            # but reprocessing wastes CPU + an embedding call.
                            continue

                        # Step 2: Update execution state (deterministic parser)
                        self._update_state_from_event(ev, event_id)

                        # Step 3: Index embedding (can lag, non-blocking)
                        # Tool outputs over the configured threshold are embedded as
                        # head+tail summary, not the full text. The full content stays
                        # in events.content for fetch_event/expand_context.
                        text_for_embedding = ev["content"]
                        if ev["type"] == "tool_output" and text_for_embedding:
                            text_for_embedding = compressor.summarize_for_embedding(
                                text_for_embedding,
                                threshold_tokens=self.config.get("tool_output_compress_threshold_tokens", 500),
                                summary_tokens=self.config.get("tool_output_summary_tokens", 100),
                            )
                        if text_for_embedding and len(text_for_embedding.strip()) > 5:
                            try:
                                from .index import JINA_MODEL
                                self.indexer.add_embedding(
                                    event_id, ev["segment_id"], text_for_embedding,
                                    JINA_MODEL, ev["token_estimate"], ev["type"]
                                )
                            except Exception as emb_err:
                                logger.warning(f"Embedding failed for {event_id}: {emb_err}")

                        # Record execution graph edges
                        self._record_graph_edges(ev, event_id)

                self._processed_msg_count = len(messages)

            # Optional LLM enrichment (Stage 5.1).
            # Hybrid trigger: every N turns OR on segment boundary.
            self._maybe_enrich(segment_changed_this_turn)

            # Save state to DB (A4: execution_state + state_history written
            # atomically in one transaction so they cannot diverge on crash).
            self.store.commit_state(self.session_id, self._current_state)

        # ---
        # 1.5. Pass-through guard: if the incoming buffer is below the budget,
        # we don't need to rewrite anything. Returning `messages` as-is keeps
        # Hermes' session lineage stable (it rotates session_id whenever
        # compress() returns and any divergence is treated as compression).
        # We've already ingested + indexed above, so retrieval/recall on later
        # turns still benefits from this turn's content.
        #
        # Use the LARGER of:
        #   (a) content-only token estimate (messages body)
        #   (b) current_tokens reported by Hermes (includes system prompt,
        #       tool schemas, reasoning — the real payload sent to the LLM)
        # If either exceeds budget, we MUST assemble (cut the tail).
        # ---
        try:
            content_tokens = sum(
                self.prompt_builder.tokenizer(str(m.get("content", ""))) + 4
                for m in messages
            )
        except Exception:
            content_tokens = 0
        # G: remember content estimate so the next update_from_response can
        # compute observed overhead = real_prompt_tokens - this estimate.
        self._last_content_estimate = content_tokens
        # G: pass-through guard must factor in overhead that content_tokens
        # cannot see — system prompt, tool schemas, tool_call/tool_output
        # JSON wrappers, reasoning. Hermes' current_tokens estimate also
        # misses some of this (its preflight is computed before the final
        # call adds wrappers), so we add the empirically-observed delta
        # from update_from_response on top.
        effective_tokens = (
            max(content_tokens, current_tokens or 0) + self._observed_prompt_overhead
        )
        # H: resume context-fill — on the first turn after a session is
        # resumed (process restart, /resume, drift-RESUME), skip the
        # pass-through shortcut even when the buffer is small. The LLM
        # needs to see retrieved context from the accumulated events;
        # pass-through would feed it only the new user message.
        if self._resume_context_fill_pending:
            logger.info(
                f"Resume context-fill: forcing assembly for first post-resume turn "
                f"(content={content_tokens}, overhead={self._observed_prompt_overhead}, "
                f"effective={effective_tokens}, budget={self._budget_tokens})."
            )
        elif effective_tokens < self._budget_tokens:
            logger.info(
                f"Pass-through: content={content_tokens}, current={current_tokens}, "
                f"overhead={self._observed_prompt_overhead}, "
                f"effective={effective_tokens} < budget={self._budget_tokens} "
                f"— returning original messages ({len(messages)}) without assembly."
            )
            return messages
        else:
            logger.info(
                f"Assembly needed: content={content_tokens}, current={current_tokens}, "
                f"overhead={self._observed_prompt_overhead}, "
                f"effective={effective_tokens} >= budget={self._budget_tokens} "
                f"— assembling context."
            )

        # ---
        # 2. Context Assembly (outside lock — read-only)
        # ---
        current_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                # Skip our own [RETRIEVED CONTEXT] envelopes — they are not the
                # real user turn, just the retrieval block we emitted previously.
                stripped = router._RETRIEVED_PREFIX_RE.sub("", content).strip()
                if stripped:
                    current_user_message = stripped
                    break

        # B5: bootstrap cross-session probe. On the first ingest of a new
        # session, run a global semantic search using the first user message.
        # Results surface via [CROSS-SESSION CANDIDATES] so the LLM sees
        # potentially relevant past sessions without having to call
        # context_search itself. Done once per session; the embedding circuit
        # breaker (A10) still applies — if open, search_global returns nothing.
        if (
            not self._bootstrap_probe_done
            and self.config.get("bootstrap_probe_enabled", True)
            and current_user_message
            and self.session_id
        ):
            try:
                own_event_count = self.store.count_events(self.session_id)
            except Exception:
                own_event_count = 0
            first_turn_threshold = int(self.config.get("protected_tail_turns", 20) or 0)
            if 0 < own_event_count <= max(1, first_turn_threshold):
                self._bootstrap_probe_done = True
                try:
                    probe_k = int(self.config.get("bootstrap_probe_k", 3) or 3)
                    cross = self.indexer.search_global(
                        current_user_message, top_k=probe_k
                    ) or []
                    cross = [
                        r for r in cross
                        if r.get("session_id") and r["session_id"] != self.session_id
                    ]
                    self._cross_session_candidates = cross
                    if cross:
                        logger.info(
                            f"Bootstrap probe: found {len(cross)} cross-session "
                            f"candidates for new session {self.session_id}"
                        )
                except Exception as e:
                    logger.warning(f"Bootstrap probe failed: {e}")

        recent_events_db = self.store.get_recent_events(self.session_id, limit=self.config.get("protected_tail_turns", 20))

        router_input = router.RouterInput(
            message=current_user_message,
            execution_state=self._current_state,
            segment_id=self.current_segment_id,
            recent_events=recent_events_db,
            token_budget=self._budget_tokens,
            session_id=self.session_id
        )

        router_result = self.context_router.run(router_input)

        # Dedup: drop retrieved candidates that are already in protected tail
        # (otherwise the same event appears twice — once as RETRIEVED CONTEXT,
        # once in the recent turns).
        protected_ids = {ev.get("id") for ev in (recent_events_db or []) if ev.get("id")}
        deduped_candidates = [c for c in router_result.candidates_selected
                              if c.event_id not in protected_ids]
        if len(deduped_candidates) != len(router_result.candidates_selected):
            logger.info(
                f"Dedup: dropped {len(router_result.candidates_selected) - len(deduped_candidates)} "
                f"candidates already in protected tail"
            )

        # Post-dedup fallback: if dedup gutted the pool (KNN returned mostly
        # the same events that already sit in protected tail — typical right
        # after RESUME-ingest, when the current segment is tiny), do another
        # router pass that explicitly searches across the whole session and
        # union the results. Without this the retrieved block degrades to 1–4
        # chunks instead of the configured ~30% of the budget.
        min_candidates = int(self.config.get("router_min_candidates", 12) or 0)
        retrieved_budget_ratio = float(self.config.get("retrieved_budget_ratio", 0.30))
        retrieved_budget_estimate = int(self._budget_tokens * retrieved_budget_ratio)
        deduped_token_estimate = sum(
            self.prompt_builder.tokenizer(str(c.content or "")) for c in deduped_candidates
        ) if deduped_candidates else 0
        too_few = min_candidates > 0 and len(deduped_candidates) < min_candidates
        too_thin = (
            retrieved_budget_estimate > 0
            and deduped_token_estimate < int(retrieved_budget_estimate * 0.2)
        )
        if (too_few or too_thin) and self.current_segment_id != "all":
            logger.info(
                f"Dedup-fallback: pool too thin (n={len(deduped_candidates)}, "
                f"tokens={deduped_token_estimate}, budget≈{retrieved_budget_estimate}) — "
                f"retrying router across all segments of session {self.session_id}."
            )
            wide_input = router.RouterInput(
                message=current_user_message,
                execution_state=self._current_state,
                segment_id="all",
                recent_events=recent_events_db,
                token_budget=self._budget_tokens,
                session_id=self.session_id,
            )
            try:
                wide_result = self.context_router.run(wide_input)
                seen_ids = {c.event_id for c in deduped_candidates} | protected_ids
                added = [c for c in wide_result.candidates_selected
                         if c.event_id not in seen_ids]
                if added:
                    deduped_candidates = list(deduped_candidates) + added
                    deduped_candidates.sort(key=lambda x: x.score, reverse=True)
                    logger.info(
                        f"Dedup-fallback: +{len(added)} cross-segment candidates "
                        f"(total {len(deduped_candidates)})"
                    )
            except Exception as e:
                logger.warning(f"Dedup-fallback: wide router pass failed: {e}")

        # Clean-query fallback: when state.goal is from an OLD turn (the
        # user moved on but we haven't re-enriched yet), `router._build_query`
        # concatenates `goal + current_step + msg` and the search drifts
        # toward the stale goal — KNN returns the intersection of two
        # unrelated topics, which is small. If retrieved is still thin
        # AFTER the wide-segment pass, run one more router-bypass search
        # using ONLY the user's current message, segment='all'. This
        # restores the "fresh question, fresh search" behaviour without
        # touching query construction in the hot path.
        deduped_token_estimate = sum(
            self.prompt_builder.tokenizer(str(c.content or "")) for c in deduped_candidates
        ) if deduped_candidates else 0
        still_thin = (
            retrieved_budget_estimate > 0
            and deduped_token_estimate < int(retrieved_budget_estimate * 0.5)
            and bool(current_user_message)
        )
        if still_thin:
            logger.info(
                f"Clean-query fallback: pool still thin "
                f"(tokens={deduped_token_estimate} / budget≈{retrieved_budget_estimate}) — "
                f"retrying with raw user message only."
            )
            try:
                top_k_cfg = int(self.config.get("router_top_k", 0) or 0)
                top_k_arg = top_k_cfg if top_k_cfg > 0 else None
                raw = self.context_router.indexer.search(
                    current_user_message, self.session_id, "all", top_k=top_k_arg
                )
                seen_ids = {c.event_id for c in deduped_candidates} | protected_ids
                added: List[router.RetrievalCandidate] = []
                for r in raw:
                    eid = r.get("event_id")
                    if not eid or eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    conn = self.store._get_connection()
                    try:
                        row = conn.execute(
                            "SELECT content, type, timestamp FROM events WHERE id = ?",
                            (eid,),
                        ).fetchone()
                    finally:
                        conn.close()
                    if not row:
                        continue
                    added.append(router.RetrievalCandidate(
                        event_id=eid,
                        score=float(r.get("similarity", 0.0)),
                        content=row["content"] or "",
                        type=row["type"],
                        timestamp=row["timestamp"],
                        embedding_model_id=r.get("embedding_model_id", ""),
                        segment_id=r.get("segment_id", ""),
                    ))
                if added:
                    deduped_candidates = list(deduped_candidates) + added
                    deduped_candidates.sort(key=lambda x: x.score, reverse=True)
                    logger.info(
                        f"Clean-query fallback: +{len(added)} candidates "
                        f"(total {len(deduped_candidates)})"
                    )
            except Exception as e:
                logger.warning(f"Clean-query fallback failed: {e}")

        # Protected tail: never include messages from BEFORE this session was
        # bound. Hermes does not truncate `messages` on /new — its history may
        # still hold pre-/new turns, which would leak into the prompt as
        # "current session content". We use _processed_msg_count as the proxy
        # for how many messages belong to this session: anything past the
        # cursor (or freshly ingested in this call) is ours; older content was
        # already in `messages` when we (re)bound and is foreign.
        protected_tail_turns = self.config.get("protected_tail_turns", 20)
        # `_processed_msg_count` was just incremented by the loop above to the
        # current `len(messages)`; the FIRST message of the current binding
        # is at index `len(messages) - len(new_messages)` from this call OR
        # earlier ingestions accumulated under the same binding. Track the
        # binding origin separately so re-binds reset it.
        if not hasattr(self, "_binding_origin_idx"):
            self._binding_origin_idx = 0
        # If this call started with cursor=0 (fresh / drift / reset), origin
        # is "everything we just saw is ours" — but actually Hermes may have
        # delivered pre-existing messages too. Conservative: origin = first
        # index of the first user_message we ingested in THIS binding.
        own_messages = messages[self._binding_origin_idx:]
        # Pass ALL own messages — prompt_builder uses `protected_tail_turns`
        # as a FLOOR and will extend backwards into older turns to fill any
        # unused headroom. Slicing here would hide the older turns from it.
        recent_messages_for_prompt = own_messages

        # Measure the system prompt once (Hermes always passes it as the first
        # message). Without this, effective_budget == total_budget and the
        # collision check fires too late: the real prompt sent to the LLM is
        # `system + tools + content` (~37k tokens of overhead in observed runs).
        if self.prompt_builder.system_prompt_tokens == 0:
            sys_msgs = [m for m in messages if (m or {}).get("role") == "system"]
            if sys_msgs:
                sp_tokens = self.prompt_builder.tokenizer(
                    str(sys_msgs[0].get("content", ""))
                )
                # If Hermes also reports current_tokens, use the bigger of the
                # two — the gap between content_tokens and current_tokens is
                # the real overhead (system prompt + tool schemas + reasoning).
                if current_tokens and content_tokens:
                    overhead = max(0, int(current_tokens) - int(content_tokens))
                    sp_tokens = max(sp_tokens, overhead)
                if sp_tokens > 0:
                    self.prompt_builder.set_system_prompt_tokens(sp_tokens)
                    logger.info(f"system_prompt_tokens measured: {sp_tokens}")

        # B5: cross-session candidates from the bootstrap probe — rendered
        # once and then cleared so they don't repeat every turn.
        cross_session_payload = self._cross_session_candidates or None
        if cross_session_payload:
            self._cross_session_candidates = []
        final_messages = self.prompt_builder.build(
            execution_state=self._current_state,
            retrieved_candidates=deduped_candidates,
            recent_messages=recent_messages_for_prompt,
            current_user_message=current_user_message,
            goal_trail=self._build_goal_trail(),
            memory_access_hint=self._build_memory_access_hint(),
            checkpoint_block=self._consume_pending_checkpoint(),
            cross_session_candidates=cross_session_payload,
        )

        # Observability logging
        try:
            budget_info = {
                "total": self._budget_tokens,
                "system_prompt": self.prompt_builder.system_prompt_tokens,
                "execution_state": self.prompt_builder.tokenizer(json.dumps(self._current_state, ensure_ascii=False)),
                "protected_tail": sum(self.prompt_builder.tokenizer(str(m.get("content", ""))) for m in recent_messages_for_prompt),
                "retrieved": sum(self.prompt_builder.tokenizer(str(c.content)) for c in router_result.candidates_selected),
            }
            self.observability.log_turn(
                session_id=self.session_id,
                segment_id=self.current_segment_id,
                intent=router_result.intent,
                signals=router_result.signals,
                query=router_result.query,
                candidates_retrieved=router_result.candidates_retrieved,
                candidates_selected=len(deduped_candidates),
                budget=budget_info,
                top_chunks=[{
                    "event_id": c.event_id,
                    "type": c.type,
                    "score": c.score,
                    "score_breakdown": {
                        "dependency": getattr(c, "dependency_bonus", 0.0),
                    },
                } for c in deduped_candidates[:5]],
                mode=getattr(router_result, "mode", "general"),
            )
        except Exception as e:
            logger.error(f"Observability logging failed: {e}")

        # H: assembly path completed — clear the resume context-fill flag
        # so subsequent turns can pass-through normally if the buffer is
        # small. Done after assembly succeeded; if we crashed earlier the
        # flag stays True and the next compress() will retry assembly.
        self._resume_context_fill_pending = False

        logger.info(f"Context assembled. Output messages: {len(final_messages)}")
        return final_messages

    def _record_graph_edges(self, ev: Dict, event_id: str):
        """Record execution graph edges for tool calls and outputs."""
        if ev["type"] == "tool_call":
            self._last_tool_call_id = event_id
        elif ev["type"] == "tool_output":
            if hasattr(self, '_last_tool_call_id'):
                self.graph.add_edge(self._last_tool_call_id, event_id, "tool_output", self.session_id)
                del self._last_tool_call_id
            self._last_tool_output_id = event_id
        elif ev["type"] == "assistant_message" and hasattr(self, '_last_tool_output_id'):
            self.graph.add_edge(self._last_tool_output_id, event_id, "decision", self.session_id)
            del self._last_tool_output_id

    def _update_state_from_event(self, ev: Dict, event_id: str):
        """Deterministic state update from event (Component 3)."""
        if ev["type"] == "user_message":
            # Fix A: increment turn_count on every ingested user_message so
            # the live state reflects actual conversation depth even when
            # Hermes doesn't call update_from_response (observed in prod:
            # 6 user_messages in events but turn_count stuck at 2). The
            # update_from_response hook stays in place as an additional
            # signal — both paths increment but only one is guaranteed.
            content = ev.get("content", "").strip()
            if content:
                self._current_state["turn_count"] = (
                    int(self._current_state.get("turn_count") or 0) + 1
                )
                # Set goal from first user message (up to 500 chars)
                if not self._current_state.get("goal"):
                    self._current_state["goal"] = content[:500]
                # Always update current_step with latest user intent
                self._current_state["current_step"] = content[:300]
        elif ev["type"] == "tool_call":
            self._current_state["last_tool"] = ev["tool_name"]
        elif ev["type"] == "tool_output":
            content = ev["content"]
            if content:
                summary = content[:100] + "..." if len(content) > 100 else content
                self._current_state["last_tool_output_summary"] = summary
        elif ev["type"] == "assistant_message":
            content = ev.get("content", "")
            if content:
                self._current_state["current_step"] = content[:200]

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Decide inheritance strictly from Hermes-supplied kwargs.

        Hermes calls this hook in three cases:
          * compression rollover — kwargs has boundary_reason="compression" and
            old_session_id=<previous>. We reassign the accumulated tail to the
            new session id, keep current_segment_id and execution state.
          * /new or /reset — kwargs has no boundary_reason. We start a clean
            session: empty state, fresh segment, cursor reset.
          * first start of a process — kwargs empty, no previous session bound.
            Same as the /new path.

        Spec note: previously this engine self-rolled on every compress() by
        setting _pending_compression=True. That created a session-rotation loop
        where every turn produced a new session_id and re-ingested the entire
        history under the new id. The flag is gone now — only Hermes can
        declare a boundary.
        """
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        previous_session_id = self.session_id
        interface = kwargs.get("platform", "unknown")
        # A5: cache for compress() fallback if hermes_logging thread-local is
        # missing. Set early so even an early return path leaves a usable hint.
        self._pending_session_id = session_id
        logger.info(
            f"on_session_start: session_id={session_id} "
            f"boundary_reason={boundary_reason!r} old_session_id={old_session_id!r} "
            f"previous_session_id={previous_session_id!r}"
        )

        # Case A: Hermes-driven compression rollover. Always carry the tail
        # over — Hermes' compression threshold is independent of the
        # plugin's budget, so the old "skip if compress() just did
        # pass-through" optimisation produced orphan sessions in the wild
        # (see tests/diagnostic/reproduce_lineage_loss.py, scenarios B and
        # E). Trust Hermes when it says "this is a compression boundary".
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            # If we missed an intermediate transition (previous_session_id
            # diverged from old_session_id), our events still live under
            # previous_session_id. Reassign that tail too so it isn't
            # orphaned. This was the root of the "0 rows moved" anomaly
            # observed at 06:51:25 in agent.log on 2026-05-08.
            if previous_session_id and previous_session_id not in (old_session_id, session_id):
                try:
                    moved_prev = self.store.reassign_session(previous_session_id, session_id, interface=interface)
                    logger.info(
                        f"Compression boundary (catch-up): {previous_session_id} -> {session_id}, "
                        f"{moved_prev} rows moved"
                    )
                except Exception as e:
                    logger.error(f"Compression catch-up reassign failed: {e}")
            try:
                moved = self.store.reassign_session(old_session_id, session_id, interface=interface)
                logger.info(
                    f"Compression boundary: {old_session_id} -> {session_id}, {moved} rows moved"
                )
            except Exception as e:
                logger.error(f"Compression boundary reassign failed: {e}")
            self.session_id = session_id
            # Sessions row for `session_id` is now created atomically inside
            # reassign_session() (A1) — no separate create_session() call
            # needed.
            # current_segment_id rebased inside reassign_session via REPLACE on
            # segment_id strings, so the in-memory pointer must follow:
            if isinstance(self.current_segment_id, str) and old_session_id in self.current_segment_id:
                self.current_segment_id = self.current_segment_id.replace(old_session_id, session_id)
            self._current_state["session_id"] = session_id
            self._current_state["segment_id"] = self.current_segment_id
            # Hermes will hand us already-compacted messages from this point;
            # reset the cursor so the next compress() ingests them as new.
            # The shrink-detection branch in compress() handles the size diff.
            self._processed_msg_count = 0
            # Compression carries over events but the protected tail of
            # `messages` is the freshly compacted view from Hermes — those
            # are legitimately ours, so allow tail from index 0.
            self._binding_origin_idx = 0
            return

        # Case B: /new, /reset, or first start. Clean slate, no carry-over.
        # ... UNLESS this session_id already exists in our DB — that means
        # Hermes is RESUMING an existing session after a gateway restart and
        # the buffer it'll hand us in compress() is the legitimate tail of
        # that conversation, not pre-/new garbage. Distinguish by querying
        # the sessions table.
        existing_status = None
        try:
            existing_status = self.store.session_exists(session_id)
        except Exception as e:
            logger.warning(f"on_session_start: session_exists check failed: {e}")

        if existing_status is not None:
            # RESUME path: rejoin the existing session.
            self.session_id = session_id
            try:
                self.store.reactivate_session(session_id)
            except Exception as e:
                logger.warning(f"on_session_start: reactivate failed: {e}")
            last_seg = None
            try:
                last_seg = self.store.latest_segment_id(session_id)
            except Exception as e:
                logger.warning(f"on_session_start: latest_segment_id failed: {e}")
            self.current_segment_id = last_seg or f"seg_{session_id}_1"
            saved_state = None
            try:
                saved_state = self.store.load_state(session_id)
            except Exception as e:
                logger.warning(f"on_session_start: load_state failed: {e}")
            if saved_state:
                self._current_state = saved_state
                self._current_state["session_id"] = session_id
                self._current_state["segment_id"] = self.current_segment_id
                state_source = "execution_state"
            else:
                # A3: state may have been lost if the process died between
                # add_event (committed) and save_state (not). If state_history
                # has at least one snapshot, recover goal/current_step/intent
                # from there; otherwise fall back to a default.
                recovered = None
                if last_seg is not None:
                    recovered = self._recover_state_from_history(session_id, self.current_segment_id)
                if recovered:
                    self._current_state = recovered
                    state_source = "state_history"
                else:
                    self._current_state = self._make_default_state(session_id, self.current_segment_id)
                    state_source = "default"
            # The whole `messages` buffer Hermes will pass next is legitimate
            # tail; ingest from index 0 with deterministic IDs (dedupes on
            # re-write).
            self._processed_msg_count = 0
            self._binding_origin_idx = 0
            self._last_enrichment_turn = -1
            # B5: this is a resume, not a new session — suppress the
            # bootstrap probe so we don't surface cross-session candidates
            # that the user already knows about.
            self._bootstrap_probe_done = True
            self._cross_session_candidates = []
            # H: force the next compress() through assembly so the prompt
            # actually includes retrieved context from the DB; otherwise
            # a small post-restart buffer triggers pass-through and the
            # LLM sees only the new user message.
            try:
                if self.store.count_events(session_id) > 0:
                    self._resume_context_fill_pending = True
            except Exception:
                self._resume_context_fill_pending = True
            logger.info(
                f"Session RESUMED: {session_id} (was status={existing_status!r}, "
                f"segment={self.current_segment_id}, state_source={state_source})"
            )
            return

        self.session_id = session_id
        # Lazy session row (A0): no INSERT into `sessions` here — the row is
        # materialized inside add_event() when a real event arrives. This
        # prevents phantom sessions when the process dies before the first
        # message is processed.
        self.current_segment_id = f"seg_{session_id}_1"
        # B5: fresh session — let the bootstrap probe run on the first user
        # message.
        self._bootstrap_probe_done = False
        self._cross_session_candidates = []
        # Sentinel: pin binding origin on the next compress() call so we
        # ignore any messages Hermes may still be carrying in its history
        # buffer from before /new.
        self._processed_msg_count = -1
        self._binding_origin_idx = -1
        self._current_state = {
            "schema_version": 1, "session_id": session_id, "segment_id": self.current_segment_id,
            "goal": "", "current_step": "", "open_loops": [], "last_tool": None,
            "last_tool_output_summary": None, "decision_stack": [], "active_entities": [],
            "turn_count": 0,
            "enrichment": {"decision_summary": None, "intent_label": None, "topic_tags": []}
        }
        self._last_enrichment_turn = -1
        logger.info(f"Session started fresh: {session_id} on {interface}")

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        logger.info(f"Session ended: {session_id}")
        conn = self.store._get_connection()
        conn.execute("UPDATE sessions SET status='closed' WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()

    def on_session_reset(self) -> None:
        """Hermes invokes this from cli.reset_session_state() on /new and /reset.

        At this point Hermes has ALREADY assigned self.agent.session_id to the
        new id (cli.py:5249), but our plugin is still bound to the previous
        session_id. Hermes does NOT call on_session_start for this path —
        only the LCM-style hook in build_system_prompt fires, and only on
        first-prompt construction. So we cannot rely on on_session_start to
        repair state.

        Strategy: drop our binding entirely (session_id=None, fresh state,
        cursor=0). The very next compress() will refuse to ingest until a
        proper on_session_start binds us — which the build_system_prompt
        hook does for fresh sessions.
        """
        super().on_session_reset()
        previous = self.session_id
        logger.info(f"on_session_reset: clearing in-memory binding (was session_id={previous!r})")
        self.session_id = None
        self.current_segment_id = "seg_unbound"
        self._processed_msg_count = 0
        self._pending_session_id = None
        # B5: clear bootstrap probe flags so the next fresh session can probe.
        self._bootstrap_probe_done = False
        self._cross_session_candidates = []
        self._current_state = self._make_default_state(None, None)
        self._last_enrichment_turn = -1
        # A6: mark previous session 'closed' if it was still active in the DB,
        # also bump last_active so the row reflects when /reset happened
        # (otherwise stale `last_active` confuses any cross-session "last
        # touched" lookup).
        if previous:
            try:
                import datetime as _dt
                now_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
                conn = self.store._get_connection()
                conn.execute(
                    "UPDATE sessions SET status='closed', last_active=? "
                    "WHERE session_id=? AND status='active'",
                    (now_iso, previous),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"on_session_reset: failed to close previous session {previous}: {e}")

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine_name"] = self.name
        status["db_path"] = self.store.db_path
        return status

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.tools.get_tool_schemas()

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        result = self.tools.handle_tool_call(name, args, self.session_id)
        try:
            memory_tools = set(self.config.get(
                "memory_tool_names",
                ["context_search", "fetch_event", "expand_context",
                 "list_segments", "get_goal_history"]
            ))
            threshold = int(self.config.get("checkpoint_after_n_memory_calls", 5) or 0)
            if name in memory_tools:
                self._consecutive_memory_calls += 1
                if threshold > 0 and self._consecutive_memory_calls >= threshold:
                    self._pending_checkpoint = self._format_checkpoint_block(
                        self._consecutive_memory_calls
                    )
            else:
                if self._consecutive_memory_calls:
                    logger.info(
                        f"memory-call counter reset by tool {name} "
                        f"(was {self._consecutive_memory_calls})"
                    )
                self._consecutive_memory_calls = 0
        except Exception as _e:
            logger.warning(f"memory-counter accounting failed: {_e}")
        return result

    def _format_checkpoint_block(self, consecutive: int) -> str:
        """Build the [CHECKPOINT] block injected when the agent is in a
        memory-call loop without producing useful work."""
        cur_goal = (self._current_state or {}).get("goal") or "<none>"
        history = []
        try:
            history = self.store.get_recent_unique_goals(self.session_id, limit=5)
        except Exception:
            history = []
        prior = ""
        if len(history) >= 2:
            prior = history[-2].get("goal") or ""
        elif history:
            prior = history[0].get("goal") or ""
        lines = [
            f"[CHECKPOINT] Ты сделал {consecutive} memory-вызовов подряд "
            f"и пока не вернулся к работе.",
            f"Текущая цель: \"{cur_goal}\"",
        ]
        if prior and prior != cur_goal:
            lines.append(f"Предыдущая цель: \"{prior}\"")
        lines.append(
            "Если ты всё ещё работаешь над текущей целью — продолжай. "
            "Если потерял нить — вызови get_goal_history(20) или ответь "
            "пользователю, что нужно уточнение."
        )
        return "\n".join(lines)

    def _consume_pending_checkpoint(self) -> Optional[str]:
        block = self._pending_checkpoint
        self._pending_checkpoint = None
        return block

    def _build_goal_trail(self) -> List[Dict[str, Any]]:
        size = int(self.config.get("goal_trail_size", 3) or 0)
        if size <= 0:
            return []
        try:
            return self.store.get_recent_unique_goals(self.session_id, limit=size)
        except Exception as e:
            logger.warning(f"_build_goal_trail failed: {e}")
            return []

    def _build_memory_access_hint(self) -> Optional[str]:
        """B4 + B1: adaptive memory hint.

        - Disabled entirely when `memory_access_hint_enabled=False`.
        - Otherwise the hint length scales with how much context lives
          outside the active window. The threshold is `protected_tail_turns`
          (the same parameter prompt_builder uses for the floor of the
          protected tail) — when total events ≤ that floor, every event
          fits in the active window and the long hint is noise. Above the
          threshold we emit the full hint AND a dynamic [MEMORY STATS]
          block so the LLM sees concrete numbers (B1).
        """
        if not self.config.get("memory_access_hint_enabled", True):
            return None
        threshold = int(self.config.get("protected_tail_turns", 20) or 0)
        if not self.session_id:
            # No session bound yet — short form is safest (full hint name-drops
            # tools that may not have run any indexing).
            return MEMORY_ACCESS_HINT_SHORT
        try:
            stats = self.store.get_session_stats(self.session_id)
        except Exception as e:
            logger.warning(f"_build_memory_access_hint: stats fetch failed: {e}")
            return MEMORY_ACCESS_HINT
        total_events = stats.get("total_events", 0) if stats else 0
        if total_events <= threshold:
            return MEMORY_ACCESS_HINT_SHORT
        stats_block = (
            f"\n\n[MEMORY STATS] В сессии {total_events} событий, "
            f"{stats.get('total_segments', 0)} сегментов; "
            f"в активном окне ~{threshold} ходов. "
            f"Остальное доступно через memory tools."
        )
        if stats.get("compressed_ancestors_count"):
            stats_block += (
                f" Сессия была скомпрессирована "
                f"{stats['compressed_ancestors_count']} раз — "
                f"раньшие фрагменты по-прежнему ищутся (segment='all')."
            )
        return MEMORY_ACCESS_HINT + stats_block

    def _make_default_state(self, session_id: str, segment_id: str) -> Dict[str, Any]:
        """Default empty execution_state. Extracted so resume paths share the
        same shape (avoids drift between three near-duplicate dict literals)."""
        return {
            "schema_version": 1,
            "session_id": session_id,
            "segment_id": segment_id,
            "goal": "",
            "current_step": "",
            "open_loops": [],
            "last_tool": None,
            "last_tool_output_summary": None,
            "decision_stack": [],
            "active_entities": [],
            "turn_count": 0,
            "enrichment": {"decision_summary": None, "intent_label": None, "topic_tags": []},
        }

    def _recover_state_from_history(self, session_id: str, segment_id: str) -> Optional[Dict[str, Any]]:
        """A3: when load_state() returns None but events exist, the previous
        process likely died between add_event() (committed) and save_state()
        (not). state_history is append-only and updated via the enricher;
        the most recent row is our best partial recovery — we know goal,
        current_step, intent_label even if open_loops/decision_stack were lost.

        Returns a state dict (with `_recovered_from_history=True` marker) or
        None if no history exists either.
        """
        try:
            history = self.store.get_state_history(session_id, limit=1)
        except Exception as e:
            logger.warning(f"_recover_state_from_history: read failed: {e}")
            return None
        if not history:
            return None
        last = history[-1]
        recovered = self._make_default_state(session_id, segment_id)
        recovered["goal"] = last.get("goal") or ""
        recovered["current_step"] = last.get("current_step") or ""
        recovered["enrichment"]["intent_label"] = last.get("intent_label")
        recovered["_recovered_from_history"] = True
        recovered["_recovered_from_timestamp"] = last.get("timestamp")
        logger.info(
            f"State recovered from state_history for {session_id}: "
            f"goal={recovered['goal']!r}, ts={recovered.get('_recovered_from_timestamp')}"
        )
        return recovered

    def rebuild_state_from_events(self, session_id: str) -> Dict[str, Any]:
        """Deterministic reducer over structured events only (spec v1.1 override).

        Only ``tool_call`` and ``tool_output`` events drive state — conversational
        text (user_message, assistant_message) is intentionally ignored, because
        it is not structured enough for a reliable reducer. State for prose is
        the LLM's job at runtime, not ours offline.
        """
        conn = self.store._get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, type, role, content, tool_name, tool_input, timestamp "
                "FROM events WHERE session_id = ? AND type IN ('tool_call', 'tool_output') "
                "ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = cursor.fetchall()

            state = {
                "schema_version": 1, "session_id": session_id, "segment_id": "seg_1",
                "goal": "", "current_step": "", "open_loops": [], "last_tool": None,
                "last_tool_output_summary": None, "decision_stack": [], "active_entities": [],
                "turn_count": 0,
                "enrichment": {"decision_summary": None, "intent_label": None, "topic_tags": []}
            }

            for row in rows:
                event_type = row["type"]
                content = row["content"]
                tool_name = row["tool_name"]

                if event_type == "tool_call":
                    state["last_tool"] = tool_name
                    state["turn_count"] += 1
                elif event_type == "tool_output" and content:
                    state["last_tool_output_summary"] = (
                        content[:100] + "..." if len(content) > 100 else content
                    )

            return state
        except Exception as e:
            logger.error(f"Failed to rebuild state from events for session {session_id}: {e}")
            return {
                "schema_version": 1, "session_id": session_id, "segment_id": "seg_1",
                "goal": "", "current_step": "", "open_loops": [], "last_tool": None,
                "last_tool_output_summary": None, "decision_stack": [], "active_entities": [],
                "turn_count": 0,
                "enrichment": {"decision_summary": None, "intent_label": None, "topic_tags": []}
            }
        finally:
            conn.close()
