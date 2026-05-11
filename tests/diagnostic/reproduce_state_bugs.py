"""Reproducers for four observable bugs in hermes-mneme that surfaced after
the lineage-loss fix landed. Each scenario runs against a fresh temporary
SQLite DB, replays a small slice of the Hermes hook sequence, and asserts
a specific invariant.

Bugs covered
------------
A. ``turn_count`` is incremented only by ``update_from_response`` (a hook
   Hermes calls explicitly after each LLM round-trip). When Hermes is slow
   to call it — or doesn't call it at all — ``turn_count`` lags by orders
   of magnitude (observed in the real DB: 2 vs 153 tool_calls).

B. ``commit_state`` skips ``state_history`` appends when goal +
   current_step + intent_label match the previous row. In real sessions
   the goal is set from the first user message and never changes, so
   ``state_history`` for the whole session stays empty:
     - ``get_goal_history`` always returns ``[]``,
     - ``_recover_state_from_history`` (the A3 crash-recovery path) has
       nothing to recover from.

C. ``context_search`` returns ``{"error": "Query is empty"}`` with no
   diagnostic logging when ``args["query"]`` is falsy. The user has no way
   to know whether the LLM sent ``""`` / ``None`` / a different key.

D. ``reassign_session(old → new)`` writes a lineage row unconditionally.
   When Hermes signals compression in both directions (X → Y, then Y → X
   later, observed in prod), we get cycles in ``session_lineage``. Not
   data-loss, but structural garbage that confuses traversal.

How to run
----------
    ~/.hermes/hermes-agent/venv/bin/python3 \\
        -m tests.diagnostic.reproduce_state_bugs

Exit 0 = all invariants hold (no bug). Exit 1 = at least one is broken.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.util
import io
import logging
import os
import sqlite3
import sys
import tempfile
import textwrap
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Stub Hermes-side imports (same machinery as reproduce_lineage_loss) ------

def _install_hermes_stubs() -> Any:
    agent_pkg = types.ModuleType("agent")
    ctx_mod = types.ModuleType("agent.context_engine")

    class ContextEngine:
        def __init__(self) -> None:
            self.agent = None

        def on_session_reset(self) -> None:
            pass

        def get_status(self) -> Dict[str, Any]:
            return {}

    ctx_mod.ContextEngine = ContextEngine
    agent_pkg.context_engine = ctx_mod
    sys.modules.setdefault("agent", agent_pkg)
    sys.modules["agent.context_engine"] = ctx_mod

    hl_mod = types.ModuleType("hermes_logging")

    class _Sctx:
        session_id: Optional[str] = None

    hl_mod._session_context = _Sctx()
    sys.modules["hermes_logging"] = hl_mod
    return hl_mod._session_context


PLUGIN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_DIR.parent))
_SCTX = _install_hermes_stubs()

_pkg_name = "hermes_mneme"
if _pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _pkg_name,
        str(PLUGIN_DIR / "__init__.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_pkg_name] = mod
    spec.loader.exec_module(mod)

engine_module = importlib.import_module("hermes_mneme.engine")
config_module = importlib.import_module("hermes_mneme.config")
CustomRouterContextEngine = engine_module.CustomRouterContextEngine


# --- Helpers ------------------------------------------------------------------

INVARIANT_FAILURES: List[str] = []


def expect(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    suffix = f"  — {detail}" if (detail and not cond) else ""
    print(f"  [{status}] {name}{suffix}")
    if not cond:
        INVARIANT_FAILURES.append(f"{name}: {detail}")


def make_engine(db_path: str) -> Any:
    cfg = config_module.PluginConfig({
        "active_window_tokens": 200,
        "context_window_usage_percent": 0.70,
        "protected_tail_turns": 4,
        "llm_enrichment_enabled": False,
        "memory_access_hint_enabled": False,
        "bootstrap_probe_enabled": False,
        "segmentation_enabled": False,
    })
    engine = CustomRouterContextEngine(db_path=db_path, config=cfg)
    # Skip the network embedding endpoint.
    engine.indexer.get_embedding = lambda *_a, **_kw: None
    return engine


def msg(role: str, content: str, tool_calls: Optional[List[Dict]] = None) -> Dict:
    out: Dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def set_sctx(session_id: Optional[str]) -> None:
    _SCTX.session_id = session_id


def run_scenario(name: str, fn) -> None:
    print("\n" + "=" * 76)
    print(f"SCENARIO: {name}")
    print("=" * 76)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "plugin.db")
        try:
            fn(db_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            INVARIANT_FAILURES.append(f"{name}: raised {exc!r}")


# --- Scenario A — turn_count must track real user turns ----------------------

def scenario_A_turn_count(db_path: str) -> None:
    """Simulate six user turns. Each compress() ingests a fresh user_message
    (the bookkeeping that the engine actually does in production). Hermes is
    intentionally NOT calling ``update_from_response`` — that's the failure
    mode we observed: 6 user turns landed in events, but turn_count stayed
    at 2 because Hermes only called update_from_response twice.

    The invariant: turn_count in execution_state must reflect the number of
    user_message events ingested, regardless of whether Hermes calls
    update_from_response.
    """
    engine = make_engine(db_path)
    S = "20260511_120000_aaaaaa"
    set_sctx(S)
    engine.on_session_start(S)

    n_user_turns = 6
    # Realistic Hermes hand-off: the buffer grows monotonically across turns.
    buffer: List[Dict[str, Any]] = []
    for i in range(n_user_turns):
        buffer.append(msg("user", f"turn {i}"))
        buffer.append(msg("assistant", f"reply {i}"))
        engine.compress(list(buffer), current_tokens=20)

    # Read what get_execution_state would surface.
    state = engine.store.load_state(S) or engine._current_state
    n_user_events = engine.store.count_events(S)
    print(f"  ingested events for {S}: {n_user_events}")
    print(f"  state.turn_count: {state.get('turn_count')!r}")

    # The plugin tracks user turns — turn_count should be ≥ n_user_turns.
    expect(
        "turn_count >= number of user turns",
        int(state.get("turn_count") or 0) >= n_user_turns,
        f"turn_count={state.get('turn_count')!r}, expected >= {n_user_turns}",
    )


# --- Scenario B — state_history keepalive ------------------------------------

def scenario_B_state_history_keepalive(db_path: str) -> None:
    """Simulate a long session where the user opens with one task ("проверить
    новости по крипте") and then drills into many sub-questions. The plugin's
    ``goal`` field stays pinned to the first user message (that's how the
    deterministic state reducer works), so ``commit_state``'s
    "skip-if-identical" check fires on every subsequent turn and
    ``state_history`` ends up empty.

    Invariant: by the end of a long session ``state_history`` must contain
    AT LEAST one snapshot per noticeable execution step (e.g. each tool_call
    burst), so:
      - get_goal_history is usable as a memory tool,
      - A3 crash-recovery from state_history actually has something to recover.
    """
    engine = make_engine(db_path)
    S = "20260511_130000_aaaaaa"
    set_sctx(S)
    engine.on_session_start(S)

    # First turn: sets goal.
    engine.compress(
        [msg("user", "long pinned goal: проверить новости по крипте"),
         msg("assistant", "ok, starting")],
        current_tokens=20,
    )

    # Eight follow-up turns. Each one is a tool_call + tool_output cycle.
    # This is the typical shape of one assistant turn in real sessions.
    for i in range(8):
        engine.compress(
            [
                msg("user", f"follow-up question #{i}"),
                msg("assistant", "calling tool"),
                # In real sessions tool_call + tool_output are stored as
                # separate events. We approximate via two assistant turns
                # with the right role names; parser.parse_message_to_event
                # accepts the "tool" role for outputs.
                {"role": "tool", "content": f"tool result #{i}",
                 "tool_call_id": f"tc_{i}"},
            ],
            current_tokens=20,
        )

    n_history = engine.store._get_connection().execute(
        "SELECT COUNT(*) FROM state_history WHERE session_id=?", (S,)
    ).fetchone()[0]
    n_events = engine.store.count_events(S)
    print(f"  events ingested: {n_events}")
    print(f"  state_history rows: {n_history}")

    # We expect at least 2 history rows (initial goal + at least one keepalive
    # snapshot) for a 9-turn session. The exact policy is plugin-internal, but
    # zero is unambiguously wrong.
    expect(
        "state_history not empty after a long session",
        n_history >= 2,
        f"got {n_history} rows for {n_events} events across 9 turns",
    )


# --- Scenario C — context_search must log on empty query ---------------------

def scenario_C_context_search_diagnostics(db_path: str) -> None:
    """Calling context_search with an empty query is allowed (legacy callers
    might do it). But the plugin's response is opaque: ``{"error": "Query is
    empty"}`` with no log line saying which args arrived. In production this
    means the user/engineer can't tell whether the LLM passed ``""``,
    ``None``, or used a different key like ``q``.

    Invariant: when context_search rejects an empty query, the plugin must
    log the full args dict at WARNING level so a downstream operator can
    diagnose.
    """
    engine = make_engine(db_path)
    S = "20260511_140000_aaaaaa"
    set_sctx(S)
    engine.on_session_start(S)

    # Capture WARNING+ log output from the plugin.
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        # Three flavours of "empty" the LLM might send:
        for args in [{}, {"query": ""}, {"query": None, "k": 5}]:
            engine.handle_tool_call("context_search", args)
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    log_text = captured.getvalue()
    print("  captured log lines containing 'context_search':")
    for line in log_text.splitlines():
        if "context_search" in line.lower():
            print(f"    {line}")

    # Expect at least one WARNING-level line about the empty-query call AND
    # the full args dict represented in it.
    has_warning = "context_search" in log_text and "args" in log_text.lower()
    expect(
        "context_search empty-query is logged with args",
        has_warning,
        "no diagnostic log line found",
    )


# --- Scenario D — lineage cycles must be rejected ----------------------------

def scenario_D_lineage_cycles(db_path: str) -> None:
    """Simulate the cycle observed in the real DB:
        on_session_start(compression, old=A, new=B)  → lineage A → B
        on_session_start(compression, old=B, new=A)  → lineage B → A  (cycle!)

    The plugin shouldn't write the second lineage row when the reverse edge
    already exists. Reassign in the cycle direction is, by definition, a
    rollback that needs no migration (the events are already where they
    belong after the first reassign).
    """
    engine = make_engine(db_path)
    A = "20260511_150000_aaaaaa"
    B = "20260511_150500_bbbbbb"

    # Phase 1: build session A with events, then compress A → B.
    set_sctx(A)
    engine.on_session_start(A)
    engine.compress(
        [msg("user", "real content " + "x" * 500),
         msg("assistant", "reply " + "y" * 500)],
        current_tokens=20_000,
    )
    set_sctx(B)
    engine.on_session_start(B, boundary_reason="compression", old_session_id=A)

    # Phase 2: Hermes (for whatever reason) tells us compress back — old=B,
    # new=A. This is the cycle.
    set_sctx(A)
    engine.on_session_start(A, boundary_reason="compression", old_session_id=B)

    conn = engine.store._get_connection()
    rows = conn.execute(
        "SELECT old_session_id, new_session_id FROM session_lineage"
    ).fetchall()
    conn.close()
    print(f"  lineage rows: {[(r[0], r[1]) for r in rows]}")

    pairs = {(r[0], r[1]) for r in rows}
    # The bug: both (A, B) and (B, A) are present.
    expect(
        "no reverse-edge lineage cycle",
        not ((A, B) in pairs and (B, A) in pairs),
        f"got cycle in lineage: {sorted(pairs)}",
    )


# --- Scenario E — context_search(session='current') must search lineage chain

def scenario_E_current_session_lineage(db_path: str) -> None:
    """In a long conversation, hermes-mneme creates many compression hops:
        S0 → S1 → S2 → ... → Sn
    reassign_session migrates all events to the latest node, so old nodes
    have 0 events. ``self.session_id`` (and therefore tools.session_id) is
    the latest node — BUT when the LLM calls context_search right after a
    compression burst, ``self.session_id`` can still be a middle node of
    the chain whose events have already been reassigned forward.

    Bug observed in prod: context_search(session='current') returned [] for
    a 1100-event conversation because session_id pointed to a pruned node
    of the chain.

    Invariant: ``session='current'`` should find events anywhere along the
    current conversation's lineage chain — i.e. ancestors-and-descendants
    of self.session_id — not just under the literal current id.
    """
    engine = make_engine(db_path)
    A = "20260511_160000_aaaaaa"
    B = "20260511_160500_bbbbbb"
    C = "20260511_161000_cccccc"

    # Build a chain A → B → C with content.
    set_sctx(A)
    engine.on_session_start(A)
    engine.compress([msg("user", "talked about pipeline"),
                     msg("assistant", "pipeline reply with big content " + "x" * 500)],
                    current_tokens=20_000)

    set_sctx(B)
    engine.on_session_start(B, boundary_reason="compression", old_session_id=A)
    engine.compress([msg("user", "more pipeline questions"),
                     msg("assistant", "more pipeline answers " + "y" * 500)],
                    current_tokens=20_000)

    set_sctx(C)
    engine.on_session_start(C, boundary_reason="compression", old_session_id=B)
    engine.compress([msg("user", "final pipeline turn"),
                     msg("assistant", "final pipeline reply " + "z" * 500)],
                    current_tokens=20_000)

    # Mock indexer to record which session_ids the search is asked to scan.
    # We don't need real vector retrieval — we just need to verify that
    # context_search(session='current') correctly broadens the scan beyond
    # the literal current session_id and includes the rest of the lineage
    # chain (where the events actually live after reassign).
    seen_session_ids: List[Any] = []

    def fake_search(query, session_id, segment_id=None, top_k=5):
        seen_session_ids.append(("search", session_id))
        return []

    def fake_search_sessions(query, session_ids, top_k=5, segment_id=None):
        seen_session_ids.append(("search_sessions", list(session_ids)))
        return []

    engine.indexer.search = fake_search
    engine.indexer.search_sessions = fake_search_sessions

    engine.handle_tool_call(
        "context_search",
        {"query": "pipeline", "session": "current", "k": 5},
    )

    print(f"  current self.session_id: {engine.session_id}")
    print(f"  index calls observed: {seen_session_ids}")

    # The expected behaviour after the fix: session='current' broadens to
    # the lineage chain. So the indexer must be called with a list of
    # session_ids that includes either ancestors or descendants of
    # self.session_id — not just the literal current one.
    saw_chain_search = False
    for kind, ids in seen_session_ids:
        if kind == "search_sessions" and isinstance(ids, list) and len(ids) > 1:
            saw_chain_search = True
            break
        if kind == "search" and isinstance(ids, str) and ids != engine.session_id:
            saw_chain_search = True
            break
    expect(
        "context_search(session='current') broadens to lineage chain",
        saw_chain_search,
        f"index was called with only the literal current session id: "
        f"{seen_session_ids}",
    )


# --- Scenario F — enricher must survive truncated JSON ----------------------

def scenario_F_truncated_json(db_path: str) -> None:
    """In long sessions the enricher LLM can return a JSON object that's
    been truncated mid-string (LLM output limit hit before closing brace).
    Observed log:
        WARNING enricher: failed to parse JSON from response:
        '{"intent_label": "Запрос на ... рын'
    The enricher already swallows the parse error, but the caller gets
    EnrichmentResult(raw=text) with every field blank, which:
      - logs noise on every enrich tick,
      - downstream consumers see no intent_label / topic_tags at all.

    Invariant: _safe_parse_json must recover at least a usable
    intent_label from a truncated-at-string JSON like the one above.
    """
    enrichment_mod = importlib.import_module("hermes_mneme.enrichment")
    truncated = '{"intent_label": "Запрос на глубокий стратегический анализ рын'
    parsed = enrichment_mod.LLMEnricher._safe_parse_json(truncated)
    print(f"  parsed: {parsed!r}")
    expect(
        "truncated JSON yields a usable partial result",
        isinstance(parsed, dict) and bool((parsed or {}).get("intent_label")),
        f"got {parsed!r}",
    )


# --- Scenario G — pass-through must account for prompt overhead -------------

def scenario_G_pass_through_overhead(db_path: str) -> None:
    """In production we see this sequence (from agent.log on 2026-05-11):

        compress: content=172196, current=159085, budget=179200
        → Pass-through (content < budget)
        update_from_response: prompt_tokens=212737 budget=179200

    The plugin's pass-through guard compares ``content_tokens`` to
    ``self._budget_tokens``, but the actual prompt sent to the LLM is
    ``content + system_prompt + tool_schemas + reasoning + tool_call/output
    JSON wrappers`` — about 40k more in real Hermes installs. So the plugin
    thinks "7k headroom", reality is "38k over budget". On a smaller model
    (200k context window with a 70 % budget = 140k) this overflows and the
    LLM truncates old messages.

    Invariant: after the plugin has observed a real prompt size via
    update_from_response, subsequent pass-through decisions must factor in
    the observed overhead and STOP pass-through when the projected real
    prompt would exceed the budget.
    """
    engine = make_engine(db_path)
    # Use real-life budget shape (smaller, easier to reason about).
    engine._budget_tokens = 18_000
    engine.prompt_builder.total_budget = 18_000

    S = "20260511_180000_aaaaaa"
    set_sctx(S)
    engine.on_session_start(S)

    # Turn 1: smaller messages, plugin passes through.
    engine.compress(
        [msg("user", "hello"), msg("assistant", "hi back")],
        current_tokens=1_000,
    )

    # Hermes reports the actual prompt size — overhead was +4000 tokens
    # (system prompt + tool schemas).
    engine.update_from_response({"prompt_tokens": 5_000, "completion_tokens": 100,
                                 "total_tokens": 5_100})

    # Turn 2: content estimate puts us under budget on paper (16k < 18k),
    # but with the +4000 overhead observed last turn, the real prompt would
    # be ~20k → over budget. The plugin must NOT pass through here.
    big_user = msg("user", "X" * 30_000)  # ~7500 tokens at cl100k
    big_asst = msg("assistant", "Y" * 30_000)
    set_sctx(S)
    result = engine.compress(
        [msg("user", "hello"), msg("assistant", "hi back"), big_user, big_asst],
        current_tokens=16_000,
    )

    # The plugin returns `messages` unchanged on pass-through. After
    # overhead-aware fix, it must assemble (different list / smaller).
    passed_through = len(result) == 4 and result[-1].get("content", "").startswith("YYYYYY")
    print(f"  turn-2 returned len={len(result)}, last looks like raw assistant: {passed_through}")
    expect(
        "pass-through suppressed when projected prompt > budget",
        not passed_through,
        f"plugin returned original messages despite projected overflow "
        f"(content+observed_overhead > budget={engine._budget_tokens})",
    )


# --- Scenario H — resume must rebuild context from DB -----------------------

def scenario_H_resume_context_fill(db_path: str) -> None:
    """Reported in prod (2026-05-11): after Hermes restart the user continued
    a session that already had ~200k tokens of events in the DB. The new
    process gave the plugin a tiny `messages` buffer (just the user's new
    question). The plugin saw content_tokens ≪ budget → pass-through →
    returned messages unchanged → the LLM got 25k of total context instead
    of the expected retrieved-context blob.

    Invariant: the first compress() after a session is resumed MUST exercise
    the assembly path (router + prompt_builder) so accumulated events from
    the DB are actually retrieved into the prompt. pass-through is only
    safe AFTER we've at least once shown the LLM the historical tail.
    """
    # Phase 1 — build a session with content, then "shut down" the engine.
    # Use a big budget so each compress() pass-throughs, leaving events
    # accumulating in the DB. Use a growing buffer (Hermes-realistic) so
    # _processed_msg_count actually advances per turn.
    engine = make_engine(db_path)
    engine._budget_tokens = 200_000
    engine.prompt_builder.total_budget = 200_000
    engine._observed_prompt_overhead = 0  # avoid early assembly during seed phase
    S = "20260511_190000_aaaaaa"
    set_sctx(S)
    engine.on_session_start(S)
    buffer: List[Dict[str, Any]] = []
    for i in range(30):
        buffer.append(msg("user", f"question {i} " + "u" * 200))
        buffer.append(msg("assistant", f"answer {i} " + "a" * 600))
        engine.compress(list(buffer), current_tokens=20_000)
    events_before = engine.store.count_events(engine.session_id)
    print(f"  events under {engine.session_id} after phase 1: {events_before}")
    del engine

    # Phase 2 — new engine instance picks up the same DB (process restart).
    engine = make_engine(db_path)
    engine._budget_tokens = 200_000
    engine.prompt_builder.total_budget = 200_000
    engine._observed_prompt_overhead = 0

    # Mock the router so we can detect whether assembly path was taken.
    # If the plugin does pass-through, router.run is never called.
    router_calls: List[Any] = []
    original_run = engine.context_router.run
    def spy_run(router_input):
        router_calls.append(router_input)
        return original_run(router_input)
    engine.context_router.run = spy_run

    # Hermes hands the plugin the standard resume signal — the same
    # session_id it had before restart, no boundary_reason.
    set_sctx(S)
    engine.on_session_start(S)

    # User asks one new question. Hermes' buffer is small (just the
    # historical replay it remembers + the new turn), well under the
    # plugin's budget. With current code, this passes through.
    engine.compress(
        [msg("user", "what did we decide about i=5?")],
        current_tokens=2_000,
    )

    print(f"  router.run() calls during resume turn: {len(router_calls)}")
    print(f"  events in DB at resume: {engine.store.count_events(engine.session_id)}")

    expect(
        "first compress() after resume invokes the assembly path",
        len(router_calls) >= 1,
        "router.run() was never called — the plugin took the pass-through "
        "shortcut and the LLM got none of the accumulated history",
    )


# --- Entry point --------------------------------------------------------------

def main() -> int:
    print(textwrap.dedent("""
        hermes-mneme: state-tracking bug reproducer
        Each scenario runs on a fresh temporary SQLite DB.
    """).strip())

    run_scenario("A. turn_count tracks user turns", scenario_A_turn_count)
    run_scenario("B. state_history keepalive after long stable goal",
                 scenario_B_state_history_keepalive)
    run_scenario("C. context_search empty-query diagnostics",
                 scenario_C_context_search_diagnostics)
    run_scenario("D. lineage cycle rejection", scenario_D_lineage_cycles)
    run_scenario("E. context_search(session='current') over lineage chain",
                 scenario_E_current_session_lineage)
    run_scenario("F. enricher recovers usable signal from truncated JSON",
                 scenario_F_truncated_json)
    run_scenario("G. pass-through guard accounts for prompt overhead",
                 scenario_G_pass_through_overhead)
    run_scenario("H. resume pulls accumulated context from DB",
                 scenario_H_resume_context_fill)

    print("\n" + "=" * 76)
    if INVARIANT_FAILURES:
        print(f"INVARIANTS FAILED: {len(INVARIANT_FAILURES)}")
        for f in INVARIANT_FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL INVARIANTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
