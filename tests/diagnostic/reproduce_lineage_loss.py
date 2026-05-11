"""Standalone repro for orphan-session / lineage-loss bugs in hermes-mneme.

Why this exists
---------------
Driving the agent through long sessions to surface these bugs is slow and
expensive. This script wires the plugin's CustomRouterContextEngine to an
isolated temporary SQLite database and replays the EXACT hook sequence
Hermes uses (on_session_start, compress, on_session_end, on_session_reset)
under several scenarios.

Each scenario prints:
  - state of `sessions`, `events`, `session_lineage`, `execution_state` after
    every Hermes hook call,
  - in-memory engine fields that matter (`self.session_id`,
    `_skip_compression_boundary`, `_processed_msg_count`, etc.),
  - PASS/FAIL on a set of invariants we expect to hold.

What it does NOT cover
----------------------
- the embedding endpoint (Jina) — the test forces ``llm_enrichment_enabled=False``
  and lets get_embedding hit the circuit breaker silently. We don't care about
  retrieval quality here, only about session bookkeeping.
- prompt_builder output token math — irrelevant to lineage bugs.

How to run
----------
    cd ~/.hermes/plugins/hermes-mneme
    python3 -m tests.diagnostic.reproduce_lineage_loss

It prints to stdout, exits non-zero if any invariant fails.

Key bug hypothesis the scenarios target
---------------------------------------
B-PASS-THROUGH-LOSS:
    compress() returns pass-through when content < self._budget_tokens. It
    sets `_skip_compression_boundary=True`. Then Hermes — independently —
    decides to rotate session_id for its OWN compression reasons (its
    compression.threshold is tied to model window %, not the plugin's
    budget). It calls on_session_start(boundary_reason='compression',
    old_session_id=OLD). The plugin sees `_skip_compression_boundary=True`,
    swaps session_id WITHOUT calling reassign_session, and returns.
    Result: events from OLD remain under OLD; new events under NEW; no
    lineage row connecting them. Repeat each turn → orphan-session pile-up.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
import sys
import tempfile
import textwrap
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Stub Hermes-side imports BEFORE importing the plugin ---------------------
# The plugin imports `from agent.context_engine import ContextEngine` and
# `from hermes_logging import _session_context`. Provide minimal stand-ins so
# the plugin can load standalone.

def _install_hermes_stubs() -> Any:
    agent_pkg = types.ModuleType("agent")
    ctx_mod = types.ModuleType("agent.context_engine")

    class ContextEngine:  # noqa: D401 — match upstream signature
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


# Ensure the plugin package is importable regardless of CWD.
PLUGIN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_DIR.parent))  # parent of `hermes-mneme/`

_SCTX = _install_hermes_stubs()

# Plugin import is package-style: import the package by its directory name.
# The directory contains a hyphen, so we re-export it under an importable
# alias via an `importlib` trick.
import importlib
import importlib.util

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

# Now use the engine module via the alias.
engine_module = importlib.import_module("hermes_mneme.engine")
config_module = importlib.import_module("hermes_mneme.config")
CustomRouterContextEngine = engine_module.CustomRouterContextEngine


# --- DB inspection helpers ----------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def snapshot(db_path: str) -> Dict[str, Any]:
    """Return a compact dict-of-rows snapshot of bookkeeping tables."""
    conn = _connect(db_path)
    try:
        sessions = [dict(r) for r in conn.execute(
            "SELECT session_id, parent_id, status, last_active FROM sessions "
            "ORDER BY created_at"
        )]
        events_by_session = {
            r["session_id"]: r["n"]
            for r in conn.execute(
                "SELECT session_id, COUNT(*) AS n FROM events "
                "GROUP BY session_id"
            )
        }
        lineage = [dict(r) for r in conn.execute(
            "SELECT old_session_id, new_session_id, created_at FROM session_lineage "
            "ORDER BY created_at"
        )]
        exec_state = {
            r["session_id"]
            for r in conn.execute("SELECT session_id FROM execution_state")
        }
        state_history = [
            (r["session_id"], r["goal"])
            for r in conn.execute(
                "SELECT session_id, goal FROM state_history ORDER BY id"
            )
        ]
        return {
            "sessions": sessions,
            "events_by_session": events_by_session,
            "lineage": lineage,
            "exec_state_sessions": sorted(exec_state),
            "state_history": state_history,
        }
    finally:
        conn.close()


def print_snapshot(label: str, snap: Dict[str, Any]) -> None:
    print(f"\n--- {label} ---")
    print(f"sessions ({len(snap['sessions'])}):")
    for s in snap["sessions"]:
        print(f"    {s['session_id']}  parent={s['parent_id']}  "
              f"status={s['status']}  last_active={s['last_active']}")
    print(f"events_by_session: {snap['events_by_session']}")
    print(f"lineage rows ({len(snap['lineage'])}):")
    for l in snap["lineage"]:
        print(f"    {l['old_session_id']} -> {l['new_session_id']}")
    print(f"execution_state sessions: {snap['exec_state_sessions']}")
    if snap["state_history"]:
        print(f"state_history goals: {snap['state_history']}")


# --- Scenario helpers ---------------------------------------------------------

def msg(role: str, content: str) -> Dict[str, Any]:
    return {"role": role, "content": content}


def make_engine(db_path: str) -> Any:
    """Build a fresh engine bound to ``db_path``. Configuration is tweaked
    so the test is fast and deterministic:
      - tiny token budget so we can force/avoid pass-through with short text,
      - LLM enrichment off (no Jina, no network),
      - bootstrap probe off (no network).
    """
    cfg = config_module.PluginConfig({
        "active_window_tokens": 200,          # tiny → easy to overflow
        "context_window_usage_percent": 0.70,
        "protected_tail_turns": 4,
        "llm_enrichment_enabled": False,
        "memory_access_hint_enabled": False,
        "bootstrap_probe_enabled": False,
        "segmentation_enabled": False,
    })
    engine = CustomRouterContextEngine(db_path=db_path, config=cfg)
    # Don't actually call out to embedding endpoint — short-circuit get_embedding.
    engine.indexer.get_embedding = lambda *_a, **_kw: None
    return engine


def set_sctx(session_id: Optional[str]) -> None:
    _SCTX.session_id = session_id


def label_for_turn(idx: int) -> str:
    return f"turn{idx}"


# --- Invariants ---------------------------------------------------------------

INVARIANT_FAILURES: List[str] = []


def expect(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{'  — ' + detail if detail else ''}")
    if not cond:
        INVARIANT_FAILURES.append(f"{name}: {detail}")


def assert_invariants(snap: Dict[str, Any], expected_session: str,
                      expect_lineage_chain: bool) -> None:
    print("Invariants:")

    sessions_with_events = {sid for sid, n in snap["events_by_session"].items() if n > 0}

    # Inv 1: every events_by_session entry has a matching sessions row
    orphan_event_sids = sessions_with_events - {s["session_id"] for s in snap["sessions"]}
    expect("events ↔ sessions parity", not orphan_event_sids,
           f"events under sessions with no row: {sorted(orphan_event_sids)}")

    # Inv 2: expected session has events
    n_expected = snap["events_by_session"].get(expected_session, 0)
    expect(f"expected session has events ({expected_session})",
           n_expected > 0, f"event count = {n_expected}")

    if expect_lineage_chain:
        # Inv 3: if more than 1 session exists, lineage rows must connect them.
        ids = {s["session_id"] for s in snap["sessions"]}
        if len(ids) > 1:
            lin_pairs = {(l["old_session_id"], l["new_session_id"]) for l in snap["lineage"]}
            # We expect at least N-1 lineage rows for N sessions in a single
            # logical conversation.
            expect("lineage row count ≥ N_sessions − 1",
                   len(lin_pairs) >= len(ids) - 1,
                   f"got {len(lin_pairs)} lineage rows for {len(ids)} sessions")
            # Inv 4: events of OLD sessions should have been migrated to the
            # latest session, so they should be empty (or absent).
            for sid in ids:
                if sid != expected_session and snap["events_by_session"].get(sid, 0) > 0:
                    expect(
                        f"old session {sid} should be empty after reassign",
                        False,
                        f"still has {snap['events_by_session'][sid]} events",
                    )


# --- Scenarios ----------------------------------------------------------------

def run_scenario(name: str, fn) -> None:
    print("\n" + "=" * 76)
    print(f"SCENARIO: {name}")
    print("=" * 76)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "plugin.db")
        try:
            fn(db_path)
        except Exception:
            print("UNEXPECTED EXCEPTION:")
            traceback.print_exc()
            INVARIANT_FAILURES.append(f"{name}: raised exception")


def scenario_A_clean_compression(db_path: str) -> None:
    """Baseline: fresh session → big messages → Hermes signals compression
    with old_session_id=S1, new=S2. We expect reassign_session to migrate
    everything and write lineage. This is the path the code WAS designed for.
    """
    engine = make_engine(db_path)
    S1 = "20260511_100000_aaaaaa"
    S2 = "20260511_100500_bbbbbb"

    set_sctx(S1)
    engine.on_session_start(S1)

    # Big user+assistant turn to force overflow / definitely not pass-through.
    big_user = "USER " + ("question " * 200)
    big_asst = "ASSISTANT " + ("answer " * 200)
    messages = [msg("user", big_user), msg("assistant", big_asst)]
    engine.compress(messages, current_tokens=10_000)
    print_snapshot("after first compress() on S1", snapshot(db_path))

    # Hermes decides to compress; tells the plugin.
    set_sctx(S2)
    engine.on_session_start(
        S2, boundary_reason="compression", old_session_id=S1
    )
    snap = snapshot(db_path)
    print_snapshot("after on_session_start(compression, old=S1)", snap)
    assert_invariants(snap, expected_session=S2, expect_lineage_chain=True)


def scenario_B_pass_through_then_compression(db_path: str) -> None:
    """The hypothesis: plugin returns pass-through (small content < budget),
    sets `_skip_compression_boundary=True`, then Hermes — independently —
    decides to compress and calls on_session_start(boundary='compression').
    Plugin's Case-A short-circuit skips reassign_session → orphan-session.
    """
    engine = make_engine(db_path)
    S1 = "20260511_110000_aaaaaa"
    S2 = "20260511_110500_bbbbbb"

    set_sctx(S1)
    engine.on_session_start(S1)

    # Tiny content — well under budget=200. Plugin will pass-through.
    engine.compress([msg("user", "hi"), msg("assistant", "hello")], current_tokens=20)
    snap_after_compress = snapshot(db_path)
    print_snapshot("after pass-through compress() on S1", snap_after_compress)
    print(f"    engine._skip_compression_boundary = {getattr(engine, '_skip_compression_boundary', None)}")

    # Hermes still decides to rotate (its own threshold may be % of model
    # window, not budget tokens) → calls boundary='compression'.
    set_sctx(S2)
    engine.on_session_start(
        S2, boundary_reason="compression", old_session_id=S1
    )
    snap = snapshot(db_path)
    print_snapshot("after on_session_start(compression, old=S1)", snap)
    print(f"    engine.session_id = {engine.session_id}")
    print(f"    engine._skip_compression_boundary = {getattr(engine, '_skip_compression_boundary', None)}")

    # New events arrive on S2.
    set_sctx(S2)
    engine.compress([msg("user", "follow-up question"), msg("assistant", "answer 2")],
                    current_tokens=50)
    snap = snapshot(db_path)
    print_snapshot("after compress() on S2", snap)

    # The killer invariant: lineage must connect S1 and S2.
    assert_invariants(snap, expected_session=S2, expect_lineage_chain=True)


def scenario_C_gateway_restart(db_path: str) -> None:
    """Hermes process restart: comes back with a new session_id but NO
    boundary_reason. Plugin can't tell this is a continuation. Drift
    detection in compress() decides FRESH-rebind (status of S2 in DB is
    None) — events under S1 stay orphan. This is the second class of
    orphan-session production.
    """
    engine = make_engine(db_path)
    S1 = "20260511_120000_aaaaaa"
    S2 = "20260511_120500_bbbbbb"

    set_sctx(S1)
    engine.on_session_start(S1)
    engine.compress([msg("user", "doing real work"), msg("assistant", "ok")],
                    current_tokens=10_000)
    print_snapshot("after S1 work", snapshot(db_path))

    # Simulate process restart by rebuilding engine on same db, then a new
    # turn arrives under a brand-new session id (no on_session_start, no
    # boundary signal — that matches what Hermes did between 221100→221741).
    del engine
    engine = make_engine(db_path)
    set_sctx(S2)
    engine.compress(
        [msg("user", "continuing the work"), msg("assistant", "sure")],
        current_tokens=10_000,
    )
    snap = snapshot(db_path)
    print_snapshot("after compress() under brand-new S2 (no on_session_start)", snap)

    # We expect either:
    #   - lineage S1→S2 (events migrated), OR
    #   - at minimum a lineage-style link so the user can navigate back.
    assert_invariants(snap, expected_session=S2, expect_lineage_chain=True)


def scenario_D_drift_without_boundary(db_path: str) -> None:
    """Hermes thread-local session_id changes mid-process without an
    on_session_start call. Plugin's drift detection in compress() handles
    it — but only correctly if the source path matches reality.
    """
    engine = make_engine(db_path)
    S1 = "20260511_130000_aaaaaa"
    S2 = "20260511_130500_bbbbbb"

    set_sctx(S1)
    engine.on_session_start(S1)
    engine.compress([msg("user", "first turn"), msg("assistant", "ack")],
                    current_tokens=10_000)
    print_snapshot("after S1 turn", snapshot(db_path))

    # Drift: Hermes silently rotates _session_context without firing
    # on_session_start (the engine.py comment specifically says this happens).
    set_sctx(S2)
    engine.compress([msg("user", "after drift"), msg("assistant", "still here")],
                    current_tokens=10_000)
    snap = snapshot(db_path)
    print_snapshot("after compress() with drifted session_id", snap)
    assert_invariants(snap, expected_session=S2, expect_lineage_chain=True)


def scenario_E_replay_observed_chain(db_path: str) -> None:
    """Replays the exact session pattern observed in the real plugin.db:
        S_a → S_b   (lineage rows present)
        S_b → S_c   (lineage row MISSING in real DB)
    We expect: scenario_B / scenario_C explain the missing lineage; this
    confirms the cumulative effect.
    """
    engine = make_engine(db_path)
    S_a = "20260510_220241_bfdf8a"
    S_b = "20260510_221100_b99995"
    S_c = "20260510_221741_f814dc"

    set_sctx(S_a)
    engine.on_session_start(S_a)
    engine.compress([msg("user", "A " + "x" * 500), msg("assistant", "A reply " + "y" * 500)],
                    current_tokens=20_000)
    set_sctx(S_b)
    engine.on_session_start(S_b, boundary_reason="compression", old_session_id=S_a)
    engine.compress([msg("user", "B " + "x" * 500), msg("assistant", "B reply " + "y" * 500)],
                    current_tokens=20_000)
    print_snapshot("after S_a → S_b", snapshot(db_path))

    # Now a pass-through followed by a Hermes-driven compression boundary —
    # this is the suspected break point that produces orphan S_c in the field.
    engine.compress([msg("user", "tiny"), msg("assistant", "ok")], current_tokens=20)
    set_sctx(S_c)
    engine.on_session_start(S_c, boundary_reason="compression", old_session_id=S_b)
    engine.compress([msg("user", "C " + "x" * 500), msg("assistant", "C reply " + "y" * 500)],
                    current_tokens=20_000)
    snap = snapshot(db_path)
    print_snapshot("after S_b → S_c", snap)

    # If the bug is the one we suspect, the S_b → S_c lineage row will be
    # missing and S_b will still hold its events.
    assert_invariants(snap, expected_session=S_c, expect_lineage_chain=True)


# --- Entry point --------------------------------------------------------------

def main() -> int:
    print(textwrap.dedent("""
        hermes-mneme: orphan-session / lineage-loss reproducer
        Each scenario runs on a fresh temporary SQLite DB.
        Failing invariants are summarised at the end.
    """).strip())

    run_scenario("A. clean compression boundary", scenario_A_clean_compression)
    run_scenario("B. pass-through then external compression (PRIME SUSPECT)",
                 scenario_B_pass_through_then_compression)
    run_scenario("C. gateway restart, new session_id, no on_session_start",
                 scenario_C_gateway_restart)
    run_scenario("D. drift via _session_context, no on_session_start",
                 scenario_D_drift_without_boundary)
    run_scenario("E. replay observed S_a → S_b → S_c chain",
                 scenario_E_replay_observed_chain)

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
