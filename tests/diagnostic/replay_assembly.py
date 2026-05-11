"""Replay context assembly against a real session in plugin.db.

Loads events, runs router → dedup → fallback → prompt_builder, exactly the
way engine.compress() does it, and prints a turn-by-turn breakdown so we
can see where retrieved candidates are lost and how much of the budget
each block actually consumes.

Usage:
    python -m tests.diagnostic.replay_assembly \
        --session 20260509_120657_097868 \
        --budget 179200

Defaults match the production log line that triggered this investigation
(budget=179200, session=20260509_120657_097868). The script does NOT
mutate the DB.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

# Make `hermes_mneme` importable when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import importlib.util
import pathlib

if "hermes_mneme" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "hermes_mneme",
        pathlib.Path(_PLUGIN_DIR) / "__init__.py",
        submodule_search_locations=[_PLUGIN_DIR],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["hermes_mneme"] = pkg
    spec.loader.exec_module(pkg)

from hermes_mneme import config as config_module  # noqa: E402
from hermes_mneme import index as index_mod  # noqa: E402
from hermes_mneme import prompt_builder as pb_mod  # noqa: E402
from hermes_mneme import router as router_mod  # noqa: E402
from hermes_mneme import segmenter as seg_mod  # noqa: E402
from hermes_mneme import store as store_mod  # noqa: E402

DEFAULT_DB = os.path.join(_PLUGIN_DIR, "db", "plugin.db")
DEFAULT_SESSION = "20260509_120657_097868"
DEFAULT_BUDGET = 179200


def load_messages(store: store_mod.ContextStore, session_id: str) -> List[Dict[str, Any]]:
    """Reconstruct an OpenAI-style message list from events table.

    We preserve role/content only — that's enough for the prompt builder
    (it only inspects `content`) and for token counting. tool_call /
    tool_output keep their original roles ("assistant" / "tool").
    """
    conn = store._get_connection()
    try:
        rows = conn.execute(
            """SELECT id, type, role, content, segment_id, timestamp
               FROM events WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    msgs: List[Dict[str, Any]] = []
    for r in rows:
        msgs.append({
            "role": r["role"] or "user",
            "content": r["content"] or "",
            "_event_id": r["id"],
            "_type": r["type"],
            "_segment_id": r["segment_id"],
        })
    return msgs


def latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content:
                # Strip any [RETRIEVED CONTEXT] envelope if it leaked in.
                stripped = router_mod._RETRIEVED_PREFIX_RE.sub("", content).strip()
                return stripped or content
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--system-prompt-tokens", type=int, default=0,
                        help="Override system-prompt overhead. 0 = leave unmeasured (effective_budget == total).")
    parser.add_argument("--protected-tail-turns", type=int, default=None,
                        help="Override config.protected_tail_turns for this run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    cfg = config_module.PluginConfig()
    if args.protected_tail_turns is not None:
        cfg._config["protected_tail_turns"] = args.protected_tail_turns

    store = store_mod.ContextStore(args.db)
    indexer = index_mod.EmbeddingIndex(args.db)
    # last_seg discovered below; init segmenter after we know it.
    segmenter = None  # set later
    rtr = router_mod.ContextRouter(store=store, indexer=indexer, config=cfg, segmenter=None)

    messages = load_messages(store, args.session)
    if not messages:
        print(f"No events for session {args.session}", file=sys.stderr)
        return 3

    # Discover segment of the most recent event.
    last_seg = messages[-1].get("_segment_id") or "seg_1"

    # Now wire segmenter and rebind on router.
    segmenter = seg_mod.SessionSegmenter(
        store=store, indexer=indexer, current_segment_id=last_seg, config=cfg
    )
    rtr.segmenter = segmenter

    state = store.load_state(args.session) or {
        "goal": "", "current_step": "", "open_loops": [],
        "decision_stack": [], "active_entities": [],
    }

    current_msg = latest_user_message(messages)
    recent_db = store.get_recent_events(args.session, limit=cfg.get("protected_tail_turns", 64))

    print("=" * 78)
    print(f"Session     : {args.session}")
    print(f"DB          : {args.db}")
    print(f"Events      : {len(messages)}")
    print(f"Last segment: {last_seg}")
    print(f"Budget      : {args.budget}")
    print(f"protected_tail_turns: {cfg.get('protected_tail_turns')}")
    print(f"retrieved_budget_ratio: {cfg.get('retrieved_budget_ratio')}")
    print(f"reranker_enabled: {cfg.get('reranker_enabled')}")
    print(f"router_min_candidates: {cfg.get('router_min_candidates')}")
    print(f"router_top_k: {cfg.get('router_top_k')}")
    print(f"current user msg ({len(current_msg)} chars): {current_msg[:200]!r}")
    print(f"goal               : {state.get('goal', '')[:120]!r}")
    print(f"current_step       : {state.get('current_step', '')[:120]!r}")
    print()

    pb = pb_mod.PromptBuilder(
        total_budget=args.budget,
        tokenizer_model=cfg.get("tokenizer_model", "cl100k_base"),
        protected_tail_turns=cfg.get("protected_tail_turns", 64),
        state_budget_ratio=cfg.get("state_budget_ratio", 0.05),
        retrieved_budget_ratio=cfg.get("retrieved_budget_ratio", 0.30),
        protected_tail_ratio=cfg.get("protected_tail_ratio", 0.55),
    )
    if args.system_prompt_tokens:
        pb.set_system_prompt_tokens(args.system_prompt_tokens)

    # ---- Stage 1: per-segment KNN ----
    raw_per_segment = indexer.search(
        current_msg, args.session, last_seg, top_k=cfg.get("router_top_k") or None
    )
    print(f"[1] per-segment KNN     : {len(raw_per_segment)} candidates")

    # ---- Stage 2: cross-segment fallback (router.run does this internally) ----
    raw_all = indexer.search(
        current_msg, args.session, "all", top_k=cfg.get("router_top_k") or None
    )
    seen_ids = {r.get("event_id") for r in raw_per_segment}
    added_cs = [r for r in raw_all if r.get("event_id") not in seen_ids]
    print(f"[2] cross-segment KNN   : {len(raw_all)} total, "
          f"+{len(added_cs)} new vs per-segment")

    # ---- Run the real router (this does steps 1+2 internally + score + rerank) ----
    router_input = router_mod.RouterInput(
        message=current_msg,
        execution_state=state,
        segment_id=last_seg,
        recent_events=recent_db,
        token_budget=args.budget,
        session_id=args.session,
    )
    result = rtr.run(router_input)
    print(f"[3] router result       : {result.candidates_retrieved} retrieved, "
          f"{len(result.candidates_selected)} after score+rerank, "
          f"intent={result.intent}, mode={result.mode}")

    # ---- Stage 4: dedup vs protected tail (engine.compress does this) ----
    protected_ids = {ev.get("id") for ev in (recent_db or []) if ev.get("id")}
    deduped = [c for c in result.candidates_selected if c.event_id not in protected_ids]
    print(f"[4] after dedup vs tail : {len(deduped)} "
          f"(dropped {len(result.candidates_selected) - len(deduped)})")

    # ---- Stage 4b: clean-query fallback (raw user message only, all segments) ----
    retrieved_budget_estimate = int(args.budget * cfg.get("retrieved_budget_ratio", 0.30))
    deduped_tokens = sum(pb.tokenizer(c.content or "") for c in deduped)
    if (
        retrieved_budget_estimate > 0
        and deduped_tokens < int(retrieved_budget_estimate * 0.5)
        and current_msg
    ):
        raw_clean = indexer.search(current_msg, args.session, "all", top_k=cfg.get("router_top_k") or None)
        seen = {c.event_id for c in deduped} | protected_ids
        added_clean = []
        for r in raw_clean:
            eid = r.get("event_id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            conn = store._get_connection()
            try:
                row = conn.execute(
                    "SELECT content, type, timestamp FROM events WHERE id = ?",
                    (eid,),
                ).fetchone()
            finally:
                conn.close()
            if not row:
                continue
            added_clean.append(router_mod.RetrievalCandidate(
                event_id=eid,
                score=float(r.get("similarity", 0.0)),
                content=row["content"] or "",
                type=row["type"],
                timestamp=row["timestamp"],
                embedding_model_id=r.get("embedding_model_id", ""),
                segment_id=r.get("segment_id", ""),
            ))
        if added_clean:
            deduped = deduped + added_clean
            deduped.sort(key=lambda x: x.score, reverse=True)
            print(f"[4b] clean-query fallback: +{len(added_clean)} "
                  f"(total {len(deduped)})")

    # Print which candidates we actually kept and their score / token cost.
    retrieved_budget_estimate2 = int(args.budget * cfg.get("retrieved_budget_ratio", 0.30))
    deduped_token_estimate = sum(pb.tokenizer(c.content or "") for c in deduped)
    print(f"    retrieved budget    ≈ {retrieved_budget_estimate2} tokens (30% of {args.budget})")
    print(f"    deduped pool tokens = {deduped_token_estimate}")
    print(f"    candidates kept:")
    for c in deduped[:20]:
        print(f"      score={c.score:.4f} type={c.type:18s} "
              f"tokens={pb.tokenizer(c.content or ''):5d} "
              f"event={c.event_id[:12]}…")

    # ---- Stage 5: build the actual prompt ----
    own_messages = messages
    # Pass ALL messages — prompt_builder treats protected_tail_turns as a
    # floor and extends back into older turns to fill headroom.
    recent_for_prompt = own_messages

    final = pb.build(
        execution_state=state,
        retrieved_candidates=deduped,
        recent_messages=recent_for_prompt,
        current_user_message=current_msg,
        goal_trail=None,
        memory_access_hint=None,
        checkpoint_block=None,
    )

    # Tally what landed in the prompt.
    state_tokens = pb.tokenizer(json.dumps(state, ensure_ascii=False))
    tail_tokens = sum(pb.tokenizer(str(m.get("content", ""))) + 4 for m in recent_for_prompt)
    current_tokens = pb.tokenizer(current_msg)
    retrieved_block_tokens = 0
    for m in final[1:]:
        content = str(m.get("content", ""))
        if content.startswith("[RETRIEVED CONTEXT]"):
            retrieved_block_tokens = pb.tokenizer(content)
            break
    final_total = sum(pb.tokenizer(str(m.get("content", ""))) + 4 for m in final)

    eff_budget = args.budget - pb.system_prompt_tokens
    print()
    print("[5] prompt_builder result:")
    print(f"    effective_budget          = {eff_budget}  (total {args.budget} − sys {pb.system_prompt_tokens})")
    print(f"    state allocated           = {int(eff_budget * cfg.get('state_budget_ratio', 0.05))}")
    print(f"    retrieved allocated       = {int(eff_budget * cfg.get('retrieved_budget_ratio', 0.30))}")
    print(f"    tail allocated            = {int(eff_budget * cfg.get('protected_tail_ratio', 0.55))}")
    print(f"    --- actual usage ---")
    print(f"    state_tokens              = {state_tokens}")
    print(f"    retrieved_block_tokens    = {retrieved_block_tokens}")
    print(f"    tail_tokens               = {tail_tokens}  (over {len(recent_for_prompt)} messages)")
    print(f"    current_user_msg_tokens   = {current_tokens}")
    print(f"    --- final prompt ---")
    print(f"    final messages            = {len(final)}")
    print(f"    final total tokens        = {final_total}  (cap {args.budget})")
    print(f"    UNUSED HEADROOM           = {args.budget - final_total}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
