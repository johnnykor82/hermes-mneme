# Divergence from Specification

This document records every place where the runtime code diverges from the
original specification (`~/.hermes/projects/context_engine/docs/speca.txt`,
v1.0). Use it as a map for reviewers: spec text → current behaviour → file
and approximate line.

The document is append-only. Each section documents one stage of work; do
not rewrite earlier sections when adding new ones.

---

## Stage A — Semantic tail fix (compression chain)

**Spec position.** Component 2 (Event Store) and Component 5 (Router) say
that retrieval is performed against the entire session history. After a
compression event the new (child) session inherits a pointer to the parent
but the spec does not describe how the *index* should follow the chain.

**Problem in code.** Before this stage, after the first `compress()` call:
- The new session_id was recorded in `sessions.parent_id`, but
- `index.search(session_id, …)` only filtered by the *current* session_id,
  so retrieval against the post-compression session returned `[]` until
  enough events had been re-embedded in the child.

The "semantic tail" — events from the parent that should still be reachable
— was effectively lost for several turns after each compression.

**Fix (current behaviour).**
1. `store.get_latest_compressed_session(session_id)` walks `sessions.parent_id`
   backwards and returns the lineage chain.
2. `index.search()` now accepts the full session lineage and OR-filters the
   KNN by `session_id IN (chain)`.
3. `router._route()` resolves the lineage once per turn and passes it down.
4. `engine` populates the lineage on session start, not lazily on the first
   miss.
5. `tools._context_search()` uses the same lineage when `session='current'`
   so agent-side recall sees the same set of events that the router uses.

**Files (approximate).**
- [store.py](../store.py) — `get_latest_compressed_session`, lineage helpers.
- [index.py](../index.py) — KNN with `session_id IN (...)`.
- [router.py](../router.py) — lineage in `_route`.
- [engine.py](../engine.py) — lineage prefetch on `__init__` / session start.
- [tools.py](../tools.py) — lineage in `_context_search` for the `current`
  session case.

**Why this is a divergence.** The spec assumes "session" is a primary key
visible to retrieval. In practice we treat session as a *chain* (parent
links). The runtime semantics of `session=current` are now "current session
plus all of its ancestors via compression", not "the literal current row".

---

## Stage B — Memory navigation (agent-side graph access)

This stage replaces the LCM-style "stuff a hierarchical DAG into the
window" approach with "graph stays in the DB, agent walks it on demand".
None of this is in the original spec.

**Goal.** Keep the active window small and stable; teach the agent that
its past work is in the database and give it tools + an in-prompt hint to
reach into it.

**Sub-stages.**

### B.1 — `state_history` table (append-only goal log)

Spec: a single `execution_state` row per session, overwritten on every
update.

Code: a new `state_history` table is appended on every state change so the
full trajectory of `goal` / `current_step` / `intent_label` /
`decisions_added` is retained for the lifetime of the session.

Schema (kept verbatim in code):

```sql
CREATE TABLE IF NOT EXISTS state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    goal TEXT,
    current_step TEXT,
    intent_label TEXT,
    decisions_added_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_state_history_session
    ON state_history(session_id, timestamp);
```

Files:
- [store.py](../store.py) — schema + `append_state_history`,
  `get_state_history`, `get_recent_unique_goals`.
- [engine.py](../engine.py) — appends after `save_state` (deduped).

### B.2 — Two new agent tools

| Tool | Purpose |
|------|---------|
| `list_segments(limit=10)` | Table of contents of the session: per-segment time range, event count, top tags, last intent label. |
| `get_goal_history(limit=20)` | Last N entries from `state_history` for the current session. |

Files:
- [tools.py](../tools.py) — schemas + handlers.
- [store.py](../store.py) — `list_segments`, `get_state_history`.

### B.3 — `expand_context` segment mode

Spec: `expand_context` only walks neighbours of a seed event.

Code: an additional `mode='segment'` returns the *spine* of the segment
that contains the seed event (or the segment with the given `segment_id`):
every user message + every tool_call + the last assistant message, capped
at 15 events.

Files:
- [tools.py](../tools.py) — `_expand_context` mode dispatch.
- [store.py](../store.py) — `get_segment_skeleton`.

### B.4 — `fetch_event` truncation flag

Spec: `fetch_event` returns the raw event.

Code: by default content is capped to ~4000 chars (~1000 tokens) with
`truncated=true` and `original_chars` so an unlucky agent does not blow
its window on a 200 KB tool output. Pass `full=true` to get the original.

Files:
- [tools.py](../tools.py) — `_fetch_event`.

### B.5 — Three system-prompt blocks

The system message assembled by `prompt_builder.build()` now contains, in
order:

1. Original system prompt (or `system_prompt_override`).
2. `[MEMORY ACCESS]` — a fixed ~250–300 token block describing the memory
   tools and the five mandatory cases when the agent must consult memory
   *before* asking the user or repeating work. Defined as
   `MEMORY_ACCESS_HINT` in [engine.py](../engine.py).
3. `[GOAL TRAIL]` — the last `goal_trail_size` (default 3) unique goals
   from `state_history`, with timestamps and a `← текущая` marker on the
   most recent.
4. `[EXECUTION STATE]` — the JSON state (unchanged).
5. `[CHECKPOINT]` — only present when the loop guard fires (see B.6).

Files:
- [engine.py](../engine.py) — `MEMORY_ACCESS_HINT`,
  `_build_memory_access_hint`, `_build_goal_trail`, call to
  `prompt_builder.build()` extended.
- [prompt_builder.py](../prompt_builder.py) — extended `build()` signature,
  `_format_goal_trail`, system message assembly.

### B.6 — Forced checkpoint on memory-tool loops

Counter `_consecutive_memory_calls` in `engine.py` increments on every
call to a tool listed in `memory_tool_names` and resets on any other
tool call. When the counter reaches
`checkpoint_after_n_memory_calls` (default 5), a `[CHECKPOINT]` block is
appended to the *next* system message: it cites the current goal and a
goal from earlier in the session and reminds the agent to either continue
the work or call `get_goal_history(20)`.

This is the third level of loop protection (after the passive
`get_goal_history` tool and the always-injected `[GOAL TRAIL]`).

Files:
- [engine.py](../engine.py) — counter, `_format_checkpoint_block`,
  `_consume_pending_checkpoint`, `handle_tool_call`.
- [prompt_builder.py](../prompt_builder.py) — `checkpoint_block` parameter.

### B.7 — Four new config flags

Added to `DEFAULT_CONFIG` in [config.py](../config.py):

| Flag | Default | Purpose |
|------|---------|---------|
| `memory_access_hint_enabled` | `True` | Inject `[MEMORY ACCESS]` into every system message. |
| `goal_trail_size` | `3` | Number of latest unique goals shown in `[GOAL TRAIL]`. |
| `checkpoint_after_n_memory_calls` | `5` | Threshold for the loop guard. `0` disables. |
| `memory_tool_names` | list of 5 tools | Tool names that count toward the loop guard. |

---

## What is *not* divergent

For clarity, these spec elements are still implemented as written:

- Token budget allocation (state 0.05 / retrieved 0.30 / tail 0.55 /
  headroom 0.10).
- Drift-based segmentation with `drift_threshold=0.35`.
- Tool-output compression at 500-token threshold to ~100-token summary.
- `tiktoken cl100k_base` as default tokenizer.
- `sqlite-vec` as the only vector backend.

Anything not listed in this document follows the spec.
