# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-16

### Added
- Native Hermes context-engine integration for current `main`:
  `select_context()` prepares request-only context before the provider call,
  and `on_turn_complete()` ingests the finalized turn after completion.
- Compatibility fallback for the pre-merge prototype hook
  `prepare_request_messages()`.
- Tests covering the native hook contract against current Hermes main.

### Changed
- `should_compress()` now disables the legacy every-turn compression path when
  the host Hermes exposes both native hooks, while preserving the old behavior
  on older Hermes builds.
- Prompt assembly preserves the current request tail during tool-continuation
  calls instead of re-appending the previous user message.
- Request-only prefill stays ahead of conversation history during assembly
  and is not ingested into the event store.
- Identical native request retries retain their selected memory context.
- `fetch_event` and `expand_context` accept unique event-id prefixes, matching
  the ids agents usually see in retrieved memory snippets.

## [0.3.0] - 2026-05-09

### Added — memory navigation (Stage B in `docs/DIVERGENCE.md`)
- New table `state_history` — append-only log of every `execution_state`
  update so the full goal trajectory of a session is preserved.
- New agent tools:
  - `list_segments(limit=10)` — table of contents of the current session.
  - `get_goal_history(limit=20)` — last N goals from `state_history`.
- `expand_context` gained `mode='segment'` — returns the spine of a
  segment (user messages + tool calls + last assistant message, max 15).
- `fetch_event` gained `full` flag; default content is capped to ~4000
  chars with a `truncated=true` marker.
- Prompt builder now emits up to three new blocks in the system message:
  - `[MEMORY ACCESS]` — fixed ~250-token hint listing the five cases when
    the agent must consult memory before asking the user or repeating work.
  - `[GOAL TRAIL]` — the last 3 unique goals with timestamps.
  - `[CHECKPOINT]` — injected once the loop guard trips.
- Loop guard: `_consecutive_memory_calls` counter in `engine.py` triggers
  a one-shot `[CHECKPOINT]` after N consecutive memory-tool calls without
  a productive action. Resets on any non-memory tool call.
- New config flags: `memory_access_hint_enabled`, `goal_trail_size`,
  `checkpoint_after_n_memory_calls`, `memory_tool_names`.

### Changed
- `prompt_builder.build()` signature accepts `goal_trail`,
  `memory_access_hint`, `checkpoint_block`. System message is composed of:
  override → memory hint → goal trail → execution state → checkpoint.
- Tool descriptions for `fetch_event`, `expand_context`,
  `get_execution_state` were rewritten to spell out when the agent should
  reach for them. `context_search` now documents cross-session recall via
  `segment='all', session='all'`.

### Notes
- The accumulated graph stays in the DB. The active window does **not**
  grow with session length: `[MEMORY ACCESS]` is fixed-size,
  `[GOAL TRAIL]` is ~50 tokens, `[RETRIEVED CONTEXT]` is bounded by
  `retrieved_budget_ratio`.
- Conscious non-goals: we do **not** replicate LCM-style hierarchical DAG
  injection (it inherits the "DAG eats the window" failure mode), and we
  do not externalize tool outputs to files.

## [0.2.0] - 2026-05-08

### Fixed — semantic tail (Stage A in `docs/DIVERGENCE.md`)
- After the first `compress()` of a session, retrieval returned `[]` for
  several turns because `index.search()` filtered by the literal current
  `session_id` and ignored the parent chain. Five-file fix:
  - `store.get_latest_compressed_session()` walks `sessions.parent_id`.
  - `index.search()` accepts a list of session ids and OR-filters the KNN.
  - `router._route()` resolves the lineage once per turn.
  - `engine` prefetches the lineage on session start.
  - `tools._context_search()` reuses the lineage when `session='current'`.
- The runtime semantics of `session='current'` are now "current session +
  all compression ancestors", not "literal current row".

## [0.1.0] - 2026-05-06

### Added
- Initial MVP implementation of the Context Engine Plugin.
- **Phase 0-6**: Event Store (SQLite), Parser, Embedding Index (Jina + sqlite-vec), Classifier, Context Router, Prompt Builder.
- **Phase 7**: Observability module with trace logging (`trace.jsonl`).
- **Phase 8**: Agent Tools (`context_search`, `fetch_event`, `expand_context`, `get_execution_state`).
- **Phase 9**: Session Segmenter (basic, hard triggers).
- **Phase 10**: Execution Graph (basic, adjacency list).
- **Phase 11**: Config system (`PluginConfig`, environment variables, YAML).
- **Phase 12**: Tests structure (`tests/unit/`, `tests/integration/`) with 9 passing tests.
- Documentation: `README.md`, `LICENSE`, `pyproject.toml`, `.gitignore`.
- Docs: `docs/architecture.md`, `docs/configuration.md`, `docs/development.md`.

### Notes
- This is an MVP (Minimal Viable Product) release.
- Segmentation, Graph traversal, and Delta Extraction are simplified.
- Performance optimization (async embedding, etc.) is left for future versions.
