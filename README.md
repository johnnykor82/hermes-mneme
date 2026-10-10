# Hermes-Mneme

> 🇷🇺 Читать на русском: [README.ru.md](README.ru.md)

> Retrieval-based context engine for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — execution-graph-aware memory replacement for the default compressor.

**Mneme** (Μνήμη — the Greek muse of memory) is a drop-in context engine plugin for Hermes. It replaces the default lossy compressor with a state-aware memory layer that persists every event, embeds it for semantic recall, tracks an execution graph for causal lookups, and assembles each prompt from a token-budget-respecting mix of recent turns, retrieved context, and execution state.

Pairs naturally with [Mnemosyne](https://github.com/johnnykor82/mnemosyne) — its mother in mythology, and a separate Hermes plugin for cross-session memory.

## Why

The default compressor drops detail when the window fills. LCM keeps everything but rebuilds context heuristically. Mneme keeps everything **and** uses retrieval + execution graph to assemble *minimal sufficient* context for each turn.

## Features

- **SQLite event store** with deterministic event_ids — re-ingest is idempotent across compactions and session resumes.
- **Embedding index** via [sqlite-vec](https://github.com/asg017/sqlite-vec) (exact scoped cosine) with Python fallback. Supports any OpenAI-compatible embedding endpoint (Jina-MLX local default, Ollama, OpenAI, …).
- **Session segmenter** — auto-detects topic boundaries via embedding drift; sliding centroid for evolving topics.
- **Intent classifier** (CONTINUATION / SWITCH / NEW_TASK / CLARIFICATION) — deterministic, no LLM in the hot path.
- **Execution graph** — tracks `tool_call → tool_output → decision` edges; powers dependency-propagation scoring (Stage 7).
- **Optional reranker** — second-stage ranking via Cohere/Jina/BGE endpoints (LiteLLM works).
- **Optional LLM enrichment** — extracts `open_loops`, `decisions`, `active_entities` every N turns.
- **Agent memory tools** — `context_search` (with cross-session mode), `fetch_event`, `expand_context`, `get_execution_state`.
- **Observability** — per-turn JSONL trace + in-memory metrics (hit rate, dependency usage, fallback rate, segmentation count).

## Install

Version 0.5.0 uses Hermes' native `select_context()` and
`on_turn_complete()` hooks when both are available. Context selection changes
only the outgoing request; completed turns are ingested into memory without
forcing `compress()` on every step. Older Hermes builds retain the legacy
compression-based integration. Restart Hermes after updating the plugin.

```bash
git clone https://github.com/johnnykor82/hermes-mneme.git \
  ~/.hermes/plugins/hermes-mneme
cd ~/.hermes/plugins/hermes-mneme
./install.sh
hermes gateway restart
```

The installer detects your Hermes venv (default `~/.hermes/hermes-agent/venv`), installs Python deps from `requirements.txt`, and verifies. Override venv location with `HERMES_VENV=...`.

After restart, watch for activation:
```bash
tail -f ~/.hermes/logs/agent.log | grep -i mneme
```

You should see: `Hermes-Mneme context engine loaded.`

## Configuration

Settings load in order: defaults → `config.yaml` → environment → explicit constructor values. Invalid values are ignored with a diagnostic. Override via:
- environment variables: `HERMES_CTX_<KEY>` (e.g. `HERMES_CTX_PROTECTED_TAIL_TURNS=12`)
- `config.yaml` in the plugin directory

Key knobs:
- `active_window_tokens` / `context_window_usage_percent` — total budget
- `protected_tail_turns` — last N turns always included verbatim
- `state_budget_ratio` / `retrieved_budget_ratio` — budget split
- `dependency_max_depth` / `dependency_decay` — execution-graph propagation
- `reranker_enabled` + `reranker_endpoint` — second-stage ranking
- `llm_enrichment_enabled` — async state enrichment
- `segmentation_enabled`, `drift_threshold`, `drift_weights` — topic boundary policy; disabled blocks hard and soft switches. Embedding-only drift is `(1-cos)/2`, without weights for unavailable signals. Threshold 0.35 is configurable and has not been calibrated on a real RU/EN gold set.
- `embedding_endpoint`, `embedding_model`, `embedding_api_key`, `embedding_dim` — instance embedding settings; dimensions 768/1024/1536 tested. Match model and dimension; changing semantics requires explicit reindexing.

`context_search(query='...', segment='current', session='current')` searches only the active topic. Use `segment='all'` for all topics and both `segment='all', session='all'` for explicit global recall. Global exact/unlimited searches can be expensive on large databases.

Impossible estimated prompt budgets raise `PromptBudgetExceeded` with `actual_tokens` and `budget_tokens`. The current Hermes host catches hook errors and falls back to its original request; provider limits still need host enforcement.

Existing shared `seg_1` data is not renamed automatically. Use the [backed-up migration guide](docs/audit-migration.md) for dry-run, explicit application and rollback. See the [audit verification report](docs/AUDIT-2026-10-10.md) for confirmed findings, production-copy integrity, benchmarks and test limitations.

Compatibility is detected by hook availability: hosts with both `select_context` and `on_turn_complete` use native mode; older hosts retain `compress`. Tests cover the native hook ABI and legacy-hook absence. A numeric minimum host release is not established.

See [`config.py`](config.py) for the full schema with inline RU+EN docs.

## Architecture

See [`docs/`](docs/) for component-level deep-dives:
- `store.py` — SQLite event store (idempotent re-ingest, session lineage)
- `index.py` — embedding index (sqlite-vec + fallback)
- `segmenter.py` — drift-based segmentation
- `classifier.py` — intent signals (no LLM)
- `router.py` — query construction, retrieval, scoring (Stages 6–7)
- `prompt_builder.py` — token-budget enforcement
- `engine.py` — main lifecycle (select_context, on_turn_complete, on_session_start; legacy compress fallback)
- `graph.py` — execution graph + dependency propagation
- `tools.py` — agent memory tools

## Tests

```bash
~/.hermes/hermes-agent/venv/bin/pytest tests/unit -q
```

## Updating

When new commits land on `main`:

```bash
cd ~/.hermes/plugins/hermes-mneme
git pull --ff-only
./install.sh              # reinstalls deps if requirements changed
hermes gateway restart
```

Your runtime data (`db/plugin.db`, `trace.jsonl`) is gitignored and survives updates.

## Contributing

Contributions and bug reports are very welcome. Standard GitHub flow:

1. **Issues** — open an issue describing the problem or feature idea.
2. **Pull requests** — fork, branch, commit, push, open a PR against `main`.

Before submitting a PR:

- Verify your change works on both **macOS** and **Linux** if it touches `install.sh` or filesystem paths.
- Run `pytest tests/unit -q`.
- Run `ruff check` to catch obvious style issues.
- Keep commits focused — one concern per commit.

## License

[Apache-2.0](LICENSE)
