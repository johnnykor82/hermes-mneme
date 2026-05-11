# Hermes-Mneme

> 🇷🇺 Читать на русском: [README.ru.md](README.ru.md)

> Retrieval-based context engine for [Hermes Agent](https://github.com/NousResearch/hermes-agent) — execution-graph-aware memory replacement for the default compressor.

**Mneme** (Μνήμη — the Greek muse of memory) is a drop-in context engine plugin for Hermes. It replaces the default lossy compressor with a state-aware memory layer that persists every event, embeds it for semantic recall, tracks an execution graph for causal lookups, and assembles each prompt from a token-budget-respecting mix of recent turns, retrieved context, and execution state.

Pairs naturally with [Mnemosyne](https://github.com/johnnykor82/mnemosyne) — its mother in mythology, and a separate Hermes plugin for cross-session memory.

## Why

The default compressor drops detail when the window fills. LCM keeps everything but rebuilds context heuristically. Mneme keeps everything **and** uses retrieval + execution graph to assemble *minimal sufficient* context for each turn.

## Features

- **SQLite event store** with deterministic event_ids — re-ingest is idempotent across compactions and session resumes.
- **Embedding index** via [sqlite-vec](https://github.com/asg017/sqlite-vec) (KNN) with Python fallback. Supports any OpenAI-compatible embedding endpoint (Jina-MLX local default, Ollama, OpenAI, …).
- **Session segmenter** — auto-detects topic boundaries via embedding drift; sliding centroid for evolving topics.
- **Intent classifier** (CONTINUATION / SWITCH / NEW_TASK / CLARIFICATION) — deterministic, no LLM in the hot path.
- **Execution graph** — tracks `tool_call → tool_output → decision` edges; powers dependency-propagation scoring (Stage 7).
- **Optional reranker** — second-stage ranking via Cohere/Jina/BGE endpoints (LiteLLM works).
- **Optional LLM enrichment** — extracts `open_loops`, `decisions`, `active_entities` every N turns.
- **Agent memory tools** — `context_search` (with cross-session mode), `fetch_event`, `expand_context`, `get_execution_state`.
- **Observability** — per-turn JSONL trace + in-memory metrics (hit rate, dependency usage, fallback rate, segmentation count).

## Install

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

All settings have sensible defaults. Override via:
- environment variables: `HERMES_CTX_<KEY>` (e.g. `HERMES_CTX_PROTECTED_TAIL_TURNS=12`)
- `config.yaml` in the plugin directory

Key knobs:
- `active_window_tokens` / `context_window_usage_percent` — total budget
- `protected_tail_turns` — last N turns always included verbatim
- `state_budget_ratio` / `retrieved_budget_ratio` — budget split
- `dependency_max_depth` / `dependency_decay` — execution-graph propagation
- `reranker_enabled` + `reranker_endpoint` — second-stage ranking
- `llm_enrichment_enabled` — async state enrichment

See [`config.py`](config.py) for the full schema with inline RU+EN docs.

## Architecture

See [`docs/`](docs/) for component-level deep-dives:
- `store.py` — SQLite event store (idempotent re-ingest, session lineage)
- `index.py` — embedding index (sqlite-vec + fallback)
- `segmenter.py` — drift-based segmentation
- `classifier.py` — intent signals (no LLM)
- `router.py` — query construction, retrieval, scoring (Stages 6–7)
- `prompt_builder.py` — token-budget enforcement
- `engine.py` — main lifecycle (compress, on_session_start, …)
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
git pull
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
