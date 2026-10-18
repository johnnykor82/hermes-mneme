# Hermes-Mneme

> 🇷🇺 Читать на русском: [README.ru.md](README.ru.md)

> Retrieval-based context engine for [Hermes Agent](https://github.com/NousResearch/hermes-agent), with persistent event memory and execution-graph-aware retrieval.

**Mneme** (Μνήμη — the Greek muse of memory) is a context-engine plugin for Hermes. It stores ingested conversation events in SQLite, indexes them for semantic and lexical retrieval, tracks tool-call relationships, and selects context for model requests within a configured token budget. It can help recover older details, while retrieved evidence still needs to be checked against its source.

Mneme pairs with [Mnemosyne](https://github.com/johnnykor82/mnemosyne), a separate Hermes plugin for cross-session memory.

## Features

- **SQLite event store** with deterministic event IDs and idempotent replay handling across session resumes and compactions.
- **Embedding index** using [sqlite-vec](https://github.com/asg017/sqlite-vec), with a Python fallback and configurable OpenAI-compatible embedding providers.
- **Topic segmentation** based on embedding drift, with optional classifier signals.
- **Intent classification** for continuation, topic switch, new task, and clarification signals; this path is deterministic and does not call an LLM.
- **Execution graph** linking tool calls, outputs, and decisions for evidence navigation.
- **Optional reranking and LLM state enrichment**, when configured.
- **Memory tools:** `context_search`, `fetch_event`, `expand_context`, `get_execution_state`, `list_segments`, `get_goal_history`, and `recall_recent`.
- **Observability** through per-turn JSONL traces and in-memory metrics.

## Install

This README describes release **v0.5.3**. The command below checks out that release tag; the `main` branch may contain later unreleased changes.

```bash
git clone --branch v0.5.3 --depth 1 https://github.com/johnnykor82/hermes-mneme.git \
  ~/.hermes/plugins/hermes-mneme
cd ~/.hermes/plugins/hermes-mneme
./install.sh
```

If Hermes is not already configured to use Mneme as its context engine, set the engine name in `~/.hermes/config.yaml`:

```yaml
context:
  engine: hermes-mneme
```

After installation and configuration, restart the Gateway:

```bash
hermes gateway restart
```

The installer detects the Hermes virtual environment (default `~/.hermes/hermes-agent/venv`), installs dependencies from `requirements.txt`, and checks the installation. Set `HERMES_VENV` to use a different environment. Python 3.10+ is supported on macOS and Linux.

To check activation:

```bash
tail -f ~/.hermes/logs/agent.log | grep -i mneme
```

## Configuration

Settings are applied in this order: shipped defaults → `config.yaml` → `HERMES_CTX_<KEY>` environment variables → explicit constructor values. Invalid values are ignored with a diagnostic. See the [configuration reference](docs/configuration.md) and [`config.py`](config.py) for the full schema.

Common settings include:

- `active_window_tokens` and `context_window_usage_percent` for the context budget
- `protected_tail_turns`, `state_budget_ratio`, and `retrieved_budget_ratio` for prompt assembly
- `dependency_max_depth` and `dependency_decay` for graph traversal
- `reranker_enabled` and `reranker_endpoint` for optional second-stage ranking
- `llm_enrichment_enabled` for optional state enrichment
- `segmentation_enabled` and `drift_threshold` for topic-boundary detection
- `embedding_endpoint`, `embedding_model`, and `embedding_dim` for the embedding service; the model and dimension must match

The shipped drift threshold is `0.35`. It is a configurable operating value, not a universal accuracy claim or a threshold calibrated on a labeled Russian/English dataset. With embedding-only drift, the comparison uses normalized drift `(1 - cosine_similarity) / 2`; optional classifier signals can contribute when available.

## Retrieval and evidence

Release v0.5.3 builds on the 0.5.x fixes for session, segment, and state lifecycle; correctly scoped retrieval; and replay handling across resumes and compactions. It adds hybrid memory search, source-aware excerpts, current-turn exclusion, and safer segment expansion.

`context_search` searches the active topic and session by default. Set `segment='all'` to search all topics in the current session; use `segment='all', session='all'` for an explicit cross-session search.

Memory-tool search combines semantic and BM25 lexical candidates with reciprocal rank fusion, then reranks a bounded candidate pool when a reranker is configured. This hybrid ranking applies to the memory search tool; it does not replace the automatic prompt router's retrieval policy. Search results contain bounded, query-relevant excerpts with source role, session, segment, and truncation metadata. Use `fetch_event` to inspect the original event when needed. A ranked result is evidence to verify, not proof that a statement was accepted or completed.

Native Hermes `select_context()` and `on_turn_complete()` hooks are used when both are available. Context selection changes the outgoing model request; completed turns are ingested separately, without writing selected context into the saved transcript. Gateway Hygiene remains an independent Hermes host feature and can still shorten saved history at its configured threshold (85% by default); Mneme does not disable or control it. Older hosts use the legacy `compress()` fallback. Native turn exclusions follow the original logical turn across synthetic continuations.

`expand_context(mode='segment')` returns a bounded segment skeleton and reports when it is incomplete; it does not establish that an absent fact never appeared. A shared legacy segment ID may need a seed event to identify its source session. For evidence work, continue with broader or alternative searches, inspect neighboring events, and fetch originals as useful. There is no fixed search-attempt cap or rule that absence from one top-k result proves absence from memory. See [retrieval behavior](docs/memory-retrieval.md).

The lexical FTS index is derived data built atomically on first use. Its initial build adds latency and disk use but does not rewrite historical events or automatically migrate old segment IDs. Before first use on a large existing database, follow the safe update procedure in the [retrieval guide](docs/memory-retrieval.md): stop database writers, make a SQLite backup through the backup API that includes WAL state, initialize the index, and verify integrity before restarting. The separate [legacy segment migration guide](docs/audit-migration.md) describes its explicit backed-up migration process.

Prompt-budget overflow raises `PromptBudgetExceeded` with the estimated `actual_tokens` and configured `budget_tokens`. These estimates use the configured tokenizer. The current Hermes host catches hook errors and can fall back to its original request, so Mneme does not guarantee strict provider-side budget enforcement.

When checking historical evidence, distinguish what the user said from what was requested or completed: a user message is evidence of their words, an assistant message may be a proposal, and a tool call records an attempted action. Verify outcomes from the corresponding tool output or original event before describing them as completed.

## Architecture

See [`docs/`](docs/) for component details:

- `store.py` — SQLite event store and session lineage
- `index.py` — semantic and lexical indexes
- `segmenter.py` — drift-based topic segmentation
- `classifier.py` — deterministic intent signals
- `router.py` — automatic prompt retrieval and scoring
- `prompt_builder.py` — token-budgeted prompt assembly
- `engine.py` — lifecycle, native hooks, and legacy compression fallback
- `graph.py` — execution graph and dependency traversal
- `tools.py` — agent memory tools

## Tests

Run the unit suite from the Hermes Python environment:

```bash
~/.hermes/hermes-agent/venv/bin/pytest tests/unit -q
```

## Updating

For a release update, first drain Gateway turns and stop Gateway and other database writers. Back up the Mneme database and Hermes `state.db` through SQLite's backup API, including WAL state, and preserve your Hermes and plugin configuration files. Then fetch tags and check out the desired release tag (this leaves Git in detached-HEAD state), run the installer, and restart Gateway:

```bash
cd ~/.hermes/plugins/hermes-mneme
git fetch --tags
git checkout v0.5.3
./install.sh
hermes gateway restart
```

Use the tag for a released version; `main` may contain unreleased changes. For the first FTS index build on a large existing database, follow the additional prewarm and integrity checks in [retrieval behavior](docs/memory-retrieval.md).

## Contributing

Contributions and bug reports are welcome. Open an issue or submit a focused pull request against `main`. Before submitting, run the unit suite and `ruff check`; check both macOS and Linux when changing installation or filesystem behavior.

## License

[Apache-2.0](LICENSE)
