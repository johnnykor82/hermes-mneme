# Configuration Reference

This document lists all configuration options for the Hermes Context Engine Plugin.

## Configuration Sources (Priority Order)

1.  **Environment Variables** (Highest priority)
2.  **User Config**: `~/.hermes/plugins/custom_router/config.yaml`
3.  **Shipped Defaults**: `config.defaults.yaml` (inside plugin dir)

## Environment Variables

All plugin settings can be set via environment variables prefixed with `HERMES_CTX_`.

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `HERMES_CTX_ACTIVE_WINDOW_TOKENS` | Total token budget per turn | `16000` | Integer |
| `HERMES_CTX_PROTECTED_TAIL_TURNS` | Number of recent turns to always include | `6` | Integer |
| `HERMES_CTX_STATE_BUDGET_RATIO` | Budget ratio for execution state | `0.05` | Float |
| `HERMES_CTX_RETRIEVED_BUDGET_RATIO` | Budget ratio for retrieved context | `0.45` | Float |
| `HERMES_CTX_TOKEN_COUNTER` | Token counter method (`tiktoken`, `huggingface`, `chars`) | `tiktoken` | String |
| `HERMES_CTX_TOKENIZER_MODEL` | Model name for tokenizer (e.g., `cl100k_base`) | `cl100k_base` | String |
| `HERMES_CTX_EMBEDDING_PROVIDER` | Provider type (`jina_compatible`, `ollama`, `openai_compatible`) | `jina_compatible` | String |
| `HERMES_CTX_EMBEDDING_MODEL` | Embedding model name | `jina-embeddings-v5-text-small-retrieval-mlx` | String |
| `HERMES_CTX_EMBEDDING_ENDPOINT` | API endpoint for embeddings | `http://127.0.0.1:8000` | String |
| `HERMES_CTX_EMBEDDING_API_KEY` | API key for embedding service | `1234` | String |
| `HERMES_CTX_SEGMENTATION_ENABLED` | Enable session segmenter | `true` | Boolean |
| `HERMES_CTX_DRIFT_THRESHOLD` | Embedding drift threshold for hard triggers | `0.35` | Float |
| `HERMES_CTX_LLM_ENRICHMENT_ENABLED` | Enable LLM enrichment of state | `false` | Boolean |
| `HERMES_CTX_DELTA_EXTRACTION_ENABLED` | Enable delta extraction on continuation | `false` | Boolean |
| `HERMES_CTX_TOOL_OUTPUT_COMPRESS_THRESHOLD` | Token threshold to summarize tool outputs | `500` | Integer |
| `HERMES_CTX_TOOL_OUTPUT_SUMMARY_TOKENS` | Max tokens for tool output summaries | `100` | Integer |
| `HERMES_CTX_REINDEX_ON_MODEL_CHANGE` | Reindex all events on model change | `false` | Boolean |
| `HERMES_CTX_DEBUG_MODE` | Enable synchronous segmenter and full tracing | `false` | Boolean |
| `HERMES_CTX_TRACE_LOG_MAX_MB` | Max size of trace.jsonl before rotation | `10` | Integer |
| `HERMES_CTX_TRACE_LOG_MAX_ROTATIONS` | Number of rotated log files to keep | `3` | Integer |
| `HERMES_CTX_MEMORY_ACCESS_HINT_ENABLED` | Inject `[MEMORY ACCESS]` hint into every system message | `true` | Boolean |
| `HERMES_CTX_GOAL_TRAIL_SIZE` | Number of latest unique goals shown in `[GOAL TRAIL]` | `3` | Integer |
| `HERMES_CTX_CHECKPOINT_AFTER_N_MEMORY_CALLS` | Force `[CHECKPOINT]` after N consecutive memory-tool calls (`0` disables) | `5` | Integer |
| `HERMES_CTX_MEMORY_TOOL_NAMES` | CSV of tool names that count toward the loop guard | `context_search,fetch_event,expand_context,list_segments,get_goal_history` | List |

## YAML Config Example

Create `~/.hermes/plugins/custom_router/config.yaml`:

```yaml
active_window_tokens: 16000
protected_tail_turns: 6
state_budget_ratio: 0.05
retrieved_budget_ratio: 0.45

token_counter: "tiktoken"
tokenizer_model: "cl100k_base"

embedding_provider: "jina_compatible"
embedding_model: "jina-embeddings-v5-text-small-retrieval-mlx"
embedding_endpoint: "http://127.0.0.1:8000"
embedding_api_key: "1234"

segmentation_enabled: true
drift_threshold: 0.35

llm_enrichment_enabled: false
delta_extraction_enabled: false

tool_output_compress_threshold_tokens: 500
tool_output_summary_tokens: 100

reindex_on_model_change: false

debug_mode: false
trace_log_max_mb: 10
trace_log_max_rotations: 3

# Stage B — memory navigation
memory_access_hint_enabled: true
goal_trail_size: 3
checkpoint_after_n_memory_calls: 5
memory_tool_names:
  - context_search
  - fetch_event
  - expand_context
  - list_segments
  - get_goal_history
```

## Hermes Main Config

In `~/.hermes/config.yaml`, activate the plugin:

```yaml
context:
  engine: custom_router
```
