# Configuration Reference

Selected configuration options for Hermes-Mneme **0.5.3**. The complete schema and shipped defaults are in [`config.py`](../config.py); examples below are illustrative, not deployment-specific settings.

## Configuration Sources (Priority Order)

Highest to lowest priority:

1. **Explicit constructor values** (`PluginConfig(config_dict=...)`).
2. **Environment variables** prefixed with `HERMES_CTX_`.
3. **Plugin YAML**: `~/.hermes/plugins/hermes-mneme/config.yaml`.
4. **Shipped defaults**: `DEFAULT_CONFIG` in `config.py`.

Invalid known setting values are ignored with a diagnostic; environment settings override YAML. Numeric lists such as `drift_weights` and string lists such as `memory_tool_names` accept comma-separated environment values.

## Environment Variables

All plugin settings can be set via environment variables prefixed with `HERMES_CTX_`.

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `HERMES_CTX_ACTIVE_WINDOW_TOKENS` | Absolute budget; `0` selects percentage mode | `0` | Integer |
| `HERMES_CTX_CONTEXT_WINDOW_USAGE_PERCENT` | Fraction used when absolute budget is disabled | `0.70` | Float |
| `HERMES_CTX_PROTECTED_TAIL_TURNS` | Requested number of recent protected turns | `64` | Integer |
| `HERMES_CTX_STATE_BUDGET_RATIO` | Budget ratio for execution state | `0.05` | Float |
| `HERMES_CTX_RETRIEVED_BUDGET_RATIO` | Budget ratio for retrieved context | `0.30` | Float |
| `HERMES_CTX_TOKEN_COUNTER` | Declared setting; the current prompt builder uses tiktoken | `tiktoken` | String |
| `HERMES_CTX_TOKENIZER_MODEL` | Model name for tokenizer (e.g., `cl100k_base`) | `cl100k_base` | String |
| `HERMES_CTX_EMBEDDING_PROVIDER` | Provider type (`jina_compatible`, `ollama`, `openai_compatible`) | `jina_compatible` | String |
| `HERMES_CTX_EMBEDDING_MODEL` | Embedding model name | `jina-embeddings-v5-text-small-retrieval-mlx` | String |
| `HERMES_CTX_EMBEDDING_ENDPOINT` | API endpoint for embeddings | `http://127.0.0.1:8000` | String |
| `HERMES_CTX_EMBEDDING_API_KEY` | API key for embedding service | `1234` | String |
| `HERMES_CTX_EMBEDDING_DIM` | Vector dimension; must match the embedding model | `1024` | Integer |
| `HERMES_CTX_SEGMENTATION_ENABLED` | Enable session segmenter | `true` | Boolean |
| `HERMES_CTX_DRIFT_THRESHOLD` | Normalized embedding drift threshold for topic boundaries | `0.35` | Float |
| `HERMES_CTX_LLM_ENRICHMENT_ENABLED` | Enable LLM enrichment of state | `true` | Boolean |
| `HERMES_CTX_DELTA_EXTRACTION_ENABLED` | Reserved setting; no delta extractor is activated | `false` | Boolean |
| `HERMES_CTX_TOOL_OUTPUT_COMPRESS_THRESHOLD_TOKENS` | Token threshold to summarize tool outputs | `500` | Integer |
| `HERMES_CTX_TOOL_OUTPUT_SUMMARY_TOKENS` | Max tokens for tool output summaries | `100` | Integer |
| `HERMES_CTX_REINDEX_ON_MODEL_CHANGE` | Reindex all events on model change | `false` | Boolean |
| `HERMES_CTX_DEBUG_MODE` | Enable synchronous segmenter and full tracing | `false` | Boolean |
| `HERMES_CTX_TRACE_LOG_MAX_MB` | Max size of trace.jsonl before rotation | `10` | Integer |
| `HERMES_CTX_TRACE_LOG_MAX_ROTATIONS` | Number of rotated log files to keep | `3` | Integer |
| `HERMES_CTX_MEMORY_ACCESS_HINT_ENABLED` | Inject `[MEMORY ACCESS]` hint into every system message | `true` | Boolean |
| `HERMES_CTX_GOAL_TRAIL_SIZE` | Number of latest unique goals shown in `[GOAL TRAIL]` | `3` | Integer |
| `HERMES_CTX_CHECKPOINT_AFTER_N_MEMORY_CALLS` | Inject a `[CHECKPOINT]` hint after N memory calls; this is not a search-attempt cap (`0` disables) | `5` | Integer |
| `HERMES_CTX_MEMORY_TOOL_NAMES` | CSV of tool names counted for the checkpoint hint | `context_search,fetch_event,expand_context,list_segments,get_goal_history` | List |

## YAML Config Example

Create `~/.hermes/plugins/hermes-mneme/config.yaml`:

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
  engine: hermes-mneme
```

## Retrieval and provider settings

`embedding_endpoint`, `embedding_model`, `embedding_api_key` and `embedding_dim` belong to each index instance. Changing model semantics or dimension requires a deliberate reindex plan; it is not a transparent model swap. Configure real credentials locally, never in committed examples.

The shipped reranker defaults are `reranker_enabled: true`, `reranker_endpoint: http://127.0.0.1:4000/v1/rerank`, and `reranker_model: rerank`. Set the endpoint and credentials for your deployment or disable reranking explicitly. Tool search retains fused semantic/BM25 order if reranking is disabled or fails. `router_top_k: 0` is the automatic router's unlimited candidate setting, not the default `context_search` result count (`k=5`). See [retrieval behavior](memory-retrieval.md) for candidate pools, RRF, reranking and search scope.

`llm_enrichment_enabled` is enabled by default and can make auxiliary LLM requests. Empty `enricher_endpoint`, `enricher_model` and `enricher_api_key` use the model configuration supplied by Hermes. Set it to `false` to disable this enrichment.

The shipped `drift_threshold` is **0.35**. It applies directly to normalized `(1-cosine_similarity)/2` when only embeddings are available. Tune it for your embedding model and workload; a locally calibrated value is not a universal default. `segmentation_enabled: false` disables both hard and soft topic switches.

On native hosts, Mneme selects the outgoing request without shortening saved Hermes history. Gateway Hygiene may therefore still compress that history independently. Budget estimates use the configured tokenizer; if prompt assembly raises `PromptBudgetExceeded`, the host may fall back to its original request. Configure the real model context window and keep host overflow protection enabled.
