# Index/config audit verification — 2026-10-10

Scope: F05, F08, F10, F11 and F17. Temporary SQLite fixtures and mocked HTTP only; no production database writes or API calls.

## Findings and fixes

| Finding | Independent result | Change |
|---|---|---|
| F05 | Confirmed by regression: A has ten matching vectors while sixty closer B vectors exhaust global KNN, both finite and unlimited modes. Baseline returns an incomplete/empty A result. | Scope canonical `embedding_index` through events before ranking. sqlite-vec uses scalar `vec_distance_cosine`, with SQL ORDER/LIMIT after scope; fallback streams exact cosine and retains only k candidates for finite k. Same session/segment/model filters and deterministic tie break apply to single, multi and global search. |
| F08 | Confirmed: YAML overwrites environment, constructor values have lower precedence, string-list environment values raise ValueError. | Defaults → YAML → environment → explicit constructor config. Parse bool/int/float/string lists by default schema; ignore invalid typed/bounded values with a diagnostic that never echoes the value. Import missing logging module. |
| F10 | Confirmed reproducibly: embedding the same event/model three times creates three vec rows and mappings on baseline. This is a reachable API defect, not evidence of production reindexing frequency. | Delete prior event/model vec rows and mappings within the embedding transaction, then insert one new mirror. Queued background embeddings resolve current event segment under a SQLite write transaction after HTTP completes, preventing stale compression metadata. Canonical PK remains unchanged; repeated writes have one mapping. |
| F11 | Duplicate embedding calls are reachable in segmentation/router/search. No production timing attribution established. | Reuse successful single-query embeddings in a thread-safe, 128-entry instance cache keyed by model/endpoint/dimension/exact text. No failure caching, defensive copies, inputs above 64k characters are not retained. Existing batch ingestion stays uncached. |
| F17 | Confirmed: endpoint/key/model settings ignored and fixed dimension not validated. | EmbeddingIndex accepts instance config, exposes `model_id`, validates configured dimension and finite values, honors response indices. Fresh Store schema accepts configured dimension; existing vec0 schema is preserved. Alternate dimensions remain searchable from canonical embeddings using exact scalar/Python cosine, without destructive vec0 replacement. |

## Verification

The original ten regression cases ran on the unmodified checkout: **10 failed** (four scoped recall cases, reembedding, config precedence, malformed config, three alternate-dimension cases). Output was captured in `/tmp/mneme-index-red.txt`; the failures and their causes are recorded here for the repository.

Current targeted command (Hermes Python with pytest from the existing shared installation):

```sh
python -m pytest -q \
  tests/unit/test_audit_index_config.py tests/integration/test_embedding_batch.py \
  tests/integration/test_store.py
```

**28 passed** after the additional scope/model/dimension, malformed YAML, HTTP response ordering and cache/background-reassign/fresh-dimension regressions. sqlite-vec was available and the vec regressions were executed, not skipped. The regression suite compares actual sqlite-vec scalar results with the independent Python exhaustive oracle, including top_k 1/5/None/0 and foreign-session dominance. Existing mocked batching tests remain green.

## Configuration and compatibility

```yaml
embedding_endpoint: http://127.0.0.1:8000 # also accepts /v1 or /v1/embeddings
embedding_model: jina-embeddings-v5-text-small-retrieval-mlx
embedding_dim: 1024                    # match provider; 768/1024/1536 tested
embedding_api_key: your-local-key
```

Existing 1024-dimensional vec0 tables are never dropped automatically when dimension changes. The canonical BLOB index supports different model dimensions; searches and centroids exclude nonmatching dimensions/model IDs. A changed provider/model does not fabricate missing embeddings or silently relabel historical vectors: old-model events remain stored, but need explicit reindexing for semantic retrieval under the new model. Use a different model ID when dimensions/embedding semantics change.

`top_k=None` and `0` retain the documented unlimited semantics. Output and SQL sorting storage can therefore grow with matching candidate count; no undocumented cap was introduced. Finite k avoids accumulating all Python result dictionaries. Exact global scanning is linear in candidate vectors and dimension.

## Synthetic performance evidence and limits

Mac host, Hermes Python 3.11, fresh temporary SQLite, 100,000 deterministic 1024-dimensional embeddings, 100 sessions, 1,000 candidates/session, top_k=5, no HTTP. Identical vectors stress deterministic sorting; this is a diagnostic fixture, not a semantic-quality benchmark. A composite events scope index was present. Five scoped runs and three global runs:

| Path | Scope | Samples ms | Median ms | Max ms |
|---|---|---|---|---|
| Python cosine | Single session | 100.35, 91.39, 92.37, 91.43, 92.24 | 92.24 | 100.35 |
| Python cosine | Global | 8759.21, 8759.83, 8718.66 | 8759.21 | 8759.83 |
| sqlite-vec scalar cosine | Single session | 8.25, 8.31, 8.43, 7.81, 8.45 | 8.31 | 8.45 |
| sqlite-vec scalar cosine | Global | 433.58, 444.45, 458.67 | 444.45 | 458.67 |

Sample sizes do not justify p95/p99 claims. One-million-vector latency, production profiling, external endpoint integration and Linux/macOS matrix CI were not established by these local measurements. These figures do not explain Gateway's reported 14.59-second compression duration.
