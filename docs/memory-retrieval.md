# Finding and using distant memory

`context_search` combines semantic similarity with lexical BM25 over stored event content. Session, segment, and event exclusions apply before candidate limits. Lexical search can recover a fact even while its embedding job is pending. Only the active embedding model participates in semantic similarity.

For ordinary bounded requests, each search supplies up to 60 semantic and 60 lexical candidates. Reciprocal rank fusion (`1 / (60 + rank)` per search) combines their orders without treating cosine and BM25 scores as comparable. The best 40 fused candidates enter the configured reranker, which receives the first 2,000 characters of each source. The requested `k` is applied after reranking. Larger explicit `k` increases the fused pool; `k=0` or `null` retains unlimited search semantics, but only the first 40 candidates are reranked. These explicit wide searches can be expensive.

Without a configured reranker, fused order is returned. A failed, incomplete, or invalid reranker response also retains fused order. `score` in tool results represents the final ranking stage; it is not a probability or proof of correctness. Automatic prompt routing retains its existing policy and scoring.

## Source previews and verification

Search results include the actual `segment_id`, source session, role, tool name, timestamp, original character count, and a `truncated` flag. The bounded preview selects a query-relevant portion of the source instead of always returning its beginning. Simple word prefixes accommodate Russian inflections; they do not replace semantic retrieval.

For JSON lists of source records, previews preserve complete record boundaries instead of cutting across serialized objects. They show the first and most query-relevant records in original order, with original URL and position plus a separate one-based array index. Titles and descriptions may be shortened; omitted fields and records are explicit. An oversized URL is omitted rather than shortened into a different address. These projections always require verification against the original event.

All previews are marked as untrusted historical data. A recognized Hermes external-result warning is shortened in the preview while its data boundary remains explicit. Original stored content is never changed. `fetch_event(full=true)` retrieves that original content; the default fetch still caps long payloads at 4,000 characters and reports truncation.

Verify a relevant source before answering a historical factual question. Ranking expresses relevance, not time order or agreement. Check topic, timestamps, and whether a statement was a proposal or a decision. Older assistant claims and legacy graph links can be inaccurate. If the sources do not establish the requested fact, report that uncertainty.

Native Hermes request hooks ingest the current question before tool execution. The engine tracks the host's original incoming message identity and the actual stored event IDs of its logical turn. Synthetic user continuations do not move that boundary. A distinct next incoming identity starts a new turn, including after a failed turn without a completion callback; completion and session reset/end clear exclusions. Compression preserves them for reassigned events, while a copied engine starts unbound. Search excludes these tracked IDs rather than guessing from the latest stored user row. Missing or ambiguous incoming identity produces no guessed exclusion; legacy hosts retain their existing behavior.

`expand_context(mode='segment')` resolves a unique segment to its actual source session, including a prior session. A shared legacy segment ID requires `seed_event_id` to disambiguate it; contradictory seed and segment IDs return an error. Segment expansion returns a bounded skeleton, not the complete segment: `total_events`, `returned_events`, and `is_partial` expose omitted events or shortened content. A missing fact in this skeleton is not proof of absence. Use searches, seed neighbors and original-event fetches to investigate further.

## Derived lexical index

The FTS5 index is built lazily, in one SQLite transaction, on the first lexical search. Insert, delete, and content-update triggers keep it synchronized. Scope changes are resolved through the events table, so session reassignment does not require rewriting lexical content. Event IDs, source content, embeddings, graph edges, and saved state are preserved.

Initial construction costs time and disk space proportional to stored text. Measure this cold-start separately from steady-state query latency. If initial construction fails, including when FTS5 is unavailable on a database without this derived index, semantic candidates remain usable. Once the index exists, retain FTS5 support in the SQLite runtime: its event-maintenance triggers require FTS5, so moving an indexed database to a runtime without it can prevent event writes. Existing SQLite backup procedures should continue to include WAL consistently. The lexical index is derived data; preserve backups and validate integrity before any deployment operation.

For a large existing archive, drain Gateway turns and stop database writers before the first index build. Preserve a SQLite backup made through the backup API, initialize the lexical index with the installed Hermes Python runtime, and verify original-row hashes and SQLite integrity before restarting Gateway. This avoids competing with live ingestion during the backfill transaction. Validate this procedure on a copy first. Rollback requires restoring the matching plugin version and complete backed-up database with writers stopped; do not copy only a live database file while ignoring its WAL.

## Working with evidence

Search results are bounded excerpts. A top-k result that omits a fact does not show that the fact is absent from memory. When evidence is insufficient, continue with a larger `k`, broaden the segment scope to `all`, try alternative query wording, or use `expand_context` to inspect related events. Histories with thousands of messages may require multiple sequential searches. There is no fixed search attempt limit; continue while useful avenues remain. If a reasonable search finds no support, describe the scope searched and say the available record does not establish the answer.

Use source events to distinguish what was said from what was requested or completed. A `user_message` is evidence of the user's words; call it a requirement only when the user framed it as one. An `assistant_message` may be a proposal or claim and does not prove completion. A `tool_call` records a requested action; `tool_output` records a returned result that may not establish an external side effect. Distinguish proposals from confirmed decisions and completed actions, and do not present unsupported historical facts.
