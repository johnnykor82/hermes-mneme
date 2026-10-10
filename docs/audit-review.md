# Independent release review — 2026-10-10

Reviewed the integrated working-tree diff and existing/regression unit and integration tests, especially Engine lifecycle, Store reassignment, configured embedding endpoints, scoped vector search, and actual installed Hermes native-hook behavior. Production data and external endpoints were not modified.

No remaining release-blocking defects were found within the reviewed scope after the fixes below. This is a scoped code/regression review, not confirmation that every acceptance scenario or production integration has been measured.

## Issues found during review

1. **Reassignment could merge distinct old segments (fixed).** Temporary fixture: old session A contains `seg_A_1` and historical `seg_B_1`; reassign A→B. Sequential UPDATE-by-segment first renamed the first segment to `seg_B_1`, then the second UPDATE unintentionally matched both rows, producing one segment. Store now snapshots a source→target mapping and applies it without intermediate segment-name matching. Recheck preserved two segments (`seg_B_1` and `seg_B_legacy1`).

2. **Failed reassignment reported committed rows and allowed Engine rebind (fixed).** A malformed saved state JSON caused reassignment to roll back events, yet return a positive row count. Engine then advanced its binding. Store now rolls back and raises; compression lifecycle paths propagate failure before rebinding. Recheck raised JSONDecodeError and preserved counts A=1/B=0. The earlier false-success result was 1 with the same counts.

3. **Tool output→assistant graph continuation across compression (fixed).** Temporary fixture: user→assistant tool-call c1→tool output c1 in S0; compression S0→S1; retained transcript replay plus a new assistant completion. Initial review observed only the call→output edge: binding reset `_last_tool_outputs`, while outstanding-call recovery did not restore unconsumed tool outputs. Engine now restores unconsumed tool-output anchors when binding the new session; a lifecycle regression covers the sequence. Independently rerunning the same fixture produced both `tool_output` and `decision` edges in S1.

4. **Native overflow does not stop the installed Hermes host (documented host limitation).** The installed `agent/conversation_loop.py` `_apply_context_engine_selection` catches all exceptions and returns original request messages. Executing that actual function with a fake engine raising `PromptBudgetExceeded(10000, 50)` confirmed it returns the original message-list object. Mneme's exception/logging is an explicit overflow signal, not an end-to-end fail-closed provider-send guarantee. No host changes are included; strict provider enforcement remains separate integration work.

## Checks and bounds

- Initial integrated run: **126 unit/integration tests passed**; final independent rerun after review fixes: **131 passed** on Hermes Python with deterministic mocks. SQLite extension tests executed. Temporary repros also independently confirmed collision separation, transactional rollback/failure propagation, and graph continuation recovery.
- Engine native request cache keys include session/host session, policy/native budgets, request/conversation/incoming payloads. Explicit `conversation_messages=None` retains the fallback contract; empty lists remain intentional empty input.
- Embedding endpoint normalization accepts base URL, `/v1`, and `/v1/embeddings`; configured instance model/dimension/key are exercised without network calls.
- Scoped scalar sqlite-vec cosine and exhaustive Python agree on tested session/segment/model filters and finite/unlimited k. Zero vectors are excluded by both paths. Model changes use instance model IDs; old-model historical events remain stored.
- Config known fields validate numeric bounds, list element types, finite weights, booleans and dimensions. Constructor overrides environment, which overrides YAML. Malformed values keep the preceding valid value and produce sanitized diagnostics.
- `git diff --check` passed during review.

Not established by this review: production Gateway timing, provider-native tokenizer accuracy, labeled RU/EN segmentation precision/recall, one-million-vector production latency, or the complete cross-platform CI matrix. These limits must remain explicit in release documentation.
