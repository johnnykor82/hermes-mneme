# Segmenter audit verification (2026-10-10)

Baseline `6de064f`, deterministic temporary SQLite fixtures only; no production DB writes or embedding requests.

- **F02 confirmed**: default ingestion fed absent entity/explicit signals as zero into a `.4 * drift` score. Orthogonal query failed to switch on baseline. Ingestion now compares normalized embedding drift directly with configured `drift_threshold`; composite weights apply only when optional classifier signals are supplied. Existing default `.35` is retained, now means cosine similarity below `.30`. This fixes inaccessible signal weighting, **not a measured semantic calibration**. RU/EN gold-set ROC/precision/recall requires a labeled dataset and actual deployed embedding model; no such dataset was supplied. Deterministic vectors establish arithmetic only. Continuation fixture uses aligned vectors; no word-count heuristic is introduced.
- **F03 confirmed**: counter starts at zero and duplicates `_1`. `bind_session` restores highest own-session numeric suffix and persistent allocated counter. Atomic `segment_sequences` reservation also distinguishes two engines that both bind before either inserts an event. Reserved empty segments are preserved across restart.
- **F04 confirmed**: 210 foreign vectors with identical legacy segment ID displaced own-session vectors in baseline (centroid `[0,1]` instead of `[1,0]`). New query scopes session, segment and configured/indexer model; incompatible dimensions are excluded.
- **F07 confirmed**: hard/soft switches ignore the disable flag and weights/threshold. New path honors all three, adds Russian `новая тема` explicit phrase, and makes disabled segmentation suppress both hard/soft triggers.
- **F09 confirmed**: baseline cache hit calls BLOB loader; unchanged event count hides late embeddings. Scoped per-thread centroid LRU checks persistent SQLite reader `PRAGMA data_version` before loading BLOBs. External commits (including same-length vector replacement) invalidate; unrelated commits conservatively invalidate too. Reader holds no transaction, and `close()` releases current-thread resources. Database indices are addressed separately in Store changes.
- **F11**: duplicate embedding request hypothesis delegated to indexer owner for shared normalized-query/model cache. Segmenter itself continues to skip requests on centroid cold start.

## Tests

Hermes runtime: `/Users/openclaw/.hermes/hermes-agent/venv/bin/python`, with pytest supplied through base Python site-packages in `PYTHONPATH`.

`tests/unit/test_audit_segmenter.py`: 12 tests pass after patch. Baseline module loaded directly from `git show 6de064f:segmenter.py`: initial 10 tests fail (some assert the new binding/close API); direct SQL/cache checks run with only legacy scope attributes supplied and reproduce wrong centroid and unnecessary BLOB reads. Initial red suite before changes: 6 failed, 1 passed; the disable regression was then tightened to avoid baseline's duplicate `_1` masking its failure.

Additional compatibility run with classifier/router dependency tests: 32 passed before two additional model/replacement regressions were added. Separate baseline reproduction confirms other-model contamination and stale same-count replacement centroid (both tests fail on baseline, pass after patch). Native/legacy Engine binding integration belongs to the Engine changes.

No production latency or segmentation-quality claims are made here. Persistent cache invalidation observes commits, rather than a model-specific indexing revision; this favors correctness over hit rate during unrelated writes.
