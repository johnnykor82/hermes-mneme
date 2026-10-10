# Explicit legacy segment migration

The plugin never runs this migration automatically. `scripts/migrate_legacy_segments.py` defaults to a read-only dry-run and imports neither `ContextStore` nor the engine. It repairs only legacy `seg_1` session collisions. It does not reconstruct historical topic boundaries.

Use the Hermes Python environment with `sqlite-vec` installed. The extension is loaded on every source, backup and verification connection when available; a vec0 database is refused if the extension is unavailable.

```sh
/Users/openclaw/.hermes/hermes-agent/venv/bin/python scripts/migrate_legacy_segments.py /path/to/copied-plugin.db > dry-run.json
/Users/openclaw/.hermes/hermes-agent/venv/bin/python scripts/migrate_legacy_segments.py /path/to/copied-plugin.db --apply --backup /path/to/new-pre-migration-backup.db > applied.json
```

Apply requires an explicitly named, **nonexisting** backup destination; existing files and symlinks are never overwritten. Validate a copy first. For the live database, stop Gateway and other writers before applying, and restart them after verification so active Engine bindings/centroid caches are recreated. `BEGIN IMMEDIATE` also excludes concurrent writers during the backup and transaction; a busy database is refused rather than partially migrated.

The SQLite backup API captures the full committed database including WAL pages; copying only a `.db` file is insufficient. While the migration connection holds its write reservation, a separate read connection backs up the consistent committed pre-migration state. Backup integrity is checked before mutation.

For each session independently, migration maps `seg_1` to `seg_<session_id>_1`. If that identifier already exists anywhere, it allocates the first unused `seg_<session_id>_legacyN`. Metadata updates use the **event IDs belonging to that session's legacy segment**. `execution_state.state_json.segment_id` changes only in that session row and only if it equals `seg_1`. Event IDs/content, graph edges and session lineage do not change. Current stored ingestion cursors contain message indices/hashes, not segment IDs, and are preserved as JSON fields. Numeric counter recovery happens when the engine next binds; existing segment counters are never reset by this utility.

Event scope/time indexes are created before the first per-session update to prevent repeated full-history scans. All updates, required event scope/time indexes and the `mneme_migrations` record commit in one transaction. The report includes `EXPLAIN QUERY PLAN` for the scoped centroid query. Any exception rolls the entire migration back; the valid pre-migration backup remains. A second run with no legacy rows is a no-op.

The JSON report includes per-session targets and affected rows; before/after counts; streaming event digest excluding only `segment_id`; graph/lineage digests; authoritative embedding BLOB and vector metadata digests excluding only segment IDs; canonical execution-state digest excluding only root segment ID; `PRAGMA integrity_check`; orphan embeddings, vector metadata, vec0 rowids and graph endpoints. Apply refuses a failing integrity check and refuses any change in protected digests/counts/orphan counts. Existing orphan counts are reported, not repaired; an unrelated existing orphan is not attributed to this migration. Reports contain session IDs, so keep them private.

## Upgrading from 0.5.1

Upgrading from 0.5.1 to 0.5.2 requires no additional segment migration. Keep existing backups. The resume fix preserves old event IDs and graph rows; retained text that an older parser omitted can be recovered on resume. It prevents new replay duplicates but does not delete historical duplicate rows. Where old call IDs are absent and repeated content is indistinguishable, matching is chronological rather than a reconstruction of the original provider identity.

## Restore

Stop Gateway and every database writer. Preserve the failed/current database and its WAL/SHM as a separate incident snapshot. Restore the named backup to a **new database path**, verify it with this utility's read-only dry-run, then point the plugin at that restored path and restart Gateway. Using a new path prevents stale WAL/SHM files from being accidentally paired with the backup. Never overwrite a live SQLite database.

## Regression evidence

`tests/unit/test_audit_migration.py`: initial run failed at collection because the migration module did not exist. After implementation: **9 passed** in Hermes Python, temporary SQLite fixtures only.

Covered: 441 shared legacy sessions; collision fallback; session-specific metadata/state; preserved event/graph digests; second-run idempotence; byte-identical read-only dry-run; required exclusive backup including rejection of dangling symlinks; injected exception after first session's updates with full rollback and usable backup; outstanding WAL included in backup; real sqlite-vec extension loaded across source/backup/reopen, 882 vector rows retained and no dangling vector mappings; canonical embedding mutation detected and fully rolled back; centroid query uses the new composite index, created before any session updates. Vector metadata has an event/model lookup index before updates.

No production database is changed by these tests. Production-copy counts and timings belong to the final release audit report.

Plain databases work even when Python has no SQLite extension-loading API. A database with vec0 requires an extension-enabled Python plus sqlite-vec; the utility fails explicitly rather than misreporting database corruption.
