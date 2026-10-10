"""Explicit, backed-up migration of session-shared legacy segment identifiers.

Importing this module never initializes or modifies a database. The migration is
not run by ContextStore or plugin startup.
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


MIGRATION_ID = 'legacy-segment-scope-v1'


def _connect(path, *, readonly=False):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True)
    try:
        try:
            import sqlite_vec
        except ImportError:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE sql LIKE '%USING vec0%' LIMIT 1").fetchone():
                raise RuntimeError('sqlite-vec is required to inspect this database')
        else:
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
        return conn
    except Exception:
        conn.close()
        raise


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _digest(conn, table, excluded=(), transforms=None):
    schema = list(conn.execute(f'PRAGMA table_info("{table}")'))
    columns = [row[1] for row in schema if row[1] not in excluded]
    digest = hashlib.sha256()
    selected = ','.join('"' + column.replace('"', '""') + '"' for column in columns)
    primary = [row[1] for row in sorted(schema, key=lambda row: row[5]) if row[5]]
    ordering = ','.join('"' + column.replace('"', '""') + '"' for column in primary) if primary else selected
    # Hash BLOBs directly, avoiding JSON/hex expansion of large embedding stores.
    for row in conn.execute(f'SELECT {selected} FROM "{table}" ORDER BY {ordering}'):
        digest.update(b'ROW')
        for column, value in zip(columns, row):
            if transforms and column in transforms:
                value = transforms[column](value)
            if value is None:
                marker, encoded = b'N', b''
            elif isinstance(value, bytes):
                marker, encoded = b'B', value
            elif isinstance(value, str):
                marker, encoded = b'S', value.encode()
            elif isinstance(value, int):
                marker, encoded = b'I', str(value).encode()
            else:
                marker, encoded = b'F', repr(value).encode()
            digest.update(marker)
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return digest.hexdigest()


def _protected_state(raw):
    state = json.loads(raw)
    state.pop('segment_id', None)
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _snapshot(conn):
    tables = _tables(conn)
    counts = {table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
              for table in ('events', 'embedding_index', 'vec_items_meta', 'execution_graph', 'execution_state', 'session_lineage', 'vec_items')
              if table in tables}
    orphan = {}
    for table in ('embedding_index', 'vec_items_meta'):
        if table in tables:
            orphan[table] = conn.execute(f'SELECT COUNT(*) FROM "{table}" i LEFT JOIN events e ON e.id=i.event_id WHERE e.id IS NULL').fetchone()[0]
    if 'vec_items' in tables and 'vec_items_meta' in tables:
        orphan['vec_items_rowid'] = conn.execute('SELECT COUNT(*) FROM vec_items_meta m LEFT JOIN vec_items v ON v.rowid=m.rowid WHERE v.rowid IS NULL').fetchone()[0]
    if 'execution_graph' in tables:
        orphan['execution_graph'] = conn.execute('''SELECT COUNT(*) FROM execution_graph g
            LEFT JOIN events a ON a.id=g.from_event_id LEFT JOIN events b ON b.id=g.to_event_id
            WHERE a.id IS NULL OR b.id IS NULL''').fetchone()[0]
    return {'counts': counts, 'orphans': orphan,
            'event_digest': _digest(conn, 'events', excluded=('segment_id',)),
            'graph_digest': _digest(conn, 'execution_graph') if 'execution_graph' in tables else None,
            'lineage_digest': _digest(conn, 'session_lineage') if 'session_lineage' in tables else None,
            'embedding_digest': _digest(conn, 'embedding_index', excluded=('segment_id',)) if 'embedding_index' in tables else None,
            'vec_meta_digest': _digest(conn, 'vec_items_meta', excluded=('segment_id',)) if 'vec_items_meta' in tables else None,
            'state_digest': _digest(conn, 'execution_state', transforms={'state_json': _protected_state}) if 'execution_state' in tables else None,
            'integrity_check': [row[0] for row in conn.execute('PRAGMA integrity_check')]}


def _plan(conn):
    occupied = {row[0] for row in conn.execute('SELECT DISTINCT segment_id FROM events')}
    groups = []
    for sid, count in conn.execute("SELECT session_id,COUNT(*) FROM events WHERE segment_id='seg_1' GROUP BY session_id ORDER BY session_id"):
        new = f'seg_{sid}_1'
        suffix = 1
        while new in occupied:
            new = f'seg_{sid}_legacy{suffix}'
            suffix += 1
        occupied.add(new)
        groups.append({'session_id': sid, 'old_segment_id': 'seg_1', 'new_segment_id': new, 'events': count})
    return groups


def _apply_group(conn, group):
    sid, new = group['session_id'], group['new_segment_id']
    tables = _tables(conn)
    # Update metadata BEFORE events, with the exact session/legacy-event scope.
    for table in ('embedding_index', 'vec_items_meta'):
        if table in tables:
            conn.execute(f'''UPDATE "{table}" SET segment_id=? WHERE event_id IN
                (SELECT id FROM events WHERE session_id=? AND segment_id='seg_1')''', (new, sid))
    conn.execute("UPDATE events SET segment_id=? WHERE session_id=? AND segment_id='seg_1'", (new, sid))
    if 'execution_state' in tables:
        row = conn.execute('SELECT state_json FROM execution_state WHERE session_id=?', (sid,)).fetchone()
        if row:
            state = json.loads(row[0])
            if state.get('segment_id') == 'seg_1':
                state['segment_id'] = new
                conn.execute('UPDATE execution_state SET state_json=? WHERE session_id=?', (json.dumps(state, ensure_ascii=False), sid))


def _backup_locked_source(path, destination):
    # Caller holds BEGIN IMMEDIATE on its writer. A separate reader provides a
    # stable committed pre-migration snapshot including any outstanding WAL.
    # Do not resolve the final component: O_EXCL must reject broken symlinks too.
    destination = Path(destination).absolute()
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    source = target = None
    try:
        source = _connect(path, readonly=True)
        target = _connect(destination)
        source.backup(target)
        target.close()
        target = None
        verify = _connect(destination, readonly=True)
        try:
            if [r[0] for r in verify.execute('PRAGMA integrity_check')] != ['ok']:
                raise RuntimeError('Backup integrity check failed')
        finally:
            verify.close()
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if source is not None:
            source.close()
        if target is not None:
            target.close()


def migrate_legacy_segments(db_path, *, apply=False, backup_path=None):
    """Return a read-only plan, or apply atomically after an exclusive new backup.

    Existing IDs/content, graph and lineage must have identical digests before
    and after. Preexisting orphans are reported; the migration may not add any.
    """
    path = Path(db_path).resolve()
    if apply and backup_path is None:
        raise ValueError('Applying migration requires an explicit nonexisting backup path')
    if apply and Path(backup_path).resolve() == path:
        raise ValueError('Backup destination must differ from database path')
    conn = _connect(path, readonly=not apply)
    try:
        conn.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
        before = _snapshot(conn)
        if before['integrity_check'] != ['ok']:
            raise RuntimeError('Source integrity check failed; migration refused')
        groups = _plan(conn)
        report = {'migration_id': MIGRATION_ID, 'applied': False, 'groups': groups,
                  'affected_events': sum(group['events'] for group in groups), 'before': before}
        if not apply or not groups:
            conn.rollback()
            return report
        _backup_locked_source(path, backup_path)
        # Build scope indexes before per-session updates; old installations may
        # otherwise scan their entire event history for each affected session.
        if 'timestamp' in {r[1] for r in conn.execute('PRAGMA table_info(events)')}:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_session_segment_time ON events(session_id,segment_id,timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id,timestamp)')
        if 'vec_items_meta' in _tables(conn):
            conn.execute('CREATE INDEX IF NOT EXISTS idx_vec_meta_event_model ON vec_items_meta(event_id,embedding_model_id)')
        for group in groups:
            _apply_group(conn, group)
        after = _snapshot(conn)
        if after != before:
            raise RuntimeError('Migration changed protected data or database integrity; rolled back')
        if _plan(conn):
            raise RuntimeError('Legacy segments remain after migration; rolled back')
        conn.execute('CREATE TABLE IF NOT EXISTS mneme_migrations (migration_id TEXT PRIMARY KEY, applied_at TEXT NOT NULL, affected_events INTEGER NOT NULL)')
        conn.execute('INSERT OR REPLACE INTO mneme_migrations VALUES(?,?,?)',
                     (MIGRATION_ID, datetime.now(timezone.utc).isoformat(), report['affected_events']))
        report.update(applied=True, backup_path=str(Path(backup_path).resolve()), after=after)
        if 'timestamp' in {r[1] for r in conn.execute('PRAGMA table_info(events)')} and 'embedding_index' in _tables(conn):
            report['centroid_query_plan'] = [r[3] for r in conn.execute(
                'EXPLAIN QUERY PLAN SELECT i.embedding FROM embedding_index i JOIN events e ON e.id=i.event_id '
                'WHERE e.session_id=? AND e.segment_id=? AND i.embedding_model_id=? ORDER BY e.timestamp DESC LIMIT 200',
                (groups[0]['session_id'], groups[0]['new_segment_id'], 'audit-model'))]
        conn.commit()
        return report
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
