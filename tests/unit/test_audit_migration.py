import json
import sqlite3

import pytest
from hermes_mneme import migrations


@pytest.fixture
def legacy(tmp_path):
    path = tmp_path / 'legacy.db'
    conn = sqlite3.connect(path)
    conn.executescript('''CREATE TABLE events(id TEXT PRIMARY KEY,session_id TEXT,segment_id TEXT,role TEXT,content TEXT,timestamp TEXT DEFAULT '2026-10-10');
    CREATE TABLE embedding_index(event_id TEXT,embedding_model_id TEXT,segment_id TEXT,embedding BLOB,PRIMARY KEY(event_id,embedding_model_id));
    CREATE TABLE vec_items_meta(rowid INTEGER PRIMARY KEY,event_id TEXT,embedding_model_id TEXT,segment_id TEXT);
    CREATE TABLE execution_state(session_id TEXT PRIMARY KEY,state_json TEXT);
    CREATE TABLE execution_graph(from_event_id TEXT,to_event_id TEXT,session_id TEXT);
    CREATE TABLE session_lineage(old_session_id TEXT,new_session_id TEXT);''')
    for n in range(441):
        sid = f'S{n}'
        for i in range(2):
            eid = f'{sid}-{i}'
            conn.execute('INSERT INTO events(id,session_id,segment_id,role,content) VALUES(?,?,?,?,?)', (eid, sid, 'seg_1', 'user', f'content {n} {i}'))
            conn.execute('INSERT INTO embedding_index VALUES(?,?,?,?)', (eid, 'model', 'seg_1', b'vector'))
            conn.execute('INSERT INTO vec_items_meta(event_id,embedding_model_id,segment_id) VALUES(?,?,?)', (eid, 'model', 'seg_1'))
        conn.execute('INSERT INTO execution_state VALUES(?,?)', (sid, json.dumps({'segment_id': 'seg_1', 'session_id': sid})))
        conn.execute('INSERT INTO execution_graph VALUES(?,?,?)', (f'{sid}-0', f'{sid}-1', sid))
    conn.execute("INSERT INTO events(id,session_id,segment_id,role,content) VALUES('collision','S0','seg_S0_1','user','other')")
    conn.commit()
    conn.close()
    return path


def test_dry_run_read_only_and_441_session_scope(legacy, tmp_path):
    before = legacy.read_bytes()
    report = migrations.migrate_legacy_segments(legacy)
    assert legacy.read_bytes() == before
    assert report['affected_events'] == 882
    assert len(report['groups']) == 441
    assert report['groups'][0]['new_segment_id'] == 'seg_S0_legacy1'
    backup = tmp_path / 'backup.db'
    done = migrations.migrate_legacy_segments(legacy, apply=True, backup_path=backup)
    assert done['applied']
    assert done['before']['event_digest'] == done['after']['event_digest']
    assert done['before']['graph_digest'] == done['after']['graph_digest']
    for key in ('embedding_digest', 'vec_meta_digest', 'state_digest'):
        assert done['before'][key] == done['after'][key]
    assert any('idx_events_session_segment_time' in row for row in done['centroid_query_plan'])
    assert done['after']['integrity_check'] == ['ok']
    conn = sqlite3.connect(legacy)
    assert conn.execute("SELECT COUNT(*) FROM events WHERE segment_id='seg_1'").fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(DISTINCT segment_id) FROM events').fetchone()[0] == 442
    assert conn.execute("SELECT segment_id FROM embedding_index WHERE event_id='S1-0'").fetchone()[0] == 'seg_S1_1'
    assert json.loads(conn.execute("SELECT state_json FROM execution_state WHERE session_id='S0'").fetchone()[0])['segment_id'] == 'seg_S0_legacy1'
    conn.close()
    again = migrations.migrate_legacy_segments(legacy)
    assert again['affected_events'] == 0
    repeated = migrations.migrate_legacy_segments(legacy, apply=True, backup_path=tmp_path / 'again.db')
    assert not repeated['applied']


def test_apply_requires_nonexisting_backup(legacy, tmp_path):
    with pytest.raises(ValueError, match='backup'):
        migrations.migrate_legacy_segments(legacy, apply=True)
    destination = tmp_path / 'existing.db'
    destination.write_text('preserve')
    with pytest.raises(FileExistsError):
        migrations.migrate_legacy_segments(legacy, apply=True, backup_path=destination)
    assert destination.read_text() == 'preserve'


def test_atomic_rollback_with_valid_backup(legacy, tmp_path, monkeypatch):
    original = migrations._apply_group
    def failure(conn, group):
        original(conn, group)
        raise RuntimeError('injected crash')
    monkeypatch.setattr(migrations, '_apply_group', failure)
    before = migrations.migrate_legacy_segments(legacy)['before']
    backup = tmp_path / 'rollback.db'
    with pytest.raises(RuntimeError, match='injected'):
        migrations.migrate_legacy_segments(legacy, apply=True, backup_path=backup)
    assert migrations.migrate_legacy_segments(legacy)['before'] == before
    assert migrations.migrate_legacy_segments(backup)['before'] == before


def test_backup_includes_uncheckpointed_wal(legacy, tmp_path):
    writer = sqlite3.connect(legacy)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute("INSERT INTO events(id,session_id,segment_id,role,content) VALUES('wal-event','WAL','seg_1','user','in wal')")
    writer.commit()
    backup = tmp_path / 'wal-backup.db'
    migrations.migrate_legacy_segments(legacy, apply=True, backup_path=backup)
    reader = sqlite3.connect(backup)
    assert reader.execute("SELECT content FROM events WHERE id='wal-event'").fetchone()[0] == 'in wal'
    reader.close()
    writer.close()


def test_vec0_extension_loaded_for_source_backup_and_reopen(legacy, tmp_path):
    sqlite_vec = pytest.importorskip('sqlite_vec')
    import struct
    conn = sqlite3.connect(legacy)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute('CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[2])')
    for (rowid,) in conn.execute('SELECT rowid FROM vec_items_meta').fetchall():
        conn.execute('INSERT INTO vec_items(rowid,embedding) VALUES(?,?)', (rowid, struct.pack('2f', 1., 0.)))
    conn.commit()
    conn.close()
    report = migrations.migrate_legacy_segments(legacy, apply=True, backup_path=tmp_path / 'vec-backup.db')
    assert report['before']['counts']['vec_items'] == 882
    assert report['after']['orphans']['vec_items_rowid'] == 0
    assert migrations.migrate_legacy_segments(tmp_path / 'vec-backup.db')['affected_events'] == 882


def test_protected_embedding_mutation_rolls_back(legacy, tmp_path, monkeypatch):
    original = migrations._apply_group
    def corrupt(conn, group):
        original(conn, group)
        conn.execute("UPDATE embedding_index SET embedding=x'0001' WHERE event_id='S0-0'")
    monkeypatch.setattr(migrations, '_apply_group', corrupt)
    before = migrations.migrate_legacy_segments(legacy)['before']
    with pytest.raises(RuntimeError, match='protected data'):
        migrations.migrate_legacy_segments(legacy, apply=True, backup_path=tmp_path / 'protected.db')
    assert migrations.migrate_legacy_segments(legacy)['before'] == before


def test_broken_symlink_backup_is_rejected(legacy, tmp_path):
    destination = tmp_path / 'broken-backup.db'
    target = tmp_path / 'must-not-create.db'
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        migrations.migrate_legacy_segments(legacy, apply=True, backup_path=destination)
    assert destination.is_symlink()
    assert not target.exists()


def test_scope_index_exists_before_first_group(legacy, tmp_path, monkeypatch):
    original = migrations._apply_group
    visited = []
    def verify_index(conn, group):
        names = {row[1] for row in conn.execute("PRAGMA index_list('events')")}
        assert 'idx_events_session_segment_time' in names
        visited.append(group['session_id'])
        original(conn, group)
    monkeypatch.setattr(migrations, '_apply_group', verify_index)
    migrations.migrate_legacy_segments(legacy, apply=True, backup_path=tmp_path / 'indexed-backup.db')
    assert len(visited) == 441


def test_plain_database_does_not_require_sqlite_extension_api(legacy, monkeypatch):
    original_connect = sqlite3.connect
    class NoExtensions:
        def __init__(self, connection):
            self.connection = connection
        def __getattr__(self, name):
            if name in ('enable_load_extension', 'load_extension'):
                raise AttributeError(name)
            return getattr(self.connection, name)
    monkeypatch.setattr(migrations.sqlite3, 'connect', lambda *args, **kwargs: NoExtensions(original_connect(*args, **kwargs)))
    assert migrations.migrate_legacy_segments(legacy)['affected_events'] == 882
