import sqlite3
import struct
from types import SimpleNamespace

import pytest
from hermes_mneme.segmenter import SessionSegmenter


@pytest.fixture
def segmenter(tmp_path):
    path = tmp_path / 'segments.db'
    conn = sqlite3.connect(path)
    conn.executescript('''CREATE TABLE events(id TEXT PRIMARY KEY, session_id TEXT, segment_id TEXT, timestamp INTEGER);
    CREATE TABLE embedding_index(event_id TEXT, embedding_model_id TEXT, embedding BLOB);''')
    conn.close()
    store = SimpleNamespace(_get_connection=lambda: sqlite3.connect(path, check_same_thread=False))
    indexer = SimpleNamespace(model_id='model', embedding_dim=2, get_embedding=lambda text: [0., 1.])
    return SessionSegmenter(store, indexer, 'seg_A_1', config={'embedding_model': 'model'})


def add(segmenter, sid, segment, count=3, vector=(1., 0.), model='model'):
    conn = segmenter.store._get_connection()
    for i in range(count):
        event_id = f'{sid}-{segment}-{model}-{i}'
        conn.execute('INSERT OR REPLACE INTO events VALUES(?,?,?,?)', (event_id, sid, segment, i))
        conn.execute('INSERT INTO embedding_index VALUES(?,?,?)', (event_id, model, struct.pack(f'{len(vector)}f', *vector)))
    conn.commit()
    conn.close()


def test_binding_and_counter_resume(segmenter):
    add(segmenter, 'A', 'seg_A_8')
    segmenter.bind_session('A', 'seg_A_8')
    assert segmenter.create_new_segment('A') == 'seg_A_9'
    segmenter.bind_session('B', 'seg_B_1')
    assert segmenter.create_new_segment('B') == 'seg_B_2'


def bind_scope(segmenter, sid, segment):
    # Also exercises the audited implementation without requiring the new API:
    # supplying its missing logical scope must still expose its faulty SQL/cache.
    if hasattr(segmenter, 'bind_session'):
        segmenter.bind_session(sid, segment)
    else:
        segmenter.session_id = sid
        segmenter.current_segment_id = segment


def test_centroid_session_model_and_dimension_scope(segmenter):
    add(segmenter, 'A', 'seg_1')
    add(segmenter, 'B', 'seg_1', count=210, vector=(0., 1.))
    add(segmenter, 'A', 'seg_1', vector=(0., 1., 1.), model='other')
    bind_scope(segmenter, 'A', 'seg_1')
    assert segmenter.get_centroid('seg_1') == [1., 0.]


def test_embedding_arrival_and_replacement_invalidates_before_blob_load(segmenter, monkeypatch):
    add(segmenter, 'A', 'seg_A_1')
    bind_scope(segmenter, 'A', 'seg_A_1')
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    original = segmenter._load_segment_embeddings
    calls = []
    monkeypatch.setattr(segmenter, '_load_segment_embeddings', lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    assert calls == []
    conn = segmenter.store._get_connection()
    conn.execute('UPDATE embedding_index SET embedding=?', (struct.pack('2f', 0., 1.),))
    conn.commit()
    conn.close()
    assert segmenter.get_centroid('seg_A_1') == [0., 1.]
    assert calls == [1]


def test_disabled_hard_and_soft(segmenter, monkeypatch):
    monkeypatch.setattr(segmenter, 'create_new_segment', lambda sid: 'unexpected')
    monkeypatch.setattr(segmenter, 'calculate_drift_score', lambda *a: 1.)
    segmenter.config['segmentation_enabled'] = False
    assert segmenter.handle_new_message('A', 'new topic', {}) == 'seg_A_1'


def test_embedding_only_drift_and_config_threshold(segmenter):
    add(segmenter, 'A', 'seg_A_1')
    segmenter.config['drift_threshold'] = .6
    assert segmenter.handle_new_message('A', 'please explain this unrelated other subject', {}) == 'seg_A_1'
    segmenter.config['drift_threshold'] = .35
    assert segmenter.handle_new_message('A', 'please explain this unrelated other subject', {}) == 'seg_A_2'


def test_russian_explicit_switch_and_short_continuation(segmenter):
    add(segmenter, 'A', 'seg_A_1')
    segmenter.indexer.get_embedding = lambda text: [1., .1]
    assert segmenter.handle_new_message('A', 'уточни пожалуйста', {}) == 'seg_A_1'
    assert segmenter.handle_new_message('A', 'новая тема', {}) == 'seg_A_2'


def test_configured_composite_weights(segmenter):
    segmenter.config['drift_weights'] = [0., 1., 0.]
    assert segmenter.calculate_drift_score('text', 'seg_A_1', entity_signal=1.) == 1.


def test_two_engines_reserve_unique_ids(segmenter):
    other = SessionSegmenter(segmenter.store, segmenter.indexer, 'seg_A_1', segmenter.config)
    segmenter.bind_session('A', 'seg_A_1')
    other.bind_session('A', 'seg_A_1')
    assert segmenter.create_new_segment('A') == 'seg_A_2'
    assert other.create_new_segment('A') == 'seg_A_3'


def test_background_embedding_arrival(segmenter):
    add(segmenter, 'A', 'seg_A_1')
    bind_scope(segmenter, 'A', 'seg_A_1')
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    conn = segmenter.store._get_connection()
    conn.execute("INSERT INTO events VALUES('later','A','seg_A_1',10)")
    conn.commit()
    conn.close()
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    conn = segmenter.store._get_connection()
    conn.execute("INSERT INTO embedding_index VALUES('later','model',?)", (struct.pack('2f', 0., 1.),))
    conn.commit()
    conn.close()
    assert segmenter.get_centroid('seg_A_1') == [.75, .25]


def test_unbound_and_close(segmenter):
    segmenter.bind_session(None, 'seg_unbound')
    assert segmenter.segment_count == 0
    assert segmenter.get_centroid('seg_unbound') is None
    segmenter.bind_session('A', 'seg_A_1')
    add(segmenter, 'A', 'seg_A_1')
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    segmenter.close()
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    segmenter.close()


def test_centroid_excludes_other_model_and_wrong_dimension(segmenter):
    add(segmenter, 'A', 'seg_A_1')
    add(segmenter, 'A', 'seg_A_1', vector=(0., 1., 1.), model='other')
    conn = segmenter.store._get_connection()
    conn.execute("INSERT INTO events VALUES('bad-dimension','A','seg_A_1',100)")
    conn.execute("INSERT INTO embedding_index VALUES('bad-dimension','model',?)", (struct.pack('3f', 0., 1., 1.),))
    conn.commit()
    conn.close()
    bind_scope(segmenter, 'A', 'seg_A_1')
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]


def test_same_event_count_replacement_refreshes_centroid(segmenter):
    add(segmenter, 'A', 'seg_A_1')
    bind_scope(segmenter, 'A', 'seg_A_1')
    assert segmenter.get_centroid('seg_A_1') == [1., 0.]
    conn = segmenter.store._get_connection()
    conn.execute('UPDATE embedding_index SET embedding=?', (struct.pack('2f', 0., 1.),))
    conn.commit()
    conn.close()
    assert segmenter.get_centroid('seg_A_1') == [0., 1.]


def test_configured_centroid_cache_capacity_evicts(segmenter, monkeypatch):
    add(segmenter, 'A', 'seg_A_1')
    add(segmenter, 'A', 'seg_A_2')
    bind_scope(segmenter, 'A', 'seg_A_1')
    segmenter.config['centroid_cache_size'] = 1
    original = segmenter._load_segment_embeddings
    calls = []
    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(segmenter, '_load_segment_embeddings', load)
    for segment in ['seg_A_1', 'seg_A_2', 'seg_A_1']:
        assert segmenter.get_centroid(segment) == [1., 0.]
    assert calls == ['seg_A_1', 'seg_A_2', 'seg_A_1']
