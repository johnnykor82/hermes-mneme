"""Regressions for audit F05/F08/F10/F17, with no external API calls."""
import math
import sqlite3
import struct

import pytest

from hermes_mneme import config as C, index as I, store as S


def fixture_index(tmp_path, monkeypatch, **config):
    path = str(tmp_path / 'index.db')
    S.ContextStore(path)
    index = I.EmbeddingIndex(path, config=config) if config else I.EmbeddingIndex(path)
    monkeypatch.setattr(index, 'get_embedding', lambda _: [1.0] + [0.0] * (index.embedding_dim - 1))
    return index


def seed(index, event_id, session, similarity, segment='same'):
    vector = [similarity, math.sqrt(1 - similarity ** 2)] + [0.0] * (index.embedding_dim - 2)
    with index._get_connection() as conn:
        conn.execute('INSERT INTO events(id,session_id,segment_id,timestamp,type,role,content) VALUES(?,?,?,?,?,?,?)',
                     (event_id, session, segment, '2026-10-10', 'user_message', 'user', event_id))
    index.get_embeddings_batch = lambda texts: [vector for _ in texts]
    index.add_embeddings_batch([(event_id, segment, event_id, getattr(index, 'embedding_model', I.JINA_MODEL), 1, 'user_message')])


@pytest.mark.parametrize('top_k', [1, 5, None, 0])
def test_scoped_vec_recall_matches_exact(tmp_path, monkeypatch, top_k):
    index = fixture_index(tmp_path, monkeypatch)
    if not index._vec_available:
        pytest.skip('requires sqlite-vec')
    for n in range(10):
        seed(index, f'a{n}', 'A', .3 + n / 100)
    for n in range(60):
        seed(index, f'b{n}', 'B', .99)
    q = index.get_embedding('query')
    expected = index._search_python(q, 'A', None, top_k)
    actual = index._search_vec0(q, 'A', None, top_k)
    assert [r['event_id'] for r in actual] == [r['event_id'] for r in expected]
    assert [r['similarity'] for r in actual] == pytest.approx([r['similarity'] for r in expected])
    actual_multi = index._search_vec0_multi(q, ['A'], None, top_k)
    assert [r['event_id'] for r in actual_multi] == [r['event_id'] for r in expected]
    assert len(index.search_global('query', top_k)) == (70 if not top_k else top_k)


def test_reembedding_has_one_vec_mapping(tmp_path, monkeypatch):
    index = fixture_index(tmp_path, monkeypatch)
    if not index._vec_available:
        pytest.skip('requires sqlite-vec')
    seed(index, 'a', 'A', .5)
    for _ in range(2):
        index.add_embeddings_batch([('a', 'same', 'a', I.JINA_MODEL, 1, 'user_message')])
    with index._get_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM embedding_index').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM vec_items_meta').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM vec_items').fetchone()[0] == 1


def test_config_precedence_and_typed_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'PLUGIN_DIR', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('router_top_k: 7\n')
    monkeypatch.setenv('HERMES_CTX_ROUTER_TOP_K', '11')
    monkeypatch.setenv('HERMES_CTX_MEMORY_TOOL_NAMES', 'context_search,fetch_event')
    assert C.PluginConfig().router_top_k == 11
    assert C.PluginConfig({'router_top_k': 13}).router_top_k == 13
    assert C.PluginConfig().memory_tool_names == ['context_search', 'fetch_event']


def test_invalid_config_does_not_break_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'PLUGIN_DIR', str(tmp_path))
    monkeypatch.setenv('HERMES_CTX_ROUTER_TOP_K', 'invalid')
    monkeypatch.setenv('HERMES_CTX_SEGMENTATION_ENABLED', 'maybe')
    config = C.PluginConfig()
    assert config.router_top_k == C.DEFAULT_CONFIG['router_top_k']
    assert config.segmentation_enabled is True


@pytest.mark.parametrize('dim', [768, 1024, 1536])
def test_instance_embedding_endpoint_model_dimension(tmp_path, monkeypatch, dim):
    index = fixture_index(tmp_path, monkeypatch, embedding_endpoint='http://example.test/v1',
                          embedding_api_key='fixture-secret', embedding_model='fixture-model', embedding_dim=dim)
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'index': 0, 'embedding': [1.0] + [0.0] * (dim - 1)}]}
    monkeypatch.setattr(I.requests, 'post', lambda url, **kwargs: calls.append((url, kwargs)) or Response())
    assert len(index.get_embeddings_batch(['text'])[0]) == dim
    assert calls[0][0] == 'http://example.test/v1/embeddings'
    assert calls[0][1]['json']['model'] == 'fixture-model'
    assert calls[0][1]['headers']['Authorization'] == 'Bearer fixture-secret'
    seed(index, 'a', 'A', .5)
    assert index.search('text', 'A')[0]['event_id'] == 'a'
    index.get_embeddings_batch = lambda texts: [[1.0, 0.0]]
    assert index.add_embeddings_batch([('a', 'same', 'text', 'fixture-model', 1, 'user_message')]) == 0


def test_segment_multi_model_and_dimension_scope(tmp_path, monkeypatch):
    index = fixture_index(tmp_path, monkeypatch)
    seed(index, 'a1', 'A', .8, 'topic1')
    seed(index, 'a2', 'A', .5, 'topic2')
    seed(index, 'c1', 'C', .7, 'topic2')
    seed(index, 'b1', 'B', .99, 'topic2')
    with index._get_connection() as conn:
        conn.execute('INSERT INTO embedding_index VALUES(?,?,?,?,?,?)',
                     ('a1', 'topic1', struct.pack('2f', 1, 0), 1, 'user_message', 'other-model'))
        conn.execute('UPDATE embedding_index SET embedding=? WHERE event_id=? AND embedding_model_id=?',
                     (struct.pack('2f', 1, 0), 'a1', I.JINA_MODEL))
    assert [r['event_id'] for r in index.search_sessions('q', ['A', 'C'], None, 'topic2')] == ['c1', 'a2']
    assert [r['event_id'] for r in index.search('q', 'A', None, None)] == ['a2']
    assert index.search_sessions('q', [], None) == []


def test_http_batch_response_indices_and_invalid_dimensions(tmp_path, monkeypatch):
    index = fixture_index(tmp_path, monkeypatch)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'index': 1, 'embedding': [2.0] * 1024},
                             {'index': 0, 'embedding': [1.0] * 1024}]}
    monkeypatch.setattr(I.requests, 'post', lambda *args, **kwargs: Response())
    values = index.get_embeddings_batch(['first', '', 'second'])
    assert values[0][0] == 1
    assert values[1] is None
    assert values[2][0] == 2
    monkeypatch.setattr(Response, 'json', lambda _: {'data': [{'embedding': [1, 0]}]})
    assert index.get_embeddings_batch(['text']) == [None]


def test_config_yaml_errors_keep_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'PLUGIN_DIR', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('drift_threshold: -1\ndrift_weights: [0, 0, 0]\nembedding_dim: 0\nmemory_tool_names: [1]\n')
    config = C.PluginConfig()
    for key in ('drift_threshold', 'drift_weights', 'embedding_dim', 'memory_tool_names'):
        assert config.get(key) == C.DEFAULT_CONFIG[key]


def test_query_cache_reuses_success_but_retries_failure(tmp_path, monkeypatch):
    index = fixture_index(tmp_path, monkeypatch)
    # Restore real single-query method; fixture usually stubs it.
    monkeypatch.setattr(index, 'get_embedding', I.EmbeddingIndex.get_embedding.__get__(index))
    calls = []
    vector = [1.0] * 1024
    monkeypatch.setattr(index, 'get_embeddings_batch', lambda texts: calls.append(texts) or [vector])
    assert index.get_embedding('same') == vector
    # Caller mutation cannot corrupt the cached vector.
    index.get_embedding('same')[0] = 7.0
    assert index.get_embedding('same')[0] == 1.0
    assert len(calls) == 1
    monkeypatch.setattr(index, 'get_embeddings_batch', lambda texts: calls.append(texts) or [None])
    assert index.get_embedding('failure') is None
    assert index.get_embedding('failure') is None
    assert len(calls) == 3
    index.model_id = 'another-model'
    assert index.get_embedding('same') is None
    assert len(calls) == 4


def test_background_embedding_uses_segment_after_reassign(tmp_path, monkeypatch):
    index = fixture_index(tmp_path, monkeypatch)
    seed(index, 'a', 'old-session', .5, 'old-segment')
    def background_response(texts):
        # Compression/reassign occurs while the HTTP request is in flight.
        with index._get_connection() as conn:
            conn.execute('UPDATE events SET session_id=?, segment_id=? WHERE id=?',
                         ('new-session', 'new-segment', 'a'))
        return [[1.0] + [0.0] * 1023]
    monkeypatch.setattr(index, 'get_embeddings_batch', background_response)
    assert index.add_embeddings_batch([('a', 'old-segment', 'text', I.JINA_MODEL, 1, 'user_message')]) == 1
    with index._get_connection() as conn:
        assert conn.execute('SELECT segment_id FROM embedding_index WHERE event_id=?', ('a',)).fetchone()[0] == 'new-segment'
        if index._vec_available:
            assert conn.execute('SELECT segment_id FROM vec_items_meta WHERE event_id=?', ('a',)).fetchone()[0] == 'new-segment'


@pytest.mark.parametrize('dim', [768, 1024, 1536])
def test_fresh_vec_schema_uses_configured_dimension(tmp_path, monkeypatch, dim):
    path = str(tmp_path / 'fresh.db')
    S.ContextStore(path, embedding_dim=dim)
    index = I.EmbeddingIndex(path, config={'embedding_dim': dim})
    if not index._vec_available:
        pytest.skip('requires sqlite-vec')
    assert index._vec_dim == dim
    seed(index, 'a', 'A', .5)
    with index._get_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM vec_items_meta').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM vec_items').fetchone()[0] == 1
    monkeypatch.setattr(index, 'get_embedding', lambda text: [1.0] + [0.0] * (dim - 1))
    assert index.search('q', 'A')[0]['event_id'] == 'a'
