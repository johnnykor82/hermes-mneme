"""Hybrid retrieval contracts on temporary SQLite, without network calls."""
import sqlite3
import struct

import pytest

from hermes_mneme import index as I, router as R, store as S


def fixture(tmp_path, monkeypatch):
    path = str(tmp_path / 'memory.db')
    store = S.ContextStore(path, embedding_dim=2)
    idx = I.EmbeddingIndex(path, {'embedding_dim': 2, 'embedding_model': 'active'})
    monkeypatch.setattr(idx, 'get_embedding', lambda text: [1., 0.])
    return store, idx


def event(store, event_id, content, session='A', segment='topic', vector=None, model='active'):
    with store._get_connection() as conn:
        conn.execute('INSERT INTO events(id,session_id,segment_id,timestamp,type,role,content) VALUES(?,?,?,?,?,?,?)',
                     (event_id, session, segment, '2026-10-11', 'user_message', 'user', content))
        if vector is not None:
            conn.execute('INSERT INTO embedding_index(event_id,segment_id,embedding,token_count,type,embedding_model_id) VALUES(?,?,?,?,?,?)',
                         (event_id, segment, struct.pack('2f', *vector), 1, 'user_message', model))


def test_lexical_recovers_unembedded_fact_and_excludes_wrong_scope(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'fact', 'Код клиента СОСНА-417', vector=None)
    for n in range(70):
        event(store, f'other{n}', 'Код клиента СОСНА-417', session='B', vector=[1, 0])
    event(store, 'topic2', 'Код клиента СОСНА-417', segment='other', vector=[1, 0])
    rows = idx.search_hybrid('СОСНА-417', ['A'], 'topic', top_k=5)
    assert [r['event_id'] for r in rows] == ['fact']
    assert rows[0]['similarity'] is None
    assert rows[0]['content'] == 'Код клиента СОСНА-417'


def test_semantic_excludes_current_question_before_candidate_limit(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    for n in range(65):
        event(store, f'echo{n}', 'текущий вопрос', vector=[1, 0])
    event(store, 'answer', 'ответ', vector=[.8, .6])
    rows = idx.search_hybrid('вопрос', ['A'], top_k=5,
                             exclude_event_ids=[f'echo{n}' for n in range(65)])
    assert [r['event_id'] for r in rows] == ['answer']


def test_fts_tracks_insert_update_delete_and_reassignment(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'before', 'Архив КЕДР-719')
    assert idx.search_hybrid('КЕДР-719', ['A'])[0]['event_id'] == 'before'
    event(store, 'after', 'Архив ТИС-941')
    with store._get_connection() as conn:
        conn.execute("UPDATE events SET content='Архив ДУБ-312', session_id='B' WHERE id='after'")
        conn.execute("DELETE FROM events WHERE id='before'")
    assert idx.search_hybrid('КЕДР-719', ['A']) == []
    assert idx.search_hybrid('ДУБ-312', ['A']) == []
    assert idx.search_hybrid('ДУБ-312', ['B'])[0]['event_id'] == 'after'
    assert idx.search_hybrid('ТИС-941', ['B']) == []
    again = I.EmbeddingIndex(idx.db_path, {'embedding_dim': 2, 'embedding_model': 'active'})
    monkeypatch.setattr(again, 'get_embedding', lambda text: None)
    assert again.search_hybrid('ДУБ-312', ['B'])[0]['event_id'] == 'after'
    with store._get_connection() as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_no_current_model_vectors_and_query_syntax_safe(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'wrong', 'irrelevant', vector=[1, 0], model='other')
    event(store, 'fact', 'КЕДР 719')
    rows = idx.search_hybrid('"КЕДР" OR (719*)', ['A'])
    assert [r['event_id'] for r in rows] == ['fact']
    assert idx.search_hybrid('что ли', ['A']) == []
    assert idx.search_hybrid('КЕДР', []) == []


@pytest.mark.parametrize('k', [None, 0])
def test_explicit_unlimited_preserves_all_events(tmp_path, monkeypatch, k):
    store, idx = fixture(tmp_path, monkeypatch)
    for n in range(65):
        event(store, str(n), 'архив', vector=[1, 0])
    assert len(idx.search_hybrid('архив', ['A'], top_k=k)) == 65


def test_bounded_pool_is_distinct_from_public_k(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    for n in range(100):
        event(store, str(n), 'архив', vector=[1, 0])
    assert len(idx.search_hybrid('архив', ['A'], top_k=5)) == 40


def test_rerank_search_preserves_provenance_and_bounds_unlimited(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    router = R.ContextRouter(store, idx, {'reranker_enabled': True, 'reranker_endpoint': 'http://example.test/rerank'})
    rows = [{'event_id': str(n), 'content': 'x'*2500, 'similarity': .8, 'retrieval_score': .01} for n in range(65)]
    sent = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'results': [{'index': n, 'relevance_score': n} for n in range(40)]}
    monkeypatch.setattr(R.requests, 'post', lambda *args, **kwargs: sent.append(kwargs) or Response())
    out = router.rerank_search('query', rows)
    assert len(sent[0]['json']['documents']) == 40
    assert len(sent[0]['json']['documents'][0]) == 2000
    assert [r['event_id'] for r in out[:2]] == ['39', '38']
    assert len(out) == 65
    assert out[0]['similarity'] == .8
    assert out[0]['rerank_score'] == 39
    assert 'rerank_score' not in rows[0]


@pytest.mark.parametrize('results', [[{'index': 0, 'relevance_score': 9}],
                                   [{'index': 0, 'relevance_score': float('nan')}, {'index': 1, 'relevance_score': 8}],
                                   [{'index': 0, 'relevance_score': 9}, {'index': 0, 'relevance_score': 8}]])
def test_invalid_partial_rerank_does_not_mix_score_scales(tmp_path, monkeypatch, results):
    store, idx = fixture(tmp_path, monkeypatch)
    router = R.ContextRouter(store, idx, {'reranker_enabled': True, 'reranker_endpoint': 'http://example.test/rerank'})
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'results': results}
    monkeypatch.setattr(R.requests, 'post', lambda *args, **kwargs: Response())
    rows = [{'event_id': 'a', 'content': 'a', 'similarity': .8}, {'event_id': 'b', 'content': 'b', 'similarity': .7}]
    assert router.rerank_search('query', rows) == rows


def test_unlimited_metadata_fetch_respects_sqlite_variable_limit(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    for n in range(501):
        event(store, str(n), 'архив', vector=[1, 0])
    connection = idx._get_connection
    if not hasattr(sqlite3.Connection, 'setlimit'):
        pytest.skip('SQLite runtime limits require Python 3.11')
    def limited():
        conn = connection()
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 500)
        return conn
    monkeypatch.setattr(idx, '_get_connection', limited)
    assert len(idx.search_hybrid('архив', ['A'], top_k=0)) == 501


def test_readonly_database_preserves_semantic_fallback(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'fact', 'архив', vector=[1, 0])
    idx._vec_available = False
    monkeypatch.setattr(idx, '_get_connection', lambda: sqlite3.connect('file:'+idx.db_path+'?mode=ro', uri=True))
    rows = idx.search_hybrid('архив', ['A'])
    assert [r['event_id'] for r in rows] == ['fact']
    assert not idx._lexical_ready
    with store._get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='events_fts'").fetchone()[0] == 0


def test_absent_fts_extension_preserves_semantic_fallback(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'fact', 'архив', vector=[1, 0])
    idx._vec_available = False
    class WithoutFTS(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith('CREATE VIRTUAL TABLE events_fts'):
                raise sqlite3.OperationalError('no such module: fts5')
            return super().execute(sql, *args, **kwargs)
    monkeypatch.setattr(idx, '_get_connection', lambda: sqlite3.connect(idx.db_path, factory=WithoutFTS))
    assert [r['event_id'] for r in idx.search_hybrid('архив', ['A'])] == ['fact']
    with store._get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'events_fts%'").fetchone()[0] == 0


def test_embedding_outage_keeps_lexical_fact(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'fact', 'номер СОСНА-417')
    monkeypatch.setattr(idx, 'get_embedding', lambda _: None)
    assert [r['event_id'] for r in idx.search_hybrid('СОСНА-417', ['A'])] == ['fact']


def test_failed_fts_transaction_leaves_no_partial_schema(tmp_path, monkeypatch):
    store, idx = fixture(tmp_path, monkeypatch)
    event(store, 'fact', 'архив', vector=[1, 0])
    idx._vec_available = False
    class FailedTrigger(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith('CREATE TRIGGER events_fts_ad'):
                raise sqlite3.OperationalError('simulated transaction interruption')
            return super().execute(sql, *args, **kwargs)
    monkeypatch.setattr(idx, '_get_connection', lambda: sqlite3.connect(idx.db_path, factory=FailedTrigger))
    assert [r['event_id'] for r in idx.search_hybrid('архив', ['A'])] == ['fact']
    with store._get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'events_fts%'").fetchone()[0] == 0
