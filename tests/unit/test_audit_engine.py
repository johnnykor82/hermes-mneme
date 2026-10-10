"""Offline regression coverage for audited lifecycle and graph paths."""
import copy
import pytest
from hermes_mneme.engine import CustomRouterContextEngine
from hermes_mneme.config import PluginConfig

@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = CustomRouterContextEngine(str(tmp_path / 'plugin.db'), PluginConfig({
        'active_window_tokens': 100000,
        'llm_enrichment_enabled': False, 'reranker_enabled': False,
        'cross_session_bootstrap_enabled': False, 'reindex_on_model_change': False,
        'pass_through_overhead_initial': 0,
    }))
    monkeypatch.setattr(e.indexer, 'add_embeddings_batch', lambda records: None)
    monkeypatch.setattr(e.indexer, 'get_embedding', lambda text: None)
    monkeypatch.setattr(e.indexer, 'get_embeddings_batch', lambda texts: [None] * len(texts))
    yield e
    e._embed_executor.shutdown(wait=True)
    e._enrichment_executor.shutdown(wait=True)


def ingest(e, messages):
    e._process_context_messages(messages, assemble=False, caller='audit')


def test_fresh_and_new_bind_segment_and_state(eng):
    for sid in ('A', 'B'):
        eng.on_session_start(sid)
        ingest(eng, [{'role': 'user', 'content': 'hello'}])
        expected = f'seg_{sid}_1'
        assert eng.current_segment_id == expected
        assert eng.segmenter.current_segment_id == expected
        assert eng.store.load_state(sid)['segment_id'] == expected


def test_hard_switch_persists_state_before_response_callback(eng):
    eng.on_session_start('A')
    messages = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'ok'}]
    ingest(eng, messages)
    ingest(eng, messages + [{'role': 'user', 'content': 'new topic'}])
    assert eng.current_segment_id == 'seg_A_2'
    assert eng.store.load_state('A')['segment_id'] == 'seg_A_2'


def test_reset_and_deepcopy_are_unbound_consistently(eng):
    eng.on_session_start('A')
    clone = copy.deepcopy(eng)
    try:
        assert clone.session_id is None
        assert clone.current_segment_id == clone.segmenter.current_segment_id == 'seg_unbound'
    finally:
        clone._embed_executor.shutdown(wait=True)
        clone._enrichment_executor.shutdown(wait=True)
    eng.on_session_reset()
    assert eng.current_segment_id == eng.segmenter.current_segment_id == 'seg_unbound'


def test_parallel_tool_outputs_match_call_ids(eng):
    eng.on_session_start('A')
    for event, eid in [
        ({'type': 'tool_call', 'tool_call_id': 'c1'}, 'call1'),
        ({'type': 'tool_call', 'tool_call_id': 'c2'}, 'call2'),
        ({'type': 'tool_output', 'tool_call_id': 'c2'}, 'out2'),
        ({'type': 'tool_output', 'tool_call_id': 'c1'}, 'out1'),
    ]:
        eng._record_graph_edges(event, eid)
    conn = eng.store._get_connection()
    edges = {tuple(r) for r in conn.execute('SELECT from_event_id,to_event_id FROM execution_graph')}
    conn.close()
    assert edges == {('call1', 'out1'), ('call2', 'out2')}


def test_rollover_replay_does_not_duplicate_logical_events(eng):
    eng.on_session_start('S0')
    messages = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'ok'}]
    ingest(eng, messages)
    for old, new in [('S0','S1'), ('S1','S2')]:
        eng.on_session_start(new, boundary_reason='compression', old_session_id=old)
        ingest(eng, messages)
        assert eng.store.count_events(new) == 2
        assert eng.store.load_state(new)['session_id'] == new


def test_native_budget_is_honored_and_overflow_explicit(eng):
    from hermes_mneme.prompt_builder import PromptBudgetExceeded
    eng.on_session_start('A')
    message = {'role': 'user', 'content': 'huge mandatory request ' * 500}
    previous = eng._budget_tokens
    with pytest.raises(PromptBudgetExceeded) as exc:
        eng.select_context([message], conversation_messages=[message], budget_tokens=50)
    assert exc.value.budget_tokens == 50
    assert eng._budget_tokens == previous
    assert eng.store.count_events('A') == 1


def test_rollover_shifted_retained_tail_does_not_duplicate(eng):
    eng.on_session_start('S0')
    messages = [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'answer first'},
                {'role': 'user', 'content': 'second'}, {'role': 'assistant', 'content': 'answer second'}]
    ingest(eng, messages)
    eng.on_session_start('S1', boundary_reason='compression', old_session_id='S0')
    ingest(eng, messages[2:])
    assert eng.store.count_events('S1') == 4


@pytest.mark.parametrize('answer', ['new answer', 'old answer'])
@pytest.mark.parametrize('binding', ['hook', 'auto'])
def test_resume_with_empty_history_preserves_identical_new_user(eng, answer, binding):
    request = {'role': 'user', 'content': 'Give me the remembered access code again.'}
    eng.on_session_start('A')
    eng.on_turn_complete([request, {'role': 'assistant', 'content': 'old answer'}])
    if binding == 'hook':
        eng.on_session_start('A')
    else:
        eng.session_id = None
        eng._processed_msg_count = 0
    eng.select_context([request], conversation_messages=[request], incoming_message=request)
    eng.on_turn_complete([request, {'role': 'assistant', 'content': answer}])
    assert eng.store.count_events('A') == 4
    assert eng.store.load_state('A')['turn_count'] == 2
    # Repeated completion of this same buffer must still be idempotent.
    eng.on_turn_complete([request, {'role': 'assistant', 'content': answer}])
    assert eng.store.count_events('A') == 4
    eng.on_session_start('B', boundary_reason='compression', old_session_id='A')
    eng.on_turn_complete([request, {'role': 'assistant', 'content': answer}])
    assert eng.store.count_events('B') == 4


def test_resume_retained_transcript_does_not_duplicate(eng):
    eng.on_session_start('A')
    messages = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'answer'}]
    eng.on_turn_complete(messages)
    eng.on_session_start('A')
    eng.on_turn_complete(messages)
    assert eng.store.count_events('A') == 2


def test_resume_legacy_pending_user_keeps_replay_identity(eng):
    eng.on_session_start('A')
    messages = [{'role': 'user', 'content': 'unfinished request'}]
    ingest(eng, messages)
    eng.on_session_start('A')
    ingest(eng, messages)
    assert eng.store.count_events('A') == 1
    assert eng.store.load_state('A')['turn_count'] == 1


def test_resume_short_retained_history_preserves_new_identical_turn(eng):
    request = {'role': 'user', 'content': 'repeat the access code'}
    answer = {'role': 'assistant', 'content': 'same answer'}
    eng.on_session_start('A')
    eng.on_turn_complete([request, answer, request, answer, request, answer])
    eng.on_session_start('A')
    retained = [request, answer]
    eng.select_context(retained + [request], conversation_messages=retained + [request], incoming_message=request)
    eng.on_turn_complete(retained + [request, answer])
    assert eng.store.count_events('A') == 8
    assert eng.store.load_state('A')['turn_count'] == 4


def test_resume_restores_namespace_for_legacy_completion_after_replay(eng):
    request = {'role': 'user', 'content': 'repeat the access code'}
    answer = {'role': 'assistant', 'content': 'same answer'}
    eng.on_session_start('A')
    eng.on_turn_complete([request, answer, request, answer, request, answer])
    eng.on_session_start('A')
    eng.select_context([request], conversation_messages=[request], incoming_message=request)
    eng.on_turn_complete([request, answer])
    nonce = eng._event_id_nonce
    assert nonce and eng.store.load_state('A')['_event_id_nonce'] == nonce
    eng.on_session_start('A')
    assert eng._event_id_nonce == nonce
    eng.on_turn_complete([request, answer])
    assert eng.store.count_events('A') == 8
    eng.on_turn_complete([request, answer, request, answer])
    assert eng.store.count_events('A') == 10
    assert eng.store.load_state('A')['turn_count'] == 5


def test_reset_waits_for_inflight_ingestion(eng, monkeypatch):
    import threading
    eng.on_session_start('A')
    entered, release, reset_done = threading.Event(), threading.Event(), threading.Event()
    original = eng.store.add_event
    errors = []
    def slow_add(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(eng.store, 'add_event', slow_add)
    def run_ingest():
        try:
            ingest(eng, [{'role': 'user', 'content': 'hello'}])
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=run_ingest)
    worker.start()
    assert entered.wait(2)
    resetter = threading.Thread(target=lambda: (eng.on_session_reset(), reset_done.set()))
    resetter.start()
    try:
        assert not reset_done.wait(.05)
    finally:
        release.set()
        worker.join(2)
        resetter.join(2)
    assert not errors
    assert eng.store.load_state('A')['session_id'] == 'A'
    assert eng.session_id is None


def test_reassign_updates_state_json_and_scopes_metadata(eng):
    s = eng.store
    a, _ = s.add_event('A', 'seg_A_1', 'user_message', 'user', 'first')
    unrelated, _ = s.add_event('OTHER', 'seg_OTHER_A_1', 'user_message', 'user', 'other')
    s.add_embedding(a, 'seg_A_1', b'1234', 'test', 1, 'user_message')
    s.add_embedding(unrelated, 'seg_OTHER_A_1', b'1234', 'test', 1, 'user_message')
    s.save_state('A', {'session_id': 'A', 'segment_id': 'seg_A_1'})
    s.reassign_session('A', 'B')
    assert s.load_state('B') == {'session_id': 'B', 'segment_id': 'seg_B_1'}
    conn = s._get_connection()
    assert conn.execute('SELECT segment_id FROM embedding_index WHERE event_id=?', (unrelated,)).fetchone()[0] == 'seg_OTHER_A_1'
    conn.close()


def test_reassign_does_not_merge_colliding_source_segment_names(eng):
    s = eng.store
    s.add_event('A', 'seg_A_1', 'user_message', 'user', 'topic1')
    s.add_event('A', 'seg_B_1', 'user_message', 'user', 'topic2')
    s.reassign_session('A', 'B')
    conn = s._get_connection()
    assert conn.execute("SELECT COUNT(DISTINCT segment_id) FROM events WHERE session_id='B'").fetchone()[0] == 2
    conn.close()


def test_failed_reassign_preserves_engine_binding(eng):
    eng.on_session_start('A')
    ingest(eng, [{'role':'user','content':'hello'}])
    conn = eng.store._get_connection()
    conn.execute("UPDATE execution_state SET state_json='invalid' WHERE session_id='A'")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError):
        eng.on_session_start('B', boundary_reason='compression', old_session_id='A')
    assert eng.session_id == 'A'
    assert eng.store.count_events('A') == 1
    assert eng.store.count_events('B') == 0


def test_compression_preserves_pending_tool_decision_edges(eng):
    eng.on_session_start('S0')
    messages = [{'role':'user','content':'run'},
                {'role':'assistant','content':'','tool_calls':[{'id':'c1','function':{'name':'run','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'c1','content':'done'}]
    ingest(eng, messages)
    eng.on_session_start('S1', boundary_reason='compression', old_session_id='S0')
    ingest(eng, messages + [{'role':'assistant','content':'finished'}])
    conn = eng.store._get_connection()
    assert conn.execute("SELECT COUNT(*) FROM execution_graph WHERE edge_type='decision'").fetchone()[0] == 1
    conn.close()
