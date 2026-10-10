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


def _tool_history():
    return [
        {'role': 'user', 'content': 'inspect the project'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call-old', 'type': 'function', 'function': {'name': 'inspect', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'project details', 'tool_call_id': 'call-old'},
        {'role': 'assistant', 'content': 'inspection complete'},
    ]


def test_resume_shifted_legacy_null_tool_ids_preserves_new_user(eng):
    eng.on_session_start('A')
    earlier = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'earlier answer'}]
    retained = _tool_history()
    eng.on_turn_complete(earlier + retained)
    conn = eng.store._get_connection()
    conn.execute("UPDATE events SET tool_call_id=NULL WHERE type IN ('tool_call','tool_output')")
    conn.commit()
    conn.close()
    eng.on_session_start('A')
    incoming = retained[0].copy()
    eng.select_context(retained + [incoming], conversation_messages=retained + [incoming], incoming_message=incoming)
    eng.on_turn_complete(retained + [incoming, {'role': 'assistant', 'content': 'new answer'}])
    assert eng.store.count_events('A') == 8
    assert eng.store.load_state('A')['turn_count'] == 3


def test_replay_prefix_keeps_distinct_non_null_tool_ids(eng):
    eng.on_session_start('A')
    history = _tool_history()
    eng.on_turn_complete(history)
    different = copy.deepcopy(history)
    different[1]['tool_calls'][0]['id'] = 'call-different'
    different[2]['tool_call_id'] = 'call-different'
    matched = eng._match_replay_prefix(different, 0)
    assert (1, 0) not in matched
    assert (2, 0) not in matched


def test_resume_legacy_mixed_text_and_duplicate_rows_align(eng):
    eng.on_session_start('A')
    retained = _tool_history()
    eng.on_turn_complete(retained)
    conn = eng.store._get_connection()
    conn.execute("UPDATE events SET tool_call_id=NULL WHERE type IN ('tool_call','tool_output')")
    call = conn.execute("SELECT * FROM events WHERE type='tool_call'").fetchone()
    conn.execute("INSERT INTO events(id,session_id,segment_id,timestamp,type,role,content) VALUES(?,?,?,?,?,?,?)",
                 ('legacy-duplicate', 'A', call['segment_id'], call['timestamp'], 'tool_call', 'assistant', call['content']))
    conn.commit()
    conn.close()
    before = eng.store.count_events('A')
    replay = copy.deepcopy(retained)
    replay[1]['content'] = 'I will inspect the project now.'
    eng.on_session_start('A')
    incoming = replay[0].copy()
    eng.select_context(replay + [incoming], conversation_messages=replay + [incoming], incoming_message=incoming)
    full = replay + [incoming, {'role': 'assistant', 'content': 'new answer'}]
    eng.on_turn_complete(full)
    assert eng.store.count_events('A') == before + 3  # missing legacy text + new user/answer
    eng.select_context(full + [incoming], conversation_messages=full + [incoming], incoming_message=incoming)
    eng.on_turn_complete(full + [incoming, {'role': 'assistant', 'content': 'another new answer'}])
    assert eng.store.count_events('A') == before + 5


def test_resume_unknown_pruned_call_cannot_use_legacy_deterministic_output(eng):
    eng.on_session_start('A')
    old = _tool_history()
    old[2]['content'] = '[Old tool output cleared to save context space]'
    eng.on_turn_complete(old)
    conn = eng.store._get_connection()
    conn.execute("UPDATE events SET tool_call_id=NULL WHERE type IN ('tool_call','tool_output')")
    # Install the pre-provider-ID deterministic output identity.
    output = conn.execute("SELECT id FROM events WHERE type='tool_output'").fetchone()[0]
    legacy_id = eng.store.make_event_id('A', 2, 'tool', old[2]['content'], 0)
    conn.execute("UPDATE execution_graph SET to_event_id=? WHERE to_event_id=?", (legacy_id, output))
    conn.execute("UPDATE events SET id=? WHERE id=?", (legacy_id, output))
    conn.commit()
    conn.close()
    new = copy.deepcopy(old)
    new[1]['tool_calls'][0]['id'] = 'unrelated-provider'
    new[1]['tool_calls'][0]['function']['arguments'] = '{"unseen": true}'
    new[2]['tool_call_id'] = 'unrelated-provider'
    eng.on_session_start('A')
    eng.on_turn_complete(new)
    assert eng.store.count_events('A') == 6
    conn = eng.store._get_connection()
    assert conn.execute("SELECT COUNT(*) FROM execution_graph WHERE edge_type='tool_output'").fetchone()[0] == 2
    conn.close()


@pytest.mark.parametrize('legacy_ids', [False, True])
@pytest.mark.parametrize('missing_graph', [False, True])
def test_resume_pruned_outputs_reuses_correlated_originals(eng, legacy_ids, missing_graph):
    eng.on_session_start('A')
    first = _tool_history()
    second = copy.deepcopy(first)
    second[1]['tool_calls'][0]['id'] = 'call-other'
    second[1]['tool_calls'][0]['function']['arguments'] = '{"other": true}'
    second[2]['tool_call_id'] = 'call-other'
    second[2]['content'] = 'other original output'
    eng.on_turn_complete(first + second)
    conn = eng.store._get_connection()
    original = list(conn.execute("SELECT id,content FROM events WHERE type='tool_output' ORDER BY timestamp,id"))
    if legacy_ids:
        conn.execute("UPDATE events SET tool_call_id=NULL WHERE type IN ('tool_call','tool_output')")
        conn.commit()
    conn.close()
    if missing_graph:
        conn = eng.store._get_connection()
        conn.execute("DELETE FROM execution_graph WHERE edge_type='tool_output'")
        conn.commit()
        conn.close()
    replay = copy.deepcopy(first + second)
    for message in replay:
        if message['role'] == 'tool':
            message['content'] = '[Old tool output cleared to save context space]'
    eng.on_session_start('A')
    eng.on_turn_complete(replay)
    assert eng.store.count_events('A') == 8
    conn = eng.store._get_connection()
    assert list(conn.execute("SELECT id,content FROM events WHERE type='tool_output' ORDER BY timestamp,id")) == original
    assert conn.execute("SELECT COUNT(*) FROM execution_graph WHERE edge_type='tool_output'").fetchone()[0] == (0 if missing_graph else 2)
    conn.close()
    # A previous anonymous placeholder must not be selected by text alone.
    eng.store.add_event('A', eng.current_segment_id, 'tool_output', 'tool',
                        '[Old tool output cleared to save context space]')
    # An unknown provider call cannot inherit either historical result.
    unknown = copy.deepcopy(second)
    unknown[1]['tool_calls'][0]['id'] = 'call-new'
    unknown[1]['tool_calls'][0]['function']['arguments'] = '{"unseen": true}'
    unknown[2]['tool_call_id'] = 'call-new'
    unknown[2]['content'] = '[Old tool output cleared to save context space]'
    eng.on_session_start('A')
    assert (2, 0) not in eng._match_replay_prefix(unknown, 0)


def test_repeated_resume_matches_backfilled_text_and_reordered_history(eng):
    eng.on_session_start('A')
    old = _tool_history()
    eng.on_turn_complete(old)
    conn = eng.store._get_connection()
    conn.execute("UPDATE events SET tool_call_id=NULL WHERE type IN ('tool_call','tool_output')")
    conn.commit()
    conn.close()
    canonical = copy.deepcopy(old)
    canonical[1]['content'] = 'Previously omitted assistant explanation'
    eng.on_session_start('A')
    eng.on_turn_complete(canonical)
    assert eng.store.count_events('A') == 5
    for replay in (canonical, [canonical[3], *canonical[:3]], canonical):
        eng.on_session_start('A')
        eng.on_turn_complete(replay)
        assert eng.store.count_events('A') == 5
    incoming = old[0].copy()
    eng.on_session_start('A')
    eng.select_context(canonical + [incoming], conversation_messages=canonical + [incoming], incoming_message=incoming)
    eng.on_turn_complete(canonical + [incoming, {'role': 'assistant', 'content': 'new answer'}])
    assert eng.store.count_events('A') == 7


def test_compression_identical_incoming_user_survives_api_retries(eng):
    incoming = {'role': 'user', 'content': 'repeat request'}
    eng.on_session_start('A')
    eng.on_turn_complete([incoming, {'role': 'assistant', 'content': 'old answer'},
                          {'role': 'user', 'content': 'different request'},
                          {'role': 'assistant', 'content': 'another answer'}])
    eng.on_session_start('B', boundary_reason='compression', old_session_id='A')
    eng.select_context([incoming], conversation_messages=[incoming], incoming_message=incoming)
    assert eng.store.count_events('B') == 5
    assert eng.store.load_state('B')['turn_count'] == 3
    nonce = eng._event_id_nonce
    # Same request cache and a changed system-prefix cache miss both retry the same turn.
    eng.select_context([incoming], conversation_messages=[incoming], incoming_message=incoming)
    eng.select_context([{'role': 'system', 'content': 'retry prefix'}, incoming],
                       conversation_messages=[incoming], incoming_message=incoming)
    assert eng.store.count_events('B') == 5
    assert eng._event_id_nonce == nonce
    eng.on_turn_complete([incoming, {'role': 'assistant', 'content': 'new answer'}])
    eng.on_turn_complete([incoming, {'role': 'assistant', 'content': 'new answer'}])
    assert eng.store.count_events('B') == 6


def test_resume_same_tool_content_distinct_known_ids_preserves_graph(eng):
    eng.on_session_start('A')
    first = _tool_history()
    second = copy.deepcopy(first)
    second[1]['tool_calls'][0]['id'] = 'call-second'
    second[2]['tool_call_id'] = 'call-second'
    eng.on_turn_complete(first + [{'role': 'user', 'content': 'separator'},
                                  {'role': 'assistant', 'content': 'separator answer'}] + second)
    before = eng.store.count_events('A')
    eng.on_session_start('A')
    conn = eng.store._get_connection()
    second_call_id = conn.execute("SELECT id FROM events WHERE type='tool_call' AND tool_call_id='call-second'").fetchone()[0]
    conn.close()
    assert eng._match_replay_prefix(second, 0).get((1, 0)) == second_call_id
    eng.on_turn_complete(second)
    assert eng.store.count_events('A') == before
    conn = eng.store._get_connection()
    pairs = conn.execute("SELECT a.tool_call_id,z.tool_call_id FROM execution_graph g "
                         "JOIN events a ON a.id=g.from_event_id JOIN events z ON z.id=g.to_event_id "
                         "WHERE g.edge_type='tool_output'").fetchall()
    assert {(r[0], r[1]) for r in pairs} == {('call-old', 'call-old'), ('call-second', 'call-second')}
    conn.close()


def test_legacy_completion_distinct_known_calls_at_same_position(eng):
    eng.on_session_start('A')
    first = _tool_history()
    eng.on_turn_complete(first)
    second = copy.deepcopy(first)
    second[1]['tool_calls'][0]['id'] = 'call-second'
    second[2]['tool_call_id'] = 'call-second'
    eng.on_session_start('A')
    eng.on_turn_complete(second)
    assert eng.store.count_events('A') == 6
    conn = eng.store._get_connection()
    pairs = conn.execute("SELECT a.tool_call_id,z.tool_call_id FROM execution_graph g "
                         "JOIN events a ON a.id=g.from_event_id JOIN events z ON z.id=g.to_event_id "
                         "WHERE g.edge_type='tool_output'").fetchall()
    assert {(r[0], r[1]) for r in pairs} == {('call-old', 'call-old'), ('call-second', 'call-second')}
    conn.close()


def test_resume_pending_legacy_call_restores_provider_alias(eng):
    eng.on_session_start('A')
    history = _tool_history()
    eng.on_turn_complete(history[:2])
    conn = eng.store._get_connection()
    conn.execute("UPDATE events SET tool_call_id=NULL WHERE type='tool_call'")
    conn.commit()
    old_call_id = conn.execute("SELECT id FROM events WHERE type='tool_call'").fetchone()[0]
    conn.close()
    eng._pending_tool_calls = {}
    eng._legacy_tool_calls = []
    eng._graph_session_id = None
    eng.on_session_start('A')
    eng.on_turn_complete(history[:3])
    assert eng.store.count_events('A') == 3
    conn = eng.store._get_connection()
    edges = conn.execute("SELECT g.from_event_id,z.tool_call_id FROM execution_graph g "
                         "JOIN events z ON z.id=g.to_event_id WHERE g.edge_type='tool_output'").fetchall()
    assert [(r[0], r[1]) for r in edges] == [(old_call_id, 'call-old')]
    assert eng._legacy_tool_calls == []
    conn.close()


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


@pytest.mark.parametrize('target_status', ['fresh', 'closed'])
def test_auto_fresh_rebind_persists_existing_event_namespace(eng, target_status):
    request = {'role': 'user', 'content': 'repeat the access code'}
    eng.on_session_start('A')
    eng.on_turn_complete([request, {'role': 'assistant', 'content': 'old answer'}])
    eng.on_session_start('A')
    eng.select_context([request], conversation_messages=[request], incoming_message=request)
    nonce = eng._event_id_nonce
    if target_status == 'closed':
        eng.store.create_session('B', 'test')
        conn = eng.store._get_connection()
        conn.execute("UPDATE sessions SET status='closed' WHERE session_id='B'")
        conn.commit()
        conn.close()
    eng._pending_session_id = 'B'
    ingest(eng, [{'role': 'user', 'content': 'a fresh request'}])
    assert eng._event_id_nonce == nonce
    assert eng.store.load_state('B')['_event_id_nonce'] == nonce


@pytest.mark.parametrize('binding', ['hook', 'auto'])
def test_resume_loads_namespace_from_resolved_recovery_state(eng, monkeypatch, binding):
    eng.on_session_start('A')
    ingest(eng, [{'role': 'user', 'content': 'unfinished request'}])
    recovered = eng._make_default_state('A', 'seg_A_1')
    recovered['_event_id_nonce'] = 'recovered-identity'
    monkeypatch.setattr(eng.store, 'load_state', lambda sid: None)
    monkeypatch.setattr(eng, '_recover_state_from_history', lambda *a: recovered)
    if binding == 'hook':
        eng.on_session_start('A')
    else:
        eng.session_id = None
        eng._processed_msg_count = 0
        ingest(eng, [])
    assert eng._event_id_nonce == 'recovered-identity'


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
