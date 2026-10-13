"""Logical-turn exclusion uses native incoming identity, not the latest user row."""
import copy
import pytest
from hermes_mneme.engine import CustomRouterContextEngine
from hermes_mneme.config import PluginConfig


@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = CustomRouterContextEngine(str(tmp_path / 'plugin.db'), PluginConfig({
        'active_window_tokens': 100000, 'llm_enrichment_enabled': False,
        'reranker_enabled': False, 'cross_session_bootstrap_enabled': False,
        'reindex_on_model_change': False, 'pass_through_overhead_initial': 0,
        'segmentation_enabled': False,
    }))
    monkeypatch.setattr(e.indexer, 'add_embeddings_batch', lambda records: None)
    monkeypatch.setattr(e.indexer, 'get_embedding', lambda text: None)
    monkeypatch.setattr(e.indexer, 'get_embeddings_batch', lambda texts: [None] * len(texts))
    yield e
    e._embed_executor.shutdown(wait=True)
    e._enrichment_executor.shutdown(wait=True)


def user(text, uid):
    return {'role': 'user', 'content': text, 'message_uid': uid}


def select(e, messages, incoming):
    return e.select_context(copy.deepcopy(messages), conversation_messages=copy.deepcopy(messages),
                            incoming_message=copy.deepcopy(incoming))


def event_ids(e):
    conn = e.store._get_connection()
    try:
        return [r['id'] for r in conn.execute('SELECT id FROM events WHERE session_id=? ORDER BY rowid', (e.session_id,))]
    finally:
        conn.close()


def test_synthetic_user_continuation_keeps_original_turn_boundary(eng):
    eng.on_session_start('A')
    previous = [user('old useful fact', 'old'), {'role': 'assistant', 'content': 'old answer'}]
    eng.on_turn_complete(previous)
    old_ids = set(event_ids(eng))
    incoming = user('recall a prior fact', 'current')
    messages = previous + [incoming]
    select(eng, messages, incoming)
    messages += [{'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'context_search', 'arguments': '{}'}}]},
                 {'role': 'tool', 'tool_call_id': 'c1', 'content': 'retrieved result'},
                 user('arbitrary host-generated continuation', 'synthetic')]
    select(eng, messages, incoming)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - old_ids
    select(eng, messages, incoming)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - old_ids


def test_next_uid_starts_new_turn_even_without_completion_callback(eng):
    eng.on_session_start('A')
    first = user('same question', 'first')
    messages = [first, {'role': 'assistant', 'content': 'useful previous answer'}]
    select(eng, messages, first)
    previous_ids = set(event_ids(eng))
    second = user('same question', 'second')
    messages.append(second)
    select(eng, messages, second)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - previous_ids
    assert len(eng.get_active_turn_event_ids()) == 1


def test_turn_complete_clears_exclusion_even_for_failure(eng):
    eng.on_session_start('A')
    incoming = user('question', 'first')
    select(eng, [incoming], incoming)
    assert eng.get_active_turn_event_ids()
    eng.on_turn_complete([incoming], failed=True, interrupted=True, turn_id='host-turn')
    assert eng.get_active_turn_event_ids() == []


def test_resume_catchup_excludes_only_actual_incoming_turn(eng):
    eng.on_session_start('A')
    history = [user('old fact', 'old'), {'role': 'assistant', 'content': 'old answer'}]
    eng.on_turn_complete(history)
    old_ids = set(event_ids(eng))
    eng.on_session_start('A')
    incoming = user('new question', 'new')
    select(eng, history + [incoming], incoming)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - old_ids
    assert len(eng.get_active_turn_event_ids()) == 1


def test_compression_reassign_retains_actual_turn_event_ids(eng):
    eng.on_session_start('A')
    history = [user('old fact', 'old'), {'role': 'assistant', 'content': 'old answer'}]
    eng.on_turn_complete(history)
    incoming = user('current question', 'current')
    messages = history + [incoming, {'role': 'assistant', 'content': 'in progress'}]
    select(eng, messages, incoming)
    active_ids = set(eng.get_active_turn_event_ids())
    eng.on_session_start('B', boundary_reason='compression', old_session_id='A')
    select(eng, messages[2:], incoming)
    assert set(eng.get_active_turn_event_ids()) == active_ids
    assert active_ids.issubset(set(event_ids(eng)))


def test_reset_new_and_deepcopy_clear_turn_state(eng):
    eng.on_session_start('A')
    incoming = user('current question', 'current')
    select(eng, [incoming], incoming)
    clone = copy.deepcopy(eng)
    try:
        assert clone.get_active_turn_event_ids() == []
    finally:
        clone._embed_executor.shutdown(wait=True)
        clone._enrichment_executor.shutdown(wait=True)
    eng.on_session_start('B')
    assert eng.get_active_turn_event_ids() == []
    select(eng, [user('new question', 'new')], user('new question', 'new'))
    eng.on_session_reset()
    assert eng.get_active_turn_event_ids() == []


def test_no_incoming_marker_does_not_guess_latest_user_boundary(eng):
    eng.on_session_start('A')
    incoming = user('current question', 'current')
    select(eng, [incoming], incoming)
    select(eng, [incoming], None)
    assert eng.get_active_turn_event_ids() == []


def test_uidless_incoming_index_survives_later_synthetic_user(eng):
    eng.on_session_start('A')
    incoming = {'role': 'user', 'content': 'question without host uid'}
    messages = [incoming]
    select(eng, messages, incoming)
    messages += [{'role': 'assistant', 'content': 'partial'}, {'role': 'user', 'content': 'unmarked synthetic'}]
    select(eng, messages, incoming)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng))


@pytest.mark.parametrize('complete', [False, True])
def test_new_uid_at_reused_buffer_position_is_a_new_human_occurrence(eng, complete):
    eng.on_session_start('A')
    first = user('identical question', 'first')
    select(eng, [first], first)
    first_ids = set(event_ids(eng))
    if complete:
        eng.on_turn_complete([first], failed=True)
    second = user('identical question', 'second')
    select(eng, [second], second)
    assert len(event_ids(eng)) == 2
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - first_ids
    assert len(eng.get_active_turn_event_ids()) == 1


def test_absorbed_host_uid_resolves_surviving_incoming_message(eng):
    eng.on_session_start('A')
    old = user('old fact', 'old')
    eng.on_turn_complete([old])
    old_ids = set(event_ids(eng))
    incoming = user('current question', 'original')
    survivor = {**incoming, 'message_uid': 'survivor', '_absorbed_message_uids': ['original']}
    select(eng, [old, survivor], incoming)
    assert set(eng.get_active_turn_event_ids()) == set(event_ids(eng)) - old_ids
    assert len(eng.get_active_turn_event_ids()) == 1


def test_resumed_known_incoming_captures_actual_replay_id(eng):
    eng.on_session_start('A')
    old = user('old fact', 'old')
    incoming = user('current question', 'current')
    history = [old, {'role': 'assistant', 'content': 'old answer'}, incoming,
               {'role': 'assistant', 'content': 'unfinished continuation'}]
    eng.on_turn_complete(history)
    all_ids = event_ids(eng)
    eng.on_session_start('A')
    select(eng, history, incoming)
    assert event_ids(eng) == all_ids
    assert set(eng.get_active_turn_event_ids()) == set(all_ids[-2:])


def test_session_end_clears_current_turn_exclusion(eng):
    eng.on_session_start('A')
    incoming = user('current question', 'current')
    select(eng, [incoming], incoming)
    eng.on_session_end('A', [incoming])
    assert eng.get_active_turn_event_ids() == []


def test_malformed_absorbed_uid_string_does_not_match_by_substring(eng):
    eng.on_session_start('A')
    incoming = user('current question', 'needle')
    unrelated = user('historical fact', 'other')
    unrelated['_absorbed_message_uids'] = 'not-a-list-containing-needle'
    select(eng, [unrelated], incoming)
    assert eng.get_active_turn_event_ids() == []


def test_ambiguous_uidless_occurrences_do_not_guess_a_boundary(eng):
    eng.on_session_start('A')
    incoming = {'role': 'user', 'content': 'same text'}
    messages = [incoming, {'role': 'assistant', 'content': 'prior response'}, copy.deepcopy(incoming)]
    select(eng, messages, incoming)
    assert eng.get_active_turn_event_ids() == []
