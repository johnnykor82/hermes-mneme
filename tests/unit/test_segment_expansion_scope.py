"""Segment navigation must preserve the source selected by the agent."""
import json

import pytest

from hermes_mneme.store import ContextStore
from hermes_mneme.tools import ContextTools


@pytest.fixture
def archive(tmp_path):
    store = ContextStore(str(tmp_path / 'memory.db'))
    for sid in ['active', 'old-a', 'old-b']:
        store.create_session(sid, 'tui')
    tools = ContextTools(store, None, None, None)
    def add(sid, segment, text):
        return store.add_event(session_id=sid, segment_id=segment,
            event_type='user_message', role='user', content=text, tool_name=None,
            tool_input=None, token_estimate=10)[0]
    return tools, add


def test_direct_unique_foreign_segment_uses_its_source_session(archive):
    tools, add = archive
    source = add('old-a', 'unique-old-a', 'Первоначальное требование')
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'segment_id': 'unique-old-a'}, 'active'))
    assert result['session_id'] == 'old-a'
    assert [e['event_id'] for e in result['events']] == [source]


def test_seed_with_explicit_segment_keeps_the_seed_session(archive):
    tools, add = archive
    source = add('old-a', 'legacy', 'Нужный разговор')
    other = add('active', 'legacy', 'Другой разговор')
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'segment_id': 'legacy', 'seed_event_id': source}, 'active'))
    assert result['session_id'] == 'old-a'
    assert [e['event_id'] for e in result['events']] == [source]
    assert other not in [e['event_id'] for e in result['events']]


def test_seed_and_segment_mismatch_is_an_explicit_error(archive):
    tools, add = archive
    source = add('old-a', 'topic-a', 'Одна тема')
    add('old-b', 'topic-b', 'Другая тема')
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'segment_id': 'topic-b', 'seed_event_id': source}, 'active'))
    assert 'error' in result
    assert 'events' not in result


def test_shared_legacy_segment_requires_a_seed(archive):
    tools, add = archive
    add('old-a', 'seg_1', 'Первая сессия')
    add('old-b', 'seg_1', 'Вторая сессия')
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'segment_id': 'seg_1'}, 'active'))
    assert 'error' in result
    assert 'seed_event_id' in result['error']
    assert 'events' not in result


def test_unknown_segment_is_not_an_empty_success(archive):
    tools, _ = archive
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'segment_id': 'missing'}, 'active'))
    assert 'error' in result


def test_bounded_spine_discloses_omitted_middle_events(archive):
    tools, add = archive
    ids = [add('old-a', 'long-topic', f'Требование {i}') for i in range(30)]
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'seed_event_id': ids[0]}, 'active'))
    assert result['total_events'] == 30
    assert result['returned_events'] == len(result['events']) == 15
    assert result['is_partial'] is True
    assert result['preview_only'] is True
    assert ids[12] not in [e['event_id'] for e in result['events']]


def test_one_long_event_still_has_only_a_preview(archive):
    tools, add = archive
    source = add('old-a', 'long-content', 'Длинный исходный текст. ' * 50)
    result = json.loads(tools._expand_context(
        {'mode': 'segment', 'seed_event_id': source}, 'active'))
    assert result['total_events'] == result['returned_events'] == 1
    assert result['is_partial'] is True
    assert result['preview_only'] is True
