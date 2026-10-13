"""Memory search must retain provenance needed for careful attribution."""

import json
from unittest.mock import Mock

from hermes_mneme.store import ContextStore
from hermes_mneme.tools import ContextTools


def test_search_and_fetch_preserve_source_role_scope_and_truncation(tmp_path):
    store = ContextStore(str(tmp_path / 'memory.db'))
    store.create_session('current', 'tui')
    store.create_session('earlier', 'tui')

    def add(session_id, segment_id, event_type, role, content, tool_name=None):
        return store.add_event(
            session_id=session_id, segment_id=segment_id, event_type=event_type,
            role=role, content=content, tool_name=tool_name, tool_input=None,
            token_estimate=10,
        )[0]

    source = (
        'Минимум 100 записей в каталоге А за смену. Для команды Б действует '
        'та же нижняя граница. ' + ('Исторический контекст. ' * 400)
    )
    user_id = add('earlier', 'plan', 'user_message', 'user', source)
    assistant_content = 'Предлагаю считать 100 общим итогом для обеих групп.'
    assistant_id = add(
        'earlier', 'plan', 'assistant_message', 'assistant',
        assistant_content,
    )
    call_content = 'Requested action: publish the selected records.'
    call_id = add(
        'earlier', 'plan', 'tool_call', 'assistant', call_content, 'publisher'
    )
    output_content = 'Tool returned success for the request.'
    output_id = add(
        'earlier', 'plan', 'tool_output', 'tool', output_content, 'publisher'
    )

    class Index:
        def search_global(self, query, top_k):
            return [{'event_id': event_id} for event_id in
                    (user_id, assistant_id, call_id, output_id)]

    tools = ContextTools(store, Index(), None, None)
    found = json.loads(tools._context_search(
        {'query': 'каталог команда записи', 'session': 'all', 'segment': 'all'}, 'current'
    ))
    by_id = {item['event_id']: item for item in found}

    assert by_id[user_id]['type'] == 'user_message'
    assert by_id[user_id]['role'] == 'user'
    assert by_id[user_id]['source_session_id'] == 'earlier'
    assert by_id[user_id]['segment_id'] == 'plan'
    assert by_id[user_id]['timestamp']
    assert by_id[user_id]['truncated'] is True
    assert by_id[user_id]['original_chars'] == len(source)
    assert by_id[assistant_id]['type'] == 'assistant_message'
    assert by_id[call_id]['type'] == 'tool_call'
    assert by_id[output_id]['type'] == 'tool_output'

    excerpt = json.loads(tools._fetch_event({'event_id': user_id}))
    assert excerpt['truncated'] is True
    assert excerpt['original_chars'] == len(source)
    original_by_id = {
        user_id: source,
        assistant_id: assistant_content,
        call_id: call_content,
        output_id: output_content,
    }
    for event_id, original in original_by_id.items():
        full = json.loads(tools._fetch_event({'event_id': event_id, 'full': True}))
        assert full['content'] == original
        assert full['type'] == by_id[event_id]['type']
        assert full['role'] == by_id[event_id]['role']
        assert full['session_id'] == by_id[event_id]['source_session_id']
        assert full['segment_id'] == by_id[event_id]['segment_id']


def test_tool_guidance_exposes_attribution_and_search_policy():
    descriptions = {
        tool['name']: tool['description']
        for tool in ContextTools(None, None, None, None).get_tool_schemas()
    }
    search = descriptions['context_search']
    fetch = descriptions['fetch_event']

    assert 'top-k result that omits a fact does not show' in search
    assert 'increase k' in search and "segment='all'" in search
    assert 'alternative query wording' in search and 'expand_context' in search
    assert 'thousands of messages' in search and 'multiple sequential searches' in search
    assert 'no fixed limit on search attempts' in search
    assert 'describe what scope you searched' in search
    assert 'user_message' in search and 'assistant_message' in search
    assert 'only when the user framed it as one' in search
    assert 'not a user requirement' in search
    assert 'proof that an action was completed' in search
    assert 'tool_call' in search and 'tool_output' in search
    assert 'may not establish an external side effect' in search
    assert 'Do not state unsupported historical facts' in search
    assert 'assistant_message' in fetch and 'tool_output' in fetch


def test_native_and_legacy_host_pins_receive_attribution_guidance():
    from hermes_mneme.engine import CustomRouterContextEngine

    schemas = ContextTools(None, None, None, None).get_tool_schemas()
    other = {'type': 'function', 'function': {'name': 'other', 'description': 'Keep me'}}
    for legacy in (False, True):
        pinned = []
        for schema in schemas:
            function = {**schema, 'description': 'Old guidance'}
            if legacy:
                function['input_schema'] = function.pop('parameters')
                function['parameters'] = {'type': 'object', 'properties': {}, 'required': []}
            pinned.append({'type': 'function', 'function': function})
        db = Mock()
        db.get_session.return_value = {
            'tool_names': json.dumps({'version': 'host', 'tools': [other, *pinned]})
        }
        host = type('Host', (), {'get_tool_schemas': lambda self: schemas})()

        CustomRouterContextEngine.bind_session_state(
            host, session_db=db, session_id='session'
        )

        updated = db.update_session_tool_names.call_args.args[1]
        refreshed = updated['tools'][1]['function']['description']
        assert updated['version'] == 'host'
        assert updated['tools'][0] == other
        assert 'proof that an action was completed' in refreshed
        assert 'no fixed limit on search attempts' in refreshed
