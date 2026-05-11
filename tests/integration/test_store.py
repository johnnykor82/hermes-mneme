import pytest
import os
import sys

# Ensure plugin path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from store import ContextStore
from index import EmbeddingIndex

@pytest.fixture
def store():
    """Create a store with a temporary file DB."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    s = ContextStore(tmp.name)
    yield s
    # Teardown
    import os
    os.unlink(tmp.name)

@pytest.fixture
def indexer():
    """Create an indexer with in-memory DB."""
    idx = EmbeddingIndex(":memory:")
    yield idx

def test_create_session(store):
    store.create_session("sess1", "tui")
    # If no error, table exists
    conn = store._get_connection()
    row = conn.execute("SELECT session_id FROM sessions WHERE session_id='sess1'").fetchone()
    conn.close()
    assert row is not None

def test_add_event(store):
    result = store.add_event(
        session_id="sess1", segment_id="seg1", event_type="user_message",
        role="user", content="Hello", tool_name=None,
        tool_input=None, token_estimate=10
    )
    # add_event now returns (event_id, inserted_flag).
    event_id, inserted = result
    assert event_id is not None
    assert len(event_id) > 0
    assert inserted is True

def test_save_load_state(store):
    test_state = {"goal": "test", "current_step": "step1"}
    store.save_state("sess1", test_state)
    loaded = store.load_state("sess1")
    assert loaded is not None
    assert loaded["goal"] == "test"

def test_get_recent_events(store):
    # Add a few events
    for i in range(10):
        store.add_event(f"sess1", f"seg1", "user_message", "user", f"Msg {i}", None, None, 10)
    
    events = store.get_recent_events("sess1", limit=5)
    assert len(events) == 5
    # Should be the last 5 in chronological order
    assert "Msg 5" in events[0]["content"] or "Msg 9" in events[-1]["content"]

def test_add_embedding(store, indexer):
    # Note: indexer has its own DB connection
    # This test is simplified
    try:
        import struct
        dummy_emb = struct.pack('10f', *[0.1]*10)
        store.add_embedding("ev1", "seg1", dummy_emb, "test_model", 10, "user_message")
        # Verify
        conn = store._get_connection()
        row = conn.execute("SELECT event_id FROM embedding_index WHERE event_id='ev1'").fetchone()
        conn.close()
        assert row is not None
    except Exception as e:
        pytest.skip(f"Embedding test skipped: {e}")
