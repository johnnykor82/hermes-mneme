"""Unit tests for Stage 7 dependency propagation in ContextRouter."""

import os
import sys
import tempfile
import pytest

from hermes_mneme import router as R
from hermes_mneme import store as S
from hermes_mneme import graph as G  # noqa: F401


class _FakeIndexer:
    def search(self, *a, **kw):
        return []


class _FakeConfig:
    def __init__(self, **overrides):
        self._d = {
            "dependency_max_depth": 4,
            "dependency_decay": 0.6,
        }
        self._d.update(overrides)

    def get(self, key, default=None):
        return self._d.get(key, default)


@pytest.fixture
def store_with_graph():
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    s = S.ContextStore(tmp.name)
    yield s
    os.unlink(tmp.name)


def _build_router(store, **cfg_overrides):
    return R.ContextRouter(
        store=store,
        indexer=_FakeIndexer(),
        config=_FakeConfig(**cfg_overrides),
    )


def _make_input(recent_events, message="why?"):
    return R.RouterInput(
        message=message,
        execution_state={},
        segment_id="seg1",
        recent_events=recent_events,
        token_budget=1000,
        session_id="sess1",
    )


# --- _collect_dependency_anchors ----------------------------------------------

def test_collect_anchors_picks_last_tool_and_assistant(store_with_graph):
    r = _build_router(store_with_graph)
    events = [
        {"id": "e1", "type": "user_message"},
        {"id": "e2", "type": "tool_call"},
        {"id": "e3", "type": "tool_output"},
        {"id": "e4", "type": "assistant_message"},
        {"id": "e5", "type": "user_message"},
    ]
    anchors = r._collect_dependency_anchors(events)
    assert set(anchors) == {"e3", "e4"}


def test_collect_anchors_empty_when_no_relevant_events(store_with_graph):
    r = _build_router(store_with_graph)
    events = [{"id": "e1", "type": "user_message"}]
    assert r._collect_dependency_anchors(events) == []


# --- _build_dependency_bonuses -----------------------------------------------

def test_bonuses_decay_with_depth(store_with_graph):
    r = _build_router(store_with_graph, dependency_max_depth=4, dependency_decay=0.5)
    # Build chain: u1 -> tc1 -> to1 -> am1 ; am1 is anchor.
    g = r.graph
    g.add_edge("u1", "tc1", "follows", "sess1")
    g.add_edge("tc1", "to1", "tool_output", "sess1")
    g.add_edge("to1", "am1", "decision", "sess1")

    inp = _make_input([
        {"id": "u1", "type": "user_message"},
        {"id": "tc1", "type": "tool_call"},
        {"id": "to1", "type": "tool_output"},
        {"id": "am1", "type": "assistant_message"},
    ])
    bonuses = r._build_dependency_bonuses(inp)

    # Anchors: to1 (last tool_output) and am1 (last assistant_message).
    assert bonuses["to1"] == pytest.approx(1.0)
    assert bonuses["am1"] == pytest.approx(1.0)
    # tc1 is 1 hop from to1 (and 2 from am1) → shortest depth wins → 0.5
    assert bonuses["tc1"] == pytest.approx(0.5)
    # u1 is 2 hops from to1 → 0.25
    assert bonuses["u1"] == pytest.approx(0.25)


def test_bonuses_respect_max_depth(store_with_graph):
    r = _build_router(store_with_graph, dependency_max_depth=1, dependency_decay=0.5)
    g = r.graph
    g.add_edge("a", "b", "follows", "sess1")
    g.add_edge("b", "c", "follows", "sess1")
    g.add_edge("c", "d", "follows", "sess1")

    inp = _make_input([
        {"id": "a", "type": "tool_output"},
        {"id": "b", "type": "assistant_message"},
    ])
    bonuses = r._build_dependency_bonuses(inp)

    # Anchors: a, b. From a depth=1 reaches b; from b depth=1 reaches a, c.
    # d is 2 hops away → not included.
    assert "d" not in bonuses
    assert bonuses.get("c") == pytest.approx(0.5)


def test_bonuses_disabled_when_max_depth_zero(store_with_graph):
    r = _build_router(store_with_graph, dependency_max_depth=0)
    r.graph.add_edge("a", "b", "follows", "sess1")
    inp = _make_input([{"id": "a", "type": "tool_output"}])
    assert r._build_dependency_bonuses(inp) == {}


def test_bonuses_empty_without_anchors(store_with_graph):
    r = _build_router(store_with_graph)
    inp = _make_input([{"id": "x", "type": "user_message"}])
    assert r._build_dependency_bonuses(inp) == {}


# --- _calculate_dependency_score lookup --------------------------------------

def test_dependency_score_lookup(store_with_graph):
    r = _build_router(store_with_graph)
    bonuses = {"e1": 0.36, "e2": 1.0}
    assert r._calculate_dependency_score("e1", _make_input([]), bonuses) == 0.36
    assert r._calculate_dependency_score("e2", _make_input([]), bonuses) == 1.0
    assert r._calculate_dependency_score("missing", _make_input([]), bonuses) == 0.0
    assert r._calculate_dependency_score("e1", _make_input([]), None) == 0.0
    assert r._calculate_dependency_score("e1", _make_input([]), {}) == 0.0


# --- MODE_WEIGHTS sanity -----------------------------------------------------

def test_mode_weights_sum_to_one_per_mode():
    for mode, w in R.MODE_WEIGHTS.items():
        total = w["similarity"] + w["recency"] + w["dependency"] + w["type"]
        assert abs(total - 1.0) < 1e-6, f"{mode} weights sum to {total}"


def test_clarification_and_debugging_have_dependency_weight():
    assert R.MODE_WEIGHTS["clarification"]["dependency"] > 0
    assert R.MODE_WEIGHTS["debugging"]["dependency"] > 0
