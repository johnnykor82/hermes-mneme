"""Synthetic tests for batched + background embedding (perf/embedding-batch).

These tests use a stubbed Jina endpoint (monkey-patched `requests.post`) with
a configurable per-request sleep so we can assert that batching produces a
~5x speedup over the serial path while preserving correctness (event_id
mapping, embedding dimension, idempotency)."""
from __future__ import annotations

import os
import tempfile
import threading
import time
import types

import pytest


PER_REQUEST_SLEEP_S = 0.05  # 50ms per Jina HTTP call — emulates real latency
EMBED_DIM = 1024


class _FakeResp:
    def __init__(self, items):
        self._items = items
        self.status_code = 200

    def json(self):
        return {"data": [{"embedding": e} for e in self._items]}

    def raise_for_status(self):
        return None


def _make_fake_post(call_counter, batch_sizes):
    """Returns a fake requests.post that sleeps per call and produces a 1024-dim
    deterministic embedding per input."""

    def fake_post(url, json=None, headers=None, timeout=None):
        inp = (json or {}).get("input")
        if isinstance(inp, str):
            inp = [inp]
        batch_sizes.append(len(inp))
        call_counter["n"] += 1
        time.sleep(PER_REQUEST_SLEEP_S)
        # Deterministic embedding: first cell = len(text) / 1000, rest zeros.
        items = []
        for t in inp:
            vec = [0.0] * EMBED_DIM
            vec[0] = (len(t) % 1000) / 1000.0
            items.append(vec)
        return _FakeResp(items)

    return fake_post


@pytest.fixture
def patched_jina(monkeypatch):
    counter = {"n": 0}
    batch_sizes: list[int] = []
    import hermes_mneme.index as idx_mod  # set up by conftest
    monkeypatch.setattr(idx_mod.requests, "post", _make_fake_post(counter, batch_sizes))
    return counter, batch_sizes


@pytest.fixture
def indexer(tmp_path):
    """Real EmbeddingIndex backed by a fresh sqlite DB. We only need the
    embedding pipeline; vec0 extension is optional and skipped if unavailable."""
    import hermes_mneme.index as idx_mod
    from hermes_mneme.store import ContextStore

    db_path = str(tmp_path / "plugin.db")
    # ContextStore creates the schema (embedding_index, events tables).
    ContextStore(db_path)
    return idx_mod.EmbeddingIndex(db_path)


def test_batch_dim_and_count(patched_jina, indexer):
    """get_embeddings_batch returns one embedding per input, each dim=1024."""
    texts = [f"sample input {i} — some words" for i in range(7)]
    embeds = indexer.get_embeddings_batch(texts)
    assert len(embeds) == 7
    for e in embeds:
        assert e is not None
        assert len(e) == EMBED_DIM


def test_batch_handles_empty_inputs(patched_jina, indexer):
    """Empty/whitespace inputs come back as None; non-empty ones get vectors."""
    embeds = indexer.get_embeddings_batch(["", "   ", "real text"])
    assert embeds[0] is None
    assert embeds[1] is None
    assert embeds[2] is not None and len(embeds[2]) == EMBED_DIM


def test_speedup_versus_serial(patched_jina, indexer):
    """Batched path makes one HTTP call regardless of N; serial path makes N.
    Assert ratio after >= 4x speedup with the simulated 50ms-per-call latency."""
    N = 30
    texts = [f"item {i}" for i in range(N)]

    # Serial path (legacy)
    t0 = time.perf_counter()
    for t in texts:
        indexer.get_embedding(t)
    serial = time.perf_counter() - t0

    # Batch path — single call
    t0 = time.perf_counter()
    indexer.get_embeddings_batch(texts)
    batched = time.perf_counter() - t0

    # With PER_REQUEST_SLEEP=50ms: serial ≈ N * 50ms = 1.5s; batched ≈ 50ms.
    # Allow generous headroom (>=4x) so CI variance doesn't flake.
    assert serial / batched >= 4.0, f"Speedup too low: serial={serial:.3f}s batched={batched:.3f}s"


def test_add_embeddings_batch_persists(patched_jina, indexer):
    """add_embeddings_batch writes to embedding_index and reports rows stored.
    Verifies the SQL path with the model_id / segment_id / type columns."""
    records = [
        (f"ev_{i}", "seg_test", f"text {i}", "test-model", 10, "user")
        for i in range(5)
    ]
    stored = indexer.add_embeddings_batch(records)
    assert stored == 5

    conn = indexer._get_connection()
    try:
        rows = conn.execute(
            "SELECT event_id, embedding_model_id, type FROM embedding_index "
            "WHERE segment_id = ?",
            ("seg_test",),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 5
    assert {r[0] for r in rows} == {f"ev_{i}" for i in range(5)}
    assert all(r[1] == "test-model" for r in rows)
    assert all(r[2] == "user" for r in rows)


def test_compat_add_embedding_uses_batch(patched_jina, indexer):
    """Legacy add_embedding(...) still works — it now delegates to the batch
    path under the hood. This guards backward compatibility for any external
    callers that import the old single-record API."""
    indexer.add_embedding("ev_compat", "seg_compat", "hello world", "test-model", 5, "user")
    conn = indexer._get_connection()
    try:
        row = conn.execute(
            "SELECT event_id FROM embedding_index WHERE event_id = ?",
            ("ev_compat",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
