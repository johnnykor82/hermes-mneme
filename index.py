import requests
import logging
import sqlite3
import json
import struct
import math
import time
from typing import List, Optional, Dict, Any

try:
    import sqlite_vec
    _SQLITE_VEC_IMPORT_OK = True
    _SQLITE_VEC_IMPORT_ERROR = None
except Exception as _e:
    sqlite_vec = None
    _SQLITE_VEC_IMPORT_OK = False
    _SQLITE_VEC_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

# Jina Configuration
JINA_API_URL = "http://127.0.0.1:8000/v1/embeddings"
JINA_API_KEY = "1234"
JINA_MODEL = "jina-embeddings-v5-text-small-retrieval-mlx"

class EmbeddingIndex:
    """Manages embedding generation and retrieval using Jina and sqlite-vec."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedding_dim = 1024
        self._vec_available = False
        self._init_vec_extension()
        # A10: simple circuit breaker for the Jina HTTP endpoint. After N
        # consecutive failures (timeouts, connection errors, non-2xx) we
        # short-circuit get_embedding() for `cooldown_seconds`. Events still
        # land in the events table — only embedding indexing is skipped, and
        # retrieval falls back on keyword/recency until the breaker resets.
        # Tunable via PluginConfig but lives here so index.py is the single
        # owner of the policy.
        self._cb_failure_threshold = 3
        self._cb_cooldown_seconds = 300
        self._cb_consecutive_failures = 0
        self._cb_open_until_ts = 0.0
        self._cb_logged_disabled = False

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        if self._vec_available:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                pass
        return conn

    def _init_vec_extension(self):
        """Test if sqlite-vec is available; cache on self._vec_available."""
        self._vec_available = False
        if not _SQLITE_VEC_IMPORT_OK:
            logger.error(
                f"sqlite-vec python package not importable: {_SQLITE_VEC_IMPORT_ERROR}. "
                f"Falling back to Python cosine."
            )
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            row = conn.execute("SELECT vec_version()").fetchone()
            conn.close()
            self._vec_available = True
            logger.info(f"sqlite-vec extension loaded successfully. Version: {row[0] if row else '?'}")
        except Exception as e:
            logger.error(f"sqlite-vec load failed: {e}. Falling back to Python cosine.")

    def _circuit_breaker_record_success(self) -> None:
        if self._cb_consecutive_failures or self._cb_open_until_ts:
            logger.info("Embedding circuit breaker reset (success).")
        self._cb_consecutive_failures = 0
        self._cb_open_until_ts = 0.0
        self._cb_logged_disabled = False

    def _circuit_breaker_record_failure(self, err: Exception) -> None:
        self._cb_consecutive_failures += 1
        if self._cb_consecutive_failures >= self._cb_failure_threshold:
            self._cb_open_until_ts = time.time() + self._cb_cooldown_seconds
            if not self._cb_logged_disabled:
                logger.warning(
                    f"Embedding endpoint disabled for {self._cb_cooldown_seconds}s after "
                    f"{self._cb_consecutive_failures} consecutive failures (last: {err}). "
                    f"Events will continue to be stored; retrieval falls back to keyword/recency."
                )
                self._cb_logged_disabled = True

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Jina API. Skips the HTTP call if the circuit
        breaker is open (3+ consecutive failures within the last 5 min)."""
        if not text or not text.strip():
            return None
        # A10: circuit breaker — skip HTTP entirely while open.
        if self._cb_open_until_ts and time.time() < self._cb_open_until_ts:
            return None
        # Cooldown elapsed: half-open. Reset failure counter so a single
        # successful call closes the breaker, but a fresh failure re-opens it.
        if self._cb_open_until_ts and time.time() >= self._cb_open_until_ts:
            logger.info("Embedding circuit breaker entering half-open (cooldown elapsed).")
            self._cb_open_until_ts = 0.0
            self._cb_consecutive_failures = 0
            self._cb_logged_disabled = False

        try:
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "input": text,
                "model": JINA_MODEL
            }
            response = requests.post(JINA_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0]["embedding"]
                logger.debug(f"Got embedding of length {len(embedding)} for text: {text[:50]}...")
                self._circuit_breaker_record_success()
                return embedding
            else:
                logger.error(f"Unexpected Jina response: {data}")
                self._circuit_breaker_record_failure(RuntimeError("unexpected response shape"))
                return None
        except Exception as e:
            logger.error(f"Failed to get embedding from Jina: {e}")
            self._circuit_breaker_record_failure(e)
            return None

    def add_embedding(self, event_id: str, segment_id: str, text_to_embed: str,
                      model_id: str, token_count: int, event_type: str):
        """Generate embedding and store in DB. Also insert into vec0 if available."""
        embedding = self.get_embedding(text_to_embed)
        if not embedding:
            logger.warning(f"Skipping embedding for event {event_id} due to generation failure.")
            return

        conn = self._get_connection()
        try:
            embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

            conn.execute(
                """INSERT OR REPLACE INTO embedding_index
                   (event_id, segment_id, embedding, embedding_model_id, token_count, type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event_id, segment_id, embedding_bytes, model_id, token_count, event_type)
            )

            # Insert into vec0 for fast KNN search
            if self._vec_available:
                try:
                    conn.execute("INSERT OR REPLACE INTO vec_items (embedding) VALUES (?)", (embedding_bytes,))
                    vec_rowid = conn.execute("SELECT rowid FROM vec_items WHERE rowid = last_insert_rowid()").fetchone()
                    if vec_rowid:
                        conn.execute(
                            "INSERT OR REPLACE INTO vec_items_meta (rowid, event_id, embedding_model_id, segment_id) VALUES (?, ?, ?, ?)",
                            (vec_rowid[0], event_id, model_id, segment_id)
                        )
                except Exception as _vec_err:
                    logger.warning(f"vec0 insert failed for {event_id}: {_vec_err}")

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
        finally:
            conn.close()

    def search(self, query_text: str, session_id: str, segment_id: str = None, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """Search for similar embeddings. Uses vec0 KNN if available, else Python cosine.
        top_k=None or 0 → unlimited (return everything semantically close)."""
        query_emb = self.get_embedding(query_text)
        if not query_emb:
            return []

        if self._vec_available:
            return self._search_vec0(query_emb, session_id, segment_id, top_k)
        else:
            return self._search_python(query_emb, session_id, segment_id, top_k)

    def search_global(self, query_text: str, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """Cross-session semantic search — no session filter applied.
        top_k=None/0 → unlimited."""
        query_emb = self.get_embedding(query_text)
        if not query_emb:
            return []
        if self._vec_available:
            return self._search_vec0_multi(query_emb, session_ids=None, segment_id=None, top_k=top_k)
        return self._search_python_multi(query_emb, session_ids=None, segment_id=None, top_k=top_k)

    def search_sessions(self, query_text: str, session_ids: List[str], top_k: Optional[int] = 20,
                        segment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Multi-session semantic search — filters to events belonging to any of session_ids.
        top_k=None/0 → unlimited."""
        if not session_ids:
            return []
        query_emb = self.get_embedding(query_text)
        if not query_emb:
            return []
        if self._vec_available:
            return self._search_vec0_multi(query_emb, session_ids=session_ids, segment_id=segment_id, top_k=top_k)
        return self._search_python_multi(query_emb, session_ids=session_ids, segment_id=segment_id, top_k=top_k)

    def _search_vec0_multi(self, query_emb: List[float], session_ids: Optional[List[str]],
                           segment_id: Optional[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """vec0 KNN with optional multi-session / no-session filter.
        session_ids=None → cross-session (no session filter).
        session_ids=[...] → IN-clause filter."""
        conn = self._get_connection()
        try:
            query_bytes = struct.pack(f'{len(query_emb)}f', *query_emb)
            unlimited = not top_k

            count_sql = "SELECT COUNT(*) FROM vec_items_meta m JOIN events e ON m.event_id=e.id WHERE m.embedding_model_id = ?"
            count_params: list = [JINA_MODEL]
            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                count_sql += f" AND e.session_id IN ({placeholders})"
                count_params.extend(session_ids)

            if unlimited:
                row = conn.execute(count_sql, count_params).fetchone()
                knn_k = max(int(row[0] or 0), 1)
            else:
                knn_k = max(top_k * 4, top_k)

            sql = """
                WITH knn AS (
                    SELECT rowid, distance
                    FROM vec_items
                    WHERE embedding MATCH ? AND k = ?
                )
                SELECT m.event_id, m.segment_id, m.embedding_model_id, knn.distance, e.session_id
                FROM knn
                JOIN vec_items_meta m ON knn.rowid = m.rowid
                JOIN events e ON m.event_id = e.id
                WHERE m.embedding_model_id = ?
            """
            params: list = [query_bytes, knn_k, JINA_MODEL]
            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                sql += f" AND e.session_id IN ({placeholders})"
                params.extend(session_ids)
            if segment_id and segment_id != "all":
                sql += " AND e.segment_id = ?"
                params.append(segment_id)
            sql += " ORDER BY knn.distance"
            if not unlimited:
                sql += " LIMIT ?"
                params.append(top_k)

            cursor = conn.execute(sql, params)
            results = []
            for row in cursor:
                distance = row[3]
                similarity = 1.0 / (1.0 + distance)
                results.append({
                    "event_id": row[0],
                    "segment_id": row[1],
                    "embedding_model_id": row[2],
                    "similarity": similarity,
                    "session_id": row[4],
                })
            return results
        except Exception as e:
            logger.warning(f"vec0 multi-search failed: {e}. Falling back to Python cosine.")
            return self._search_python_multi(query_emb, session_ids, segment_id, top_k)
        finally:
            conn.close()

    def _search_python_multi(self, query_emb: List[float], session_ids: Optional[List[str]],
                             segment_id: Optional[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """Python cosine fallback for cross-session / multi-session search."""
        conn = self._get_connection()
        try:
            base_sql = """
                SELECT e.id, i.embedding, i.type, i.embedding_model_id, i.segment_id, e.session_id
                FROM embedding_index i
                JOIN events e ON i.event_id = e.id
                WHERE i.embedding_model_id = ?
            """
            params: list = [JINA_MODEL]
            if session_ids:
                placeholders = ",".join("?" for _ in session_ids)
                base_sql += f" AND e.session_id IN ({placeholders})"
                params.extend(session_ids)
            if segment_id and segment_id != "all":
                base_sql += " AND e.segment_id = ?"
                params.append(segment_id)

            cursor = conn.execute(base_sql, params)
            results = []
            query_norm = math.sqrt(sum(x * x for x in query_emb))
            for row in cursor:
                event_id, emb_bytes, ev_type, model_id, seg_id, sess_id = row
                if not emb_bytes:
                    continue
                try:
                    emb_len = len(emb_bytes) // 4
                    emb = struct.unpack(f'{emb_len}f', emb_bytes)
                    dot_product = sum(a * b for a, b in zip(query_emb, emb))
                    emb_norm = math.sqrt(sum(x * x for x in emb))
                    if emb_norm == 0 or query_norm == 0:
                        continue
                    similarity = dot_product / (emb_norm * query_norm)
                    results.append({
                        "event_id": event_id,
                        "similarity": similarity,
                        "type": ev_type,
                        "embedding_model_id": model_id,
                        "segment_id": seg_id,
                        "session_id": sess_id,
                    })
                except Exception as e:
                    logger.warning(f"Error processing embedding for {event_id}: {e}")
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results if not top_k else results[:top_k]
        except Exception as e:
            logger.error(f"Multi-session search failed: {e}")
            return []
        finally:
            conn.close()

    def _search_vec0(self, query_emb: List[float], session_id: str, segment_id: str = None, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """Fast KNN search using sqlite-vec. top_k=None/0 → unlimited."""
        if not self._vec_available:
            logger.info("_search_vec0: vec not available, falling back to Python")
            return self._search_python(query_emb, session_id, segment_id, top_k)
        conn = self._get_connection()
        try:
            query_bytes = struct.pack(f'{len(query_emb)}f', *query_emb)

            unlimited = not top_k  # None or 0
            # vec0 KNN requires `k = ?` directly on the vec_items query.
            # When unlimited: probe full table size for this session.
            if unlimited:
                row = conn.execute(
                    "SELECT COUNT(*) FROM vec_items_meta m JOIN events e ON m.event_id=e.id "
                    "WHERE e.session_id = ? AND m.embedding_model_id = ?",
                    (session_id, JINA_MODEL)
                ).fetchone()
                knn_k = max(int(row[0] or 0), 1)
            else:
                # Over-fetch so post-filter by session/segment leaves enough.
                knn_k = max(top_k * 4, top_k)
            use_segment = bool(segment_id and segment_id != "all")
            sql = """
                WITH knn AS (
                    SELECT rowid, distance
                    FROM vec_items
                    WHERE embedding MATCH ? AND k = ?
                )
                SELECT m.event_id, m.segment_id, m.embedding_model_id, knn.distance
                FROM knn
                JOIN vec_items_meta m ON knn.rowid = m.rowid
                JOIN events e ON m.event_id = e.id
                WHERE e.session_id = ?
                  AND m.embedding_model_id = ?
            """
            params: list = [query_bytes, knn_k, session_id, JINA_MODEL]
            if use_segment:
                sql += " AND e.segment_id = ?"
                params.append(segment_id)
            sql += " ORDER BY knn.distance"
            if not unlimited:
                sql += " LIMIT ?"
                params.append(top_k)

            cursor = conn.execute(sql, params)
            results = []
            for row in cursor:
                # Convert distance to similarity (sqlite-vec returns L2 distance)
                distance = row[3]
                similarity = 1.0 / (1.0 + distance)  # Convert L2 to similarity
                results.append({
                    "event_id": row[0],
                    "segment_id": row[1],
                    "embedding_model_id": row[2],
                    "similarity": similarity
                })
            return results
        except Exception as e:
            logger.warning(f"vec0 search failed: {e}. Falling back to Python cosine.")
            return self._search_python(query_emb, session_id, segment_id, top_k)
        finally:
            conn.close()

    def _search_python(self, query_emb: List[float], session_id: str, segment_id: str = None, top_k: Optional[int] = 20) -> List[Dict[str, Any]]:
        """Fallback: Python cosine similarity scan. top_k=None/0 → unlimited."""
        conn = self._get_connection()
        try:
            base_sql = """
                SELECT e.id, i.embedding, i.type, i.embedding_model_id, i.segment_id
                FROM embedding_index i
                JOIN events e ON i.event_id = e.id
                WHERE e.session_id = ? AND i.embedding_model_id = ?
            """
            params: list = [session_id, JINA_MODEL]

            if segment_id and segment_id != "all":
                base_sql += " AND e.segment_id = ?"
                params.append(segment_id)

            cursor = conn.execute(base_sql, params)

            results = []
            query_norm = math.sqrt(sum(x * x for x in query_emb))

            for row in cursor:
                event_id, emb_bytes, ev_type, model_id, seg_id = row
                if not emb_bytes:
                    continue

                try:
                    emb_len = len(emb_bytes) // 4
                    emb = struct.unpack(f'{emb_len}f', emb_bytes)

                    dot_product = sum(a * b for a, b in zip(query_emb, emb))
                    emb_norm = math.sqrt(sum(x * x for x in emb))

                    if emb_norm == 0 or query_norm == 0:
                        continue
                    similarity = dot_product / (emb_norm * query_norm)

                    results.append({
                        "event_id": event_id,
                        "similarity": similarity,
                        "type": ev_type,
                        "embedding_model_id": model_id,
                        "segment_id": seg_id
                    })
                except Exception as e:
                    logger.warning(f"Error processing embedding for {event_id}: {e}")

            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results if not top_k else results[:top_k]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        finally:
            conn.close()
