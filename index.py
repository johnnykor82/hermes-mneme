import requests
import logging
import sqlite3
import struct
import math
import time
import heapq
import re
import threading
from collections import OrderedDict
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

    def __init__(self, db_path: str, config=None):
        self.db_path = db_path
        cfg = config if config is not None else {}
        self.embedding_dim = int(cfg.get("embedding_dim", 1024))
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.model_id = cfg.get("embedding_model", JINA_MODEL)
        self.embedding_model = self.model_id
        self.embedding_api_key = cfg.get("embedding_api_key", JINA_API_KEY)
        endpoint = cfg.get("embedding_endpoint", JINA_API_URL).rstrip("/")
        self.embedding_endpoint = (endpoint if endpoint.endswith("/embeddings") else
                                   endpoint + ("/embeddings" if endpoint.endswith("/v1") else "/v1/embeddings"))
        self._embedding_cache = OrderedDict()
        self._embedding_cache_lock = threading.RLock()
        self._lexical_ready = False
        self._lexical_lock = threading.RLock()
        self._vec_dim = None
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
            schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='vec_items'").fetchone()
            match = re.search(r"float\[(\d+)\]", schema[0]) if schema else None
            self._vec_dim = int(match.group(1)) if match else None
            conn.close()
            self._vec_available = True
            if self._vec_dim is not None and self._vec_dim != self.embedding_dim:
                logger.info("Configured embedding dimension %s differs from existing vec0 dimension %s; preserving vec0 and using canonical exact search", self.embedding_dim, self._vec_dim)
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
        """Reuse successful same-model query vectors across segmentation/retrieval.

        Exact input is the cache key: whitespace normalization can alter the
        embedding provider's semantics. Limit entries and cacheable input size.
        The lock coalesces concurrent identical single-query requests.
        """
        key = (self.model_id, self.embedding_endpoint, self.embedding_dim, text)
        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(key)
            if cached is not None:
                self._embedding_cache.move_to_end(key)
                return list(cached)
            results = self.get_embeddings_batch([text])
            embedding = results[0] if results else None
            if self._valid_embedding(embedding) and text and len(text) <= 64000:
                self._embedding_cache[key] = tuple(embedding)
                while len(self._embedding_cache) > 128:
                    self._embedding_cache.popitem(last=False)
            return embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Batch-embed via one Jina POST. Returns same-length list with None for
        empty/whitespace inputs or when the circuit breaker is open. Jina's
        OpenAI-compatible API accepts a list in `input`."""
        n = len(texts)
        if n == 0:
            return []
        idx_to_pos: List[int] = []
        clean: List[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                idx_to_pos.append(i)
                clean.append(t)
        out: List[Optional[List[float]]] = [None] * n
        if not clean:
            return out
        if self._cb_open_until_ts and time.time() < self._cb_open_until_ts:
            return out
        if self._cb_open_until_ts and time.time() >= self._cb_open_until_ts:
            logger.info("Embedding circuit breaker entering half-open (cooldown elapsed).")
            self._cb_open_until_ts = 0.0
            self._cb_consecutive_failures = 0
            self._cb_logged_disabled = False
        try:
            headers = {
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            }
            payload = {"input": clean, "model": self.model_id}
            response = requests.post(self.embedding_endpoint, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            items = (response.json() or {}).get("data") or []
            if len(items) != len(clean):
                logger.error(f"Jina batch returned {len(items)}/{len(clean)} embeddings")
                self._circuit_breaker_record_failure(RuntimeError("count mismatch"))
                return out
            seen = set()
            for fallback_index, item in enumerate(items):
                item_index = item.get("index", fallback_index)
                if not isinstance(item_index, int) or item_index not in range(len(clean)) or item_index in seen:
                    raise ValueError("invalid embedding response index")
                seen.add(item_index)
                emb = item.get("embedding")
                if not self._valid_embedding(emb):
                    raise ValueError("embedding dimension or values do not match configuration")
                out[idx_to_pos[item_index]] = emb
            self._circuit_breaker_record_success()
            return out
        except Exception as e:
            logger.error(f"Failed to batch-embed from Jina: {e}")
            self._circuit_breaker_record_failure(e)
            return out

    def add_embedding(self, event_id: str, segment_id: str, text_to_embed: str,
                      model_id: str, token_count: int, event_type: str):
        """Single-record path. Internally batches one record."""
        self.add_embeddings_batch([
            (event_id, segment_id, text_to_embed, model_id, token_count, event_type)
        ])

    def add_embeddings_batch(self, records: List[tuple]) -> int:
        """Embed and persist many events in a single Jina request + one DB
        transaction. Each record is (event_id, segment_id, text, model_id,
        token_count, type). Returns count of rows successfully stored."""
        if not records:
            return 0
        texts = [r[2] for r in records]
        embeddings = self.get_embeddings_batch(texts)
        conn = self._get_connection()
        stored = 0
        try:
            # Embedding HTTP happens before the transaction. Resolve the event's
            # current segment under the write lock: a compression rebind can
            # have moved it while the request was in flight.
            conn.execute("BEGIN IMMEDIATE")
            for rec, emb in zip(records, embeddings):
                event_id, segment_id, _text, model_id, token_count, ev_type = rec
                current = conn.execute("SELECT segment_id FROM events WHERE id=?", (event_id,)).fetchone()
                if current:
                    segment_id = current[0]
                if not self._valid_embedding(emb):
                    logger.warning("Skipping embedding with invalid dimension/values for event %s", event_id)
                    continue
                try:
                    embedding_bytes = struct.pack(f'{len(emb)}f', *emb)
                    conn.execute(
                        """INSERT OR REPLACE INTO embedding_index
                           (event_id, segment_id, embedding, embedding_model_id, token_count, type)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (event_id, segment_id, embedding_bytes, model_id, token_count, ev_type),
                    )
                    if self._vec_available:
                        # Replace every old mapping for this event/model in the same
                        # transaction; repeated embedding must not grow vec0.
                        old_rows = conn.execute(
                            "SELECT rowid FROM vec_items_meta WHERE event_id=? AND embedding_model_id=?",
                            (event_id, model_id),
                        ).fetchall()
                        for (rowid,) in old_rows:
                            conn.execute("DELETE FROM vec_items WHERE rowid=?", (rowid,))
                        conn.execute("DELETE FROM vec_items_meta WHERE event_id=? AND embedding_model_id=?", (event_id, model_id))
                        if self._vec_dim == len(emb):
                            cursor = conn.execute("INSERT INTO vec_items (embedding) VALUES (?)", (embedding_bytes,))
                            conn.execute(
                                "INSERT INTO vec_items_meta (rowid, event_id, embedding_model_id, segment_id) VALUES (?, ?, ?, ?)",
                                (cursor.lastrowid, event_id, model_id, segment_id),
                            )
                        # Existing databases can have a different vec0 dimension.
                        # Preserve them; canonical embeddings remain searchable exactly.
                    stored += 1
                except Exception as inner:
                    logger.warning(f"Failed to persist embedding for {event_id}: {inner}")
            conn.commit()
        except Exception as e:
            logger.error(f"Failed batch-store: {e}")
        finally:
            conn.close()
        return stored

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

    def _ensure_lexical_index(self):
        """Build a derived FTS index lazily; original events remain untouched."""
        with self._lexical_lock:
            if self._lexical_ready:
                return True
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='events_fts'").fetchone()
                if not exists:
                    conn.execute("CREATE VIRTUAL TABLE events_fts USING fts5(content, content='events', content_rowid='rowid', tokenize='unicode61')")
                    conn.execute("CREATE TRIGGER events_fts_ai AFTER INSERT ON events BEGIN INSERT INTO events_fts(rowid,content) VALUES(new.rowid,new.content); END")
                    conn.execute("CREATE TRIGGER events_fts_ad AFTER DELETE ON events BEGIN INSERT INTO events_fts(events_fts,rowid,content) VALUES('delete',old.rowid,old.content); END")
                    conn.execute("CREATE TRIGGER events_fts_au AFTER UPDATE OF content ON events BEGIN INSERT INTO events_fts(events_fts,rowid,content) VALUES('delete',old.rowid,old.content); INSERT INTO events_fts(rowid,content) VALUES(new.rowid,new.content); END")
                    conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
                conn.commit()
                self._lexical_ready = True
                return True
            except Exception as exc:
                conn.rollback()
                logger.warning("Lexical index unavailable; retaining semantic search: %s", exc)
                return False
            finally:
                conn.close()

    @staticmethod
    def _lexical_query(query_text):
        # Plain Unicode terms are quoted, never interpreted as FTS operators.
        # Porter is deliberately absent: its stemmer is English, not Russian.
        stop = set("а без в во вот все всё вы да для до его ее её если есть же за и из или их к как ко когда ли мы на над не но о об от по под при с со так то тот у уже что это я мне меня мой моя мои нашу нашем какой какая какие какую какого каким сколько напомни возвращаюсь прежде всего тогда том этом после хотел хотела просил просила".split())
        terms = list(dict.fromkeys(word for word in re.findall(r"\w+", query_text.casefold())
                                   if len(word) > 2 and word not in stop))[:40]
        return " OR ".join('"' + word + '"' for word in terms)

    @staticmethod
    def _scope_sql(session_ids, segment_id, exclude_event_ids):
        clauses, params = [], []
        if session_ids is not None:
            clauses.append("e.session_id IN (" + ",".join("?" for _ in session_ids) + ")")
            params.extend(session_ids)
        if segment_id and segment_id != "all":
            clauses.append("e.segment_id=?")
            params.append(segment_id)
        if exclude_event_ids:
            clauses.append("e.id NOT IN (" + ",".join("?" for _ in exclude_event_ids) + ")")
            params.extend(exclude_event_ids)
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def search_hybrid(self, query_text, session_ids=None, segment_id=None, top_k=5,
                      exclude_event_ids=None):
        """Return a bounded evidence pool, before reranking/public top-k.

        DEV comparison on an immutable real corpus selected semantic60 plus
        lexical60, RRF60, then a pool of40. Explicit None/0 retains unlimited
        results; the router still bounds its reranker request to40 documents.
        Lexical facts do not require a completed embedding job.
        """
        if session_ids == []:
            return []
        if isinstance(session_ids, str):
            session_ids = [session_ids]
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        excluded = list(dict.fromkeys(exclude_event_ids or []))
        candidate_limit = max(60, top_k) if top_k else None
        pool_limit = max(40, top_k) if top_k else None
        query_emb = self.get_embedding(query_text)
        semantic = []
        if query_emb:
            semantic = self._search_vec0_multi(query_emb, session_ids, segment_id,
                                                candidate_limit, excluded)
        lexical = []
        match = self._lexical_query(query_text)
        if match and self._ensure_lexical_index():
            conn = self._get_connection()
            try:
                scope, params = self._scope_sql(session_ids, segment_id, excluded)
                sql = "SELECT e.id FROM events_fts JOIN events e ON e.rowid=events_fts.rowid WHERE events_fts MATCH ?" + scope
                sql += " ORDER BY bm25(events_fts), e.id"
                params = [match] + params
                if candidate_limit:
                    sql += " LIMIT ?"
                    params.append(candidate_limit)
                lexical = [row[0] for row in conn.execute(sql, params)]
            except Exception as exc:
                logger.warning("Lexical search failed; retaining semantic candidates: %s", exc)
            finally:
                conn.close()
        by_id = {row['event_id']: row for row in semantic}
        fused = {}
        for ranked in ([row['event_id'] for row in semantic], lexical):
            for rank, event_id in enumerate(ranked, 1):
                fused[event_id] = fused.get(event_id, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(fused, key=lambda event_id: (-fused[event_id], event_id))
        if pool_limit:
            ordered = ordered[:pool_limit]
        if not ordered:
            return []
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = {}
            # Explicit unlimited search must also work on SQLite builds with
            # a small bound-variable limit; normal pools need only one batch.
            for offset in range(0, len(ordered), 500):
                chunk = ordered[offset:offset + 500]
                rows.update({row['id']: dict(row) for row in conn.execute(
                    "SELECT id,session_id,segment_id,type,content,timestamp,tool_name FROM events WHERE id IN (" + ",".join("?" for _ in chunk) + ")", chunk)})
            output = []
            for event_id in ordered:
                if event_id not in rows:
                    continue
                row = rows[event_id]
                output.append(dict(row, event_id=event_id,
                                   similarity=by_id.get(event_id, {}).get('similarity'),
                                   embedding_model_id=by_id.get(event_id, {}).get('embedding_model_id'),
                                   retrieval_score=fused[event_id]))
            return output
        finally:
            conn.close()

    def _valid_embedding(self, embedding):
        return (isinstance(embedding, (list, tuple)) and len(embedding) == self.embedding_dim
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding))

    def _search_vec0_multi(self, query_emb, session_ids, segment_id, top_k, exclude_event_ids=None):
        """Exact sqlite-vec cosine over SQL-scoped canonical vectors before top-k.

        Scalar distance has no global KNN candidate limit, so filtering cannot
        discard the requested session. Python remains the exhaustive oracle.
        """
        if not self._vec_available:
            return self._search_python_multi(query_emb, session_ids, segment_id, top_k, exclude_event_ids)
        if session_ids == [] or not self._valid_embedding(query_emb) or not any(query_emb):
            return []
        if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        conn = self._get_connection()
        try:
            sql = """SELECT e.id, i.type, i.embedding_model_id, e.segment_id, e.session_id,
                            1.0 - vec_distance_cosine(i.embedding, ?) AS similarity
                     FROM embedding_index i JOIN events e ON i.event_id=e.id
                     WHERE i.embedding_model_id=? AND length(i.embedding)=?
                       AND similarity IS NOT NULL"""
            params = [struct.pack(f'{len(query_emb)}f', *query_emb), self.model_id, 4*self.embedding_dim]
            if session_ids is not None:
                sql += " AND e.session_id IN (" + ",".join("?" for _ in session_ids) + ")"
                params.extend(session_ids)
            if segment_id and segment_id != "all":
                sql += " AND e.segment_id=?"
                params.append(segment_id)
            if exclude_event_ids:
                sql += " AND e.id NOT IN (" + ",".join("?" for _ in exclude_event_ids) + ")"
                params.extend(exclude_event_ids)
            sql += " ORDER BY similarity DESC, e.id DESC"
            if top_k:
                sql += " LIMIT ?"
                params.append(top_k)
            return [{"event_id": row[0], "type": row[1], "embedding_model_id": row[2],
                     "segment_id": row[3], "session_id": row[4], "similarity": row[5]}
                    for row in conn.execute(sql, params)]
        except Exception as exc:
            logger.warning("Scoped sqlite-vec cosine failed; using Python: %s", exc)
            return self._search_python_multi(query_emb, session_ids, segment_id, top_k, exclude_event_ids)
        finally:
            conn.close()

    def _search_python_multi(self, query_emb: List[float], session_ids: Optional[List[str]],
                             segment_id: Optional[str], top_k: Optional[int], exclude_event_ids=None) -> List[Dict[str, Any]]:
        if session_ids == [] or not self._valid_embedding(query_emb):
            return []
        if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        query_norm = math.sqrt(sum(x * x for x in query_emb))
        if not query_norm:
            return []
        conn = self._get_connection()
        try:
            sql = """SELECT e.id, i.embedding, i.type, i.embedding_model_id, e.segment_id, e.session_id
                     FROM embedding_index i JOIN events e ON i.event_id=e.id
                     WHERE i.embedding_model_id=?"""
            params = [self.model_id]
            if session_ids is not None:
                sql += " AND e.session_id IN (" + ",".join("?" for _ in session_ids) + ")"
                params.extend(session_ids)
            if segment_id and segment_id != "all":
                sql += " AND e.segment_id=?"
                params.append(segment_id)
            if exclude_event_ids:
                sql += " AND e.id NOT IN (" + ",".join("?" for _ in exclude_event_ids) + ")"
                params.extend(exclude_event_ids)

            def candidates():
                for event_id, blob, ev_type, model_id, seg_id, sess_id in conn.execute(sql, params):
                    if not blob or len(blob) != 4 * self.embedding_dim:
                        continue
                    emb = struct.unpack(f'{self.embedding_dim}f', blob)
                    if not all(math.isfinite(x) for x in emb):
                        continue
                    norm = math.sqrt(sum(x * x for x in emb))
                    if not norm:
                        continue
                    similarity = sum(a * b for a, b in zip(query_emb, emb)) / (norm * query_norm)
                    yield {"event_id": event_id, "similarity": similarity, "type": ev_type,
                           "embedding_model_id": model_id, "segment_id": seg_id, "session_id": sess_id}

            # SQL cursor streams vectors; a finite top_k stores only k candidates.
            rank = lambda item: (item['similarity'], item['event_id'])
            if top_k:
                return heapq.nlargest(top_k, candidates(), key=rank)
            return sorted(candidates(), key=rank, reverse=True)
        except Exception as exc:
            logger.error("Scoped embedding search failed: %s", exc)
            return []
        finally:
            conn.close()

    def _search_vec0(self, query_emb, session_id, segment_id=None, top_k=20):
        return self._search_vec0_multi(query_emb, [session_id], segment_id, top_k)

    def _search_python(self, query_emb, session_id, segment_id=None, top_k=20):
        return self._search_python_multi(query_emb, [session_id], segment_id, top_k)
