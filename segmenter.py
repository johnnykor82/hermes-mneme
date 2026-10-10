import re
import math
import struct
import logging
import threading
from collections import OrderedDict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.35
DEFAULT_WEIGHTS = [0.4, 0.3, 0.3]   # embedding / entity / explicit
COLD_START_MIN_EVENTS = 3           # drift = 0 until segment has at least this many embeddings.
SEGMENT_LARGE_WARN = 300            # Log a warning past this many events — sign of broken drift detection.
DEFAULT_CENTROID_CACHE = 100
DEFAULT_CENTROID_WINDOW = 200       # 0 = no window, average all segment events


class _CentroidCache:
    """Thread-safe scoped centroid LRU, invalidated by SQLite commit revision."""

    def __init__(self, max_entries: int = DEFAULT_CENTROID_CACHE):
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    def resize(self, max_entries: int):
        with self._lock:
            self._max = max(int(max_entries), 1)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, segment_id, fingerprint: tuple) -> Optional[List[float]]:
        with self._lock:
            entry = self._data.get(segment_id)
            if entry is None:
                return None
            vec, cached_fp = entry
            if cached_fp != fingerprint:
                return None
            self._data.move_to_end(segment_id)
            return vec

    def put(self, segment_id, vector: List[float], fingerprint: tuple):
        with self._lock:
            self._data[segment_id] = (vector, fingerprint)
            self._data.move_to_end(segment_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def invalidate(self, segment_id: str):
        with self._lock:
            self._data.pop(segment_id, None)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SessionSegmenter:
    """Detects topic/task switches within a session."""

    def __init__(self, store, indexer, current_segment_id: str, config=None):
        self.store = store
        self.indexer = indexer
        self.current_segment_id = current_segment_id
        self.segment_count = 0
        self.session_id = None
        self._centroid_local = threading.local()
        self.config = config
        self._warned_segments: set = set()

    def _cfg(self, key: str, default):
        if self.config is None:
            return default
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _model_id(self):
        return getattr(self.indexer, "model_id", None) or self._cfg(
            "embedding_model", "jina-embeddings-v5-text-small-retrieval-mlx")

    def bind_session(self, session_id: Optional[str], segment_id: str):
        """Restore binding and the highest allocated counter, including legacy resumes."""
        if self.session_id == session_id and self.current_segment_id == segment_id:
            return
        self.session_id = session_id
        self.current_segment_id = segment_id
        if session_id is None:
            self.segment_count = 0
            return
        pattern = re.compile(r"^seg_" + re.escape(session_id) + r"_(\d+)$")
        counter = 0
        conn = self.store._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS segment_sequences "
                         "(session_id TEXT PRIMARY KEY, counter INTEGER NOT NULL)")
            segments = [segment_id] + [r[0] for r in conn.execute(
                "SELECT DISTINCT segment_id FROM events WHERE session_id=?", (session_id,))]
            for candidate in segments:
                match = pattern.match(candidate or "")
                if match:
                    counter = max(counter, int(match.group(1)))
            conn.execute("INSERT INTO segment_sequences VALUES (?,?) "
                         "ON CONFLICT(session_id) DO UPDATE SET counter=MAX(counter,excluded.counter)",
                         (session_id, counter))
            self.segment_count = conn.execute(
                "SELECT counter FROM segment_sequences WHERE session_id=?", (session_id,)).fetchone()[0]
            conn.commit()
        finally:
            conn.close()

    def _centroid_context(self):
        # data_version is meaningful only on a persistent connection. A per-thread
        # connection observes commits from all other connections without reading BLOBs.
        local = self._centroid_local
        if not hasattr(local, "connection"):
            local.connection = self.store._get_connection()
            local.cache = _CentroidCache()
        return local.connection, local.cache

    def close(self):
        """Release the current thread's centroid reader (no transaction is retained)."""
        local = self._centroid_local
        if hasattr(local, "connection"):
            local.connection.close()
            del local.connection
            del local.cache

    # ------------------------------------------------------------------
    # Centroid + drift
    # ------------------------------------------------------------------

    def _load_segment_embeddings(self, segment_id: str, window: int = 0) -> List[List[float]]:
        if not self.session_id:
            return []
        conn = self.store._get_connection()
        try:
            sql = ("SELECT i.embedding FROM embedding_index i JOIN events e ON i.event_id=e.id "
                   "WHERE e.session_id=? AND e.segment_id=? AND i.embedding_model_id=? "
                   "AND i.embedding IS NOT NULL ORDER BY e.timestamp DESC, e.id DESC")
            params = [self.session_id, segment_id, self._model_id()]
            if window > 0:
                sql += " LIMIT ?"
                params.append(window)
            embeddings = []
            expected_dim = getattr(self.indexer, "embedding_dim", None)
            for row in conn.execute(sql, params):
                blob = row[0]
                if not blob or len(blob) % 4:
                    continue
                dim = len(blob) // 4
                if expected_dim is None:
                    expected_dim = dim
                if dim != expected_dim:
                    logger.warning("Ignoring centroid vector with incompatible dimension: %s", dim)
                    continue
                embeddings.append(list(struct.unpack(f"{dim}f", blob)))
            return embeddings
        except Exception as exc:
            logger.warning("Failed to load scoped segment embeddings: %s", exc)
            return []
        finally:
            conn.close()

    def get_centroid(self, segment_id: str) -> Optional[List[float]]:
        """Return cached or freshly-computed centroid for a segment.

        Sliding window: when ``centroid_window > 0``, average only the most
        recent N embeddings — segment_id stays stable, retrieval still sees
        the WHOLE topic, but drift-detection follows the active part.

        Returns None when the segment has fewer than COLD_START_MIN_EVENTS
        embeddings (cold start).
        """
        window = self._cfg("centroid_window", DEFAULT_CENTROID_WINDOW)
        try:
            window = int(window or 0)
        except (TypeError, ValueError):
            window = 0
        if window < 0:
            window = 0

        if not self.session_id:
            return None
        conn, cache = self._centroid_context()
        cache.resize(self._cfg("centroid_cache_size", DEFAULT_CENTROID_CACHE))
        key = (self.session_id, segment_id, self._model_id(),
               getattr(self.indexer, "embedding_dim", None))
        fingerprint = (conn.execute("PRAGMA data_version").fetchone()[0],
                       conn.total_changes, window)
        cached = cache.get(key, fingerprint)
        if cached is not None:
            return cached
        total = self._segment_event_count(segment_id)
        if total >= SEGMENT_LARGE_WARN and key not in self._warned_segments:
            logger.warning("Segment %s/%s has %s events; centroid_window=%s",
                           self.session_id, segment_id, total, window)
            self._warned_segments.add(key)
        embeddings = self._load_segment_embeddings(segment_id, window=window)
        n = len(embeddings)
        if n < COLD_START_MIN_EVENTS:
            return None

        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v
        centroid = [v / n for v in centroid]
        cache.put(key, centroid, fingerprint)
        return centroid

    def calculate_embedding_drift(self, message: str, segment_id: str) -> float:
        """Cosine distance between the message embedding and the segment centroid.

        Returns 0.0 on cold start, missing embedding, or any failure — drift is
        a soft signal, errors must not destabilize the classifier.
        """
        centroid = self.get_centroid(segment_id)
        if centroid is None:
            return 0.0
        try:
            msg_emb = self.indexer.get_embedding(message)
        except Exception as e:
            logger.warning(f"Drift: failed to embed message: {e}")
            return 0.0
        if not msg_emb or len(msg_emb) != len(centroid):
            return 0.0
        similarity = _cosine(msg_emb, centroid)
        # cosine ∈ [-1, 1]; map to drift ∈ [0, 1] (higher = more different)
        drift = (1.0 - similarity) / 2.0
        if drift < 0.0:
            drift = 0.0
        elif drift > 1.0:
            drift = 1.0
        return drift

    def _segment_event_count(self, segment_id: str) -> int:
        try:
            conn = self.store._get_connection()
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id=? AND segment_id=?",
                (self.session_id, segment_id)
            ).fetchone()
            conn.close()
            return int(row[0] or 0)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def check_hard_triggers(self, message: str, last_tool: Optional[str] = None) -> bool:
        """Synchronous triggers that force a segment cut immediately.

        Only explicit user phrases trigger here. We deliberately do NOT cut
        on segment size — large segments on one topic are legitimate, and the
        sliding-window centroid keeps drift detection sharp without losing
        retrieval coverage.
        """
        switch_patterns = [
            r"let's switch to", r"forget that", r"new topic",
            r"instead of that", r"давай переключимся", r"забудь это", r"новая тема"
        ]
        for p in switch_patterns:
            if re.search(p, message, re.IGNORECASE):
                logger.info("Hard trigger: explicit switch phrase detected.")
                return True
        return False

    def calculate_drift_score(self, message: str, segment_id: str,
                               weights: List[float] = None,
                               entity_signal: Optional[float] = None,
                               explicit_signal: Optional[float] = None) -> float:
        """Composite drift = w0*embedding_drift + w1*entity + w2*explicit.

        entity_signal and explicit_signal are optional normalized classifier signals;
        when neither is supplied, use embedding drift directly (no absent-signal weight penalty).
        """
        w = weights if weights is not None else self._cfg("drift_weights", DEFAULT_WEIGHTS)
        emb_drift = self.calculate_embedding_drift(message, segment_id)
        if entity_signal is None and explicit_signal is None:
            return emb_drift
        score = w[0] * emb_drift + w[1] * (entity_signal or 0.) + w[2] * (explicit_signal or 0.)
        return min(max(score, 0.0), 1.0)

    def create_new_segment(self, session_id: str) -> str:
        old = self.current_segment_id
        if self.session_id != session_id:
            self.bind_session(session_id, f"seg_{session_id}_1")
        conn = self.store._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE segment_sequences SET counter=counter+1 WHERE session_id=?", (session_id,))
            self.segment_count = conn.execute(
                "SELECT counter FROM segment_sequences WHERE session_id=?", (session_id,)).fetchone()[0]
            conn.commit()
        finally:
            conn.close()
        new_segment_id = f"seg_{session_id}_{self.segment_count}"
        self.current_segment_id = new_segment_id
        logger.info(f"New segment created: {new_segment_id} (previous: {old})")
        return new_segment_id

    def handle_new_message(self, session_id: str, message: str,
                           execution_state: Dict[str, Any]) -> str:
        if self.session_id != session_id:
            initial = self.current_segment_id if self.session_id is None and self.current_segment_id.startswith(
                f"seg_{session_id}_") else f"seg_{session_id}_1"
            self.bind_session(session_id, initial)
        if not self._cfg("segmentation_enabled", True):
            return self.current_segment_id
        if self.check_hard_triggers(message, execution_state.get("last_tool")):
            return self.create_new_segment(session_id)

        drift_score = self.calculate_drift_score(message, self.current_segment_id)
        threshold = self._cfg("drift_threshold", DRIFT_THRESHOLD)
        if drift_score > threshold:
            logger.info(f"Drift trigger: score {drift_score:.2f} > {threshold} → new segment.")
            return self.create_new_segment(session_id)

        return self.current_segment_id
