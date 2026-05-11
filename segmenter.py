import re
import math
import struct
import logging
import threading
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.35
DEFAULT_WEIGHTS = [0.4, 0.3, 0.3]   # embedding / entity / explicit
COLD_START_MIN_EVENTS = 3           # drift = 0 until segment has at least this many embeddings.
SEGMENT_LARGE_WARN = 300            # Log a warning past this many events — sign of broken drift detection.
DEFAULT_CENTROID_CACHE = 100
DEFAULT_CENTROID_WINDOW = 200       # 0 = no window, average all segment events


class _CentroidCache:
    """Thread-safe LRU cache of (segment_id -> (centroid_vector, fingerprint)).

    Invalidation is by fingerprint — caller passes (event_count, window_size);
    a change in either invalidates the entry.
    """

    def __init__(self, max_entries: int = DEFAULT_CENTROID_CACHE):
        self._data: "OrderedDict[str, Tuple[List[float], Tuple[int, int]]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    def resize(self, max_entries: int):
        with self._lock:
            self._max = max(int(max_entries), 1)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, segment_id: str, fingerprint: Tuple[int, int]) -> Optional[List[float]]:
        with self._lock:
            entry = self._data.get(segment_id)
            if entry is None:
                return None
            vec, cached_fp = entry
            if cached_fp != fingerprint:
                return None
            self._data.move_to_end(segment_id)
            return vec

    def put(self, segment_id: str, vector: List[float], fingerprint: Tuple[int, int]):
        with self._lock:
            self._data[segment_id] = (vector, fingerprint)
            self._data.move_to_end(segment_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def invalidate(self, segment_id: str):
        with self._lock:
            self._data.pop(segment_id, None)


_centroid_cache = _CentroidCache()


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
        self.config = config
        cache_size = self._cfg("centroid_cache_size", DEFAULT_CENTROID_CACHE)
        try:
            _centroid_cache.resize(int(cache_size))
        except (TypeError, ValueError):
            pass
        self._warned_segments: set = set()

    def _cfg(self, key: str, default):
        if self.config is None:
            return default
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    # ------------------------------------------------------------------
    # Centroid + drift
    # ------------------------------------------------------------------

    def _load_segment_embeddings(self, segment_id: str, window: int = 0) -> List[List[float]]:
        """Pull embeddings for a segment from embedding_index.

        window > 0 → only the last N embeddings (by events.timestamp DESC, then reversed).
        window = 0 → all embeddings in segment.
        """
        try:
            conn = self.store._get_connection()
            if window and window > 0:
                cursor = conn.execute(
                    "SELECT i.embedding FROM embedding_index i "
                    "JOIN events e ON i.event_id = e.id "
                    "WHERE e.segment_id = ? AND i.embedding IS NOT NULL "
                    "ORDER BY e.timestamp DESC LIMIT ?",
                    (segment_id, window)
                )
            else:
                cursor = conn.execute(
                    "SELECT i.embedding FROM embedding_index i "
                    "JOIN events e ON i.event_id = e.id "
                    "WHERE e.segment_id = ? AND i.embedding IS NOT NULL",
                    (segment_id,)
                )
            embeddings: List[List[float]] = []
            for row in cursor:
                blob = row[0]
                if not blob:
                    continue
                emb_len = len(blob) // 4
                embeddings.append(list(struct.unpack(f"{emb_len}f", blob)))
            conn.close()
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load segment embeddings for {segment_id}: {e}")
            return []

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

        # Total event count is part of cache fingerprint so a new event
        # invalidates the cache even when the window stays full.
        total = self._segment_event_count(segment_id)
        if total >= SEGMENT_LARGE_WARN and segment_id not in self._warned_segments:
            logger.warning(
                f"Segment {segment_id} has {total} events — drift detection may be tuned wrong "
                f"(check drift_threshold). centroid_window={window} keeps centroid focused."
            )
            self._warned_segments.add(segment_id)

        embeddings = self._load_segment_embeddings(segment_id, window=window)
        n = len(embeddings)
        if n < COLD_START_MIN_EVENTS:
            return None

        fingerprint = (total, window)
        cached = _centroid_cache.get(segment_id, fingerprint)
        if cached is not None:
            return cached

        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v
        centroid = [v / n for v in centroid]
        _centroid_cache.put(segment_id, centroid, fingerprint)
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
        if not msg_emb:
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
                "SELECT COUNT(*) FROM events WHERE segment_id = ?", (segment_id,)
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
            r"instead of that", r"давай переключимся", r"забудь это"
        ]
        for p in switch_patterns:
            if re.search(p, message, re.IGNORECASE):
                logger.info("Hard trigger: explicit switch phrase detected.")
                return True
        return False

    def calculate_drift_score(self, message: str, segment_id: str,
                               weights: List[float] = None,
                               entity_signal: float = 0.0,
                               explicit_signal: float = 0.0) -> float:
        """Composite drift = w0*embedding_drift + w1*entity + w2*explicit.

        entity_signal and explicit_signal are 0/1 floats from classifier;
        when not supplied, only embedding drift contributes.
        """
        w = weights or DEFAULT_WEIGHTS
        emb_drift = self.calculate_embedding_drift(message, segment_id)
        score = w[0] * emb_drift + w[1] * entity_signal + w[2] * explicit_signal
        return min(max(score, 0.0), 1.0)

    def create_new_segment(self, session_id: str) -> str:
        old = self.current_segment_id
        self.segment_count += 1
        new_segment_id = f"seg_{session_id}_{self.segment_count}"
        self.current_segment_id = new_segment_id
        _centroid_cache.invalidate(old)
        logger.info(f"New segment created: {new_segment_id} (previous: {old})")
        return new_segment_id

    def handle_new_message(self, session_id: str, message: str,
                           execution_state: Dict[str, Any]) -> str:
        if self.check_hard_triggers(message, execution_state.get("last_tool")):
            return self.create_new_segment(session_id)

        drift_score = self.calculate_drift_score(message, self.current_segment_id)
        if drift_score > DRIFT_THRESHOLD:
            logger.info(f"Drift trigger: score {drift_score:.2f} > {DRIFT_THRESHOLD} → new segment.")
            return self.create_new_segment(session_id)

        return self.current_segment_id
