import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_LOG_PATH = os.path.join(PLUGIN_DIR, "trace.jsonl")
MAX_TRACE_MB = 10
MAX_ROTATIONS = 3

class Observability:
    """Handles tracing and metrics for the context engine."""

    def __init__(self, trace_path: str = TRACE_LOG_PATH):
        self.trace_path = trace_path
        self.metrics = {
            "context_hit_rate": 0.0,
            "graph_dependency_usage": 0.0,
            "fallback_rate": 0.0,
            "segmentation_count": 0
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)
        logger.info(f"Observability initialized. Trace: {self.trace_path}")

    def log_turn(self, session_id: str, segment_id: str,
                 intent: str, signals: Dict,
                 query: str, candidates_retrieved: int, candidates_selected: int,
                 budget: Dict, top_chunks: List[Dict],
                 fallback: Optional[str] = None, segmenter_signal: str = "none",
                 delta_extracted: List[Dict] = None,
                 mode: str = "general"):
        """Writes a single JSON object to the trace log."""

        self.rotate_log_if_needed()

        # Normalize signals: accept dataclass or dict
        if is_dataclass(signals):
            signals = asdict(signals)

        turn_data = {
            "turn_id": self._generate_id(),
            "session_id": session_id,
            "segment_id": segment_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "intent": intent,
            "mode": mode,
            "classifier_signals": signals,
            "query_built_from": query,
            "candidates_retrieved": candidates_retrieved,
            "candidates_selected": candidates_selected,
            "budget": budget,
            "top_chunks": top_chunks[:5], # Limit to top 5 for brevity
            "fallback_triggered": fallback,
            "segmenter_signal": segmenter_signal,
            "delta_extracted": delta_extracted or []
        }

        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                json.dump(turn_data, f, ensure_ascii=False)
                f.write("\n")
            logger.debug(f"Trace logged for turn {turn_data['turn_id']}")
        except Exception as e:
            logger.error(f"Failed to write trace log: {e}")

        # Update metrics (simplified)
        if candidates_selected > 0:
            # hit rate
            self.metrics["context_hit_rate"] = (self.metrics["context_hit_rate"] + 1.0) / 2.0 # Running avg
        if fallback:
            self.metrics["fallback_rate"] = (self.metrics["fallback_rate"] + 1.0) / 2.0

        # graph_dependency_usage: ratio of selected chunks whose dependency
        # bonus was non-zero this turn. Tells us how often Stage 7 propagation
        # actually contributes to ranking. Computed off the trimmed top_chunks
        # list because that's what's persisted; full list isn't passed in.
        if candidates_selected > 0 and top_chunks:
            with_dep = sum(
                1 for c in top_chunks
                if (c.get("score_breakdown") or {}).get("dependency", 0) > 0
            )
            ratio = with_dep / max(len(top_chunks), 1)
            self.metrics["graph_dependency_usage"] = (
                self.metrics["graph_dependency_usage"] + ratio
            ) / 2.0

        # Surface metrics into agent.log so they're visible without a tool call.
        logger.info(
            "metrics: hit=%.2f dep_usage=%.2f fallback=%.2f segments=%d",
            self.metrics["context_hit_rate"],
            self.metrics["graph_dependency_usage"],
            self.metrics["fallback_rate"],
            self.metrics["segmentation_count"],
        )

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def rotate_log_if_needed(self):
        """Check file size and rotate if > MAX_TRACE_MB."""
        try:
            size_mb = os.path.getsize(self.trace_path) / (1024 * 1024)
            if size_mb > MAX_TRACE_MB:
                self._rotate_log()
        except FileNotFoundError:
            pass

    def _rotate_log(self):
        """Rotate trace log: rename to .1, .2, .3."""
        for i in range(MAX_ROTATIONS, 0, -1):
            old = f"{self.trace_path}.{i}"
            new = f"{self.trace_path}.{i+1}" if i < MAX_ROTATIONS else None
            if os.path.exists(old):
                if new:
                    os.replace(old, new)
                else:
                    os.remove(old) # delete oldest
        os.replace(self.trace_path, f"{self.trace_path}.1")
        logger.info("Trace log rotated.")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
