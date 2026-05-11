import json
import logging
import math
import re
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from . import classifier
from . import index
from . import store
from . import graph

logger = logging.getLogger(__name__)

# Strip our own [RETRIEVED CONTEXT] block when it leaks back as a "user message"
# from the previous turn — otherwise we keep searching for what we just retrieved.
_RETRIEVED_PREFIX_RE = re.compile(r"^\s*\[RETRIEVED CONTEXT\][\s\S]*?(?:\n\n|\Z)")

# Budget defaults
DEFAULT_TOTAL_TOKENS = 16000
DEFAULT_PROTECTED_TAIL = 6
DEFAULT_STATE_BUDGET_RATIO = 0.05
DEFAULT_RETRIEVED_BUDGET_RATIO = 0.45

# Mode-dependent weights. Stage 7 enables non-zero dependency weights now that
# execution-graph propagation is implemented (see _calculate_dependency_score).
# CLARIFICATION/debugging benefit most from graph context — those queries are
# almost always about the immediately-prior causal chain.
MODE_WEIGHTS = {
    'general':       {'similarity': 0.6,  'recency': 0.2,  'dependency': 0.1,  'type': 0.1},
    'reasoning':     {'similarity': 0.7,  'recency': 0.15, 'dependency': 0.05, 'type': 0.1},
    'factual':       {'similarity': 0.55, 'recency': 0.3,  'dependency': 0.05, 'type': 0.1},
    'debugging':     {'similarity': 0.35, 'recency': 0.35, 'dependency': 0.2,  'type': 0.1},
    'clarification': {'similarity': 0.4,  'recency': 0.35, 'dependency': 0.15, 'type': 0.1},
}

# Type weights
TYPE_WEIGHTS = {
    'decision': 1.0,
    'user_message': 0.8,
    'tool_output': 0.6,
    'assistant_message': 0.5,
}

@dataclass
class RouterInput:
    message: str
    execution_state: Dict[str, Any]
    segment_id: str
    recent_events: List[Dict[str, Any]] # Raw events from DB
    token_budget: int
    session_id: str

@dataclass
class RetrievalCandidate:
    event_id: str
    score: float
    content: str
    type: str
    timestamp: str
    embedding_model_id: str = ""
    segment_id: str = ""
    dependency_bonus: float = 0.0

@dataclass
class RouterResult:
    intent: str
    signals: Any # ClassifierSignals
    query: str
    candidates_retrieved: int
    candidates_selected: List[RetrievalCandidate]
    mode: str = "general"
    budget: Dict[str, Any] = field(default_factory=dict)

class ContextRouter:
    """Assembles context package for the next LLM call."""

    def __init__(self, store: store.ContextStore, indexer: index.EmbeddingIndex,
                 config: Optional[Any] = None, segmenter=None):
        self.store = store
        self.indexer = indexer
        self.graph = graph.ExecutionGraph(store.db_path)
        self.config = config
        self.segmenter = segmenter
        logger.info("ContextRouter initialized")

    def _cfg(self, key: str, default=None):
        if self.config is None:
            return default
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def run(self, router_input: RouterInput) -> RouterResult:
        """
        Main routing logic:
        1. Classify intent
        2. Build query
        3. Retrieve candidates
        4. Score and rank
        Returns RouterResult containing intent, signals, query, and candidates.
        """

        # 1. Intent Classification
        signals = self._get_signals(router_input)
        intent = classifier.resolve_intent(signals)
        # Intent overrides the heuristic mode picker — a CLARIFICATION question
        # always wants the clarification weights regardless of keyword matches.
        if intent == classifier.INTENT_CLARIFICATION:
            mode = "clarification"
        else:
            mode = self._get_retrieval_mode(router_input)

        logger.info(f"Intent classified: {intent}, mode={mode}")

        # 2. Build Query
        query = self._build_query(router_input, intent)
        if not query:
            logger.warning("Empty query, skipping retrieval.")
            return RouterResult(
                intent=intent, signals=signals, query="",
                candidates_retrieved=0, candidates_selected=[], mode=mode
            )

        # 3. Retrieve Candidates
        # router_top_k: 0 / missing → unlimited (return everything semantically close).
        top_k_cfg = self._cfg("router_top_k", 0)
        try:
            top_k_int = int(top_k_cfg) if top_k_cfg else 0
        except (TypeError, ValueError):
            top_k_int = 0
        top_k_arg = top_k_int if top_k_int > 0 else None
        raw_candidates = self.indexer.search(
            query, router_input.session_id, router_input.segment_id, top_k=top_k_arg
        )

        # Cross-segment fallback: when the current segment is too small (cold
        # start, RESUME-shred, drift-happy session) the per-segment KNN returns
        # almost the same events that already sit in protected_tail. Widen to
        # the whole session so retrieval can pull in older topics.
        min_candidates = int(self._cfg("router_min_candidates", 12) or 0)
        if min_candidates > 0 and len(raw_candidates) < min_candidates:
            raw_all = self.indexer.search(
                query, router_input.session_id, "all", top_k=top_k_arg
            )
            if raw_all:
                seen = {r.get("event_id") for r in raw_candidates}
                added = [r for r in raw_all if r.get("event_id") not in seen]
                if added:
                    raw_candidates = list(raw_candidates) + added
                    logger.info(
                        f"Cross-segment fallback: +{len(added)} candidates "
                        f"(total {len(raw_candidates)}, min_required={min_candidates})"
                    )

        if not raw_candidates:
            logger.info("No candidates found.")
            return RouterResult(
                intent=intent, signals=signals, query=query,
                candidates_retrieved=0, candidates_selected=[], mode=mode
            )

        # 4. Score and Rank
        dep_bonuses = self._build_dependency_bonuses(router_input)
        scored = self._score_candidates(
            raw_candidates, router_input, mode=mode, dep_bonuses=dep_bonuses
        )

        # 5. Optional reranker (second-stage)
        if self._cfg("reranker_enabled", False) and scored:
            scored = self._rerank(query, scored)

        return RouterResult(
            intent=intent,
            signals=signals,
            query=query,
            candidates_retrieved=len(raw_candidates),
            candidates_selected=scored,
            mode=mode,
        )

    def _get_signals(self, inp: RouterInput) -> classifier.ClassifierSignals:
        """Extract classifier signals from input and state."""
        state = inp.execution_state
        msg = inp.message

        signals = classifier.ClassifierSignals()
        signals.explicit_switch = classifier.classify_explicit_switch(msg)
        signals.entity_contradiction = classifier.classify_entity_contradiction(
            msg, state.get("active_entities", [])
        )
        # Use segmenter drift score if available
        signals.embedding_drift = self._get_embedding_drift_score(inp)
        # Per spec, question_about_output checks entities from the LAST assistant
        # turn specifically — not the running aggregate. Pull them out of
        # recent_events on the fly (cheap regex extraction, no LLM).
        last_assistant_entities = self._extract_last_assistant_entities(inp.recent_events or [])
        signals.question_about_output = classifier.classify_question_about_output(
            msg, last_assistant_entities
        )
        return signals

    def _extract_last_assistant_entities(self, recent_events: List[Dict[str, Any]]) -> List[str]:
        """Find the most recent assistant_message in recent_events and pull
        deterministic entities out of its content. Returns [] if there is no
        such event yet (cold start)."""
        for ev in reversed(recent_events):
            if ev.get("type") != "assistant_message":
                continue
            content = ev.get("content") or ""
            if not content:
                continue
            return classifier.extract_entities(content)
        return []

    def _get_embedding_drift_score(self, inp: RouterInput) -> float:
        """Cosine drift between message and current segment centroid.
        Falls back to 0.0 on cold start or when segmenter is unavailable."""
        if self.segmenter is None:
            return 0.0
        try:
            return self.segmenter.calculate_embedding_drift(inp.message or "", inp.segment_id)
        except Exception as e:
            logger.warning(f"Drift score failed: {e}")
            return 0.0

    def _build_query(self, inp: RouterInput, intent: str) -> str:
        """Construct the retrieval query string."""
        state = inp.execution_state
        # Strip our own [RETRIEVED CONTEXT] block — without this, the message
        # we just retrieved gets fed back as the next query and the search
        # collapses into "find what's similar to my last retrieval".
        msg_clean = _RETRIEVED_PREFIX_RE.sub("", inp.message or "").strip()

        # Always include message if available — even for CONTINUATION.
        # The spec says "don't use message for query" but that assumes
        # goal/current_step are populated. In practice, without message
        # the query is often empty and retrieval never fires.
        query_parts = [
            state.get("goal", ""),
            state.get("current_step", ""),
        ]
        if intent != classifier.INTENT_CONTINUATION:
            query_parts.append(state.get("last_tool_output_summary", ""))
        # CLARIFICATION: pull entities from the last assistant turn into the
        # query. The user is asking about *that* output, so the retrieval
        # query needs to anchor on what was said, not just on the current
        # message ("why?", "how come?" alone give no signal).
        if intent == classifier.INTENT_CLARIFICATION:
            last_assistant_entities = self._extract_last_assistant_entities(
                inp.recent_events or []
            )
            if last_assistant_entities:
                query_parts.append(" ".join(last_assistant_entities[:10]))
        if msg_clean:
            query_parts.append(msg_clean)

        query = " ".join(filter(None, query_parts)).strip()
        if not query:
            # Fallback: pull last user message from recent_events so retrieval
            # still fires even when execution state is empty (fresh session,
            # post-/reset, etc.). Skip events whose content is our own
            # [RETRIEVED CONTEXT] envelope.
            for ev in reversed(inp.recent_events or []):
                if ev.get("role") != "user":
                    continue
                content = str(ev.get("content") or "")
                content = _RETRIEVED_PREFIX_RE.sub("", content).strip()
                if content:
                    query = content
                    logger.info("Using fallback query from recent user event.")
                    break
        if not query:
            logger.warning("Empty query built from state+message. Retrieval will be skipped.")
        return query

    def _score_candidates(self, candidates: List[Dict], inp: RouterInput,
                          mode: str = "general",
                          dep_bonuses: Optional[Dict[str, float]] = None) -> List[RetrievalCandidate]:
        """Apply scoring formula and rank."""
        scored_list = []
        weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS['general'])

        for c in candidates:
            event_id = c.get("event_id")
            similarity = c.get("similarity", 0.0)
            embedding_model_id = c.get("embedding_model_id", "")
            segment_id = c.get("segment_id", "")
            type_ = c.get("type", "")

            # Fetch event details from store for scoring
            try:
                conn = self.store._get_connection()
                row = conn.execute(
                    "SELECT content, type, timestamp FROM events WHERE id = ?",
                    (event_id,)
                ).fetchone()
                conn.close()
                if not row:
                    continue

                content = row['content']
                event_type = row['type']
                timestamp = row['timestamp']

                # Calculate recency score
                recency = self._calculate_recency(timestamp)

                # Calculate dependency score
                dependency_score = self._calculate_dependency_score(event_id, inp, dep_bonuses)

                # Get type weight
                type_weight = TYPE_WEIGHTS.get(event_type, 0.5)

                # Calculate final score
                final_score = (
                    weights['similarity'] * similarity +
                    weights['recency'] * recency +
                    weights['dependency'] * dependency_score +
                    weights['type'] * type_weight
                )

                rc = RetrievalCandidate(
                    event_id=event_id,
                    score=final_score,
                    content=content,
                    type=event_type,
                    timestamp=timestamp,
                    embedding_model_id=embedding_model_id,
                    segment_id=segment_id,
                    dependency_bonus=dependency_score,
                )
                scored_list.append(rc)

            except Exception as e:
                logger.warning(f"Failed to fetch details for {event_id}: {e}")

        # Sort by score
        scored_list.sort(key=lambda x: x.score, reverse=True)
        return scored_list

    def _get_retrieval_mode(self, inp: RouterInput) -> str:
        """Pick retrieval mode based on message + last_tool signals.

        Modes shape MODE_WEIGHTS — debugging biases toward recency,
        factual/reasoning biases toward similarity, general is balanced.
        """
        msg = (inp.message or "").lower()
        state = inp.execution_state or {}
        last_tool = (state.get("last_tool") or "").lower()

        debug_kw = (
            "error", "fail", "failed", "traceback", "exception", "stacktrace",
            "debug", "broken", "crash", "bug",
            "ошибк", "не работает", "падает", "сломал", "не запускается", "краш",
        )
        reasoning_kw = (
            "почему", "зачем", "как лучше", "что если", "стоит ли", "какой подход",
            "why", "how should", "what if", "should i", "best way", "tradeoff",
        )
        factual_kw = (
            "what is", "when ", "where ", "who ", "list ", "show me",
            "что такое", "когда", "где находится", "перечисли", "покажи",
        )

        debug_tools = {"read", "bash", "grep", "glob"}

        if any(k in msg for k in debug_kw) or last_tool in debug_tools and any(k in msg for k in ("?", "wrong", "не так")):
            return "debugging"
        if any(k in msg for k in reasoning_kw) or len(msg) > 500:
            return "reasoning"
        if any(k in msg for k in factual_kw):
            return "factual"
        return "general"

    def _calculate_recency(self, timestamp: str) -> float:
        """Calculate recency score using exponential decay."""
        try:
            import datetime
            event_time = datetime.datetime.fromisoformat(timestamp)
            now = datetime.datetime.now(datetime.timezone.utc)
            minutes_since = (now - event_time).total_seconds() / 60
            lambda_val = 0.01  # Decay rate
            return math.exp(-lambda_val * minutes_since)
        except Exception:
            return 0.0

    def _build_dependency_bonuses(self, inp: RouterInput) -> Dict[str, float]:
        """Walk the execution graph backwards from anchor events and assign a
        bonus per visited event_id: bonus = decay ** depth.

        Anchor events are the most recent tool_output and assistant_message
        in recent_events — the points the user is most likely "asking about".
        Returns {} when propagation is disabled (max_depth=0) or graph is empty.
        """
        max_depth = int(self._cfg("dependency_max_depth", 4) or 0)
        if max_depth <= 0:
            return {}
        try:
            decay = float(self._cfg("dependency_decay", 0.6))
        except (TypeError, ValueError):
            decay = 0.6
        if decay <= 0:
            return {}

        anchors = self._collect_dependency_anchors(inp.recent_events or [])
        if not anchors:
            return {}

        bonuses: Dict[str, float] = {}
        # BFS so the shortest-path depth wins when multiple anchors hit the
        # same event. Manual queue instead of recursion to avoid the per-call
        # `visited` recursion semantics in graph.get_neighbors.
        from collections import deque
        queue = deque((aid, 0) for aid in anchors)
        seen = set(anchors)
        # Anchors themselves get full bonus (depth 0).
        for aid in anchors:
            bonuses[aid] = 1.0

        while queue:
            cur_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            try:
                neighbors = self.graph.get_neighbors(cur_id, depth=1)
            except Exception as e:
                logger.warning(f"Dependency traversal failed at {cur_id}: {e}")
                continue
            next_depth = depth + 1
            score = decay ** next_depth
            for n in neighbors:
                nid = n.get("event_id")
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                # First time we see this id is the shortest path — keep score.
                if nid not in bonuses or bonuses[nid] < score:
                    bonuses[nid] = score
                queue.append((nid, next_depth))

        return bonuses

    def _collect_dependency_anchors(self, recent_events: List[Dict[str, Any]]) -> List[str]:
        """Pick anchor event_ids for graph traversal: the most recent
        tool_output and the most recent assistant_message. Both are likely
        targets of the user's next question."""
        anchors: List[str] = []
        seen_types = set()
        for ev in reversed(recent_events):
            ev_type = ev.get("type")
            if ev_type not in ("tool_output", "assistant_message"):
                continue
            if ev_type in seen_types:
                continue
            ev_id = ev.get("event_id") or ev.get("id")
            if not ev_id:
                continue
            anchors.append(ev_id)
            seen_types.add(ev_type)
            if len(seen_types) == 2:
                break
        return anchors

    def _calculate_dependency_score(self, event_id: str, inp: RouterInput,
                                    dep_bonuses: Optional[Dict[str, float]] = None) -> float:
        """Lookup precomputed bonus for this event_id. 0.0 means the event is
        not on any causal path within max_depth from the current anchors."""
        if not dep_bonuses:
            return 0.0
        return float(dep_bonuses.get(event_id, 0.0))

    def _rerank(self, query: str, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Optional second-stage reranker.

        Calls a Jina/BGE/Cohere-compatible rerank endpoint and resorts candidates
        by the returned scores. On any failure we keep the original ordering —
        rerank is a quality boost, not a correctness requirement.
        """
        endpoint = self._cfg("reranker_endpoint")
        if not endpoint or not candidates:
            return candidates
        try:
            documents = [c.content[:2000] for c in candidates]
            payload = {
                "model": self._cfg("reranker_model") or "rerank",
                "query": query,
                "documents": documents,
            }
            headers = {"Content-Type": "application/json"}
            api_key = self._cfg("reranker_api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            r = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            # Accept multiple response shapes:
            #   {results: [{index, relevance_score}, ...]}      (Cohere/BGE)
            #   {scores: [...]} aligned with documents          (Jina-style)
            new_scores = [None] * len(candidates)
            if isinstance(data.get("results"), list):
                for item in data["results"]:
                    idx = item.get("index")
                    score = item.get("relevance_score", item.get("score"))
                    if isinstance(idx, int) and 0 <= idx < len(new_scores) and score is not None:
                        new_scores[idx] = float(score)
            elif isinstance(data.get("scores"), list):
                for idx, score in enumerate(data["scores"]):
                    if idx < len(new_scores):
                        new_scores[idx] = float(score)
            else:
                logger.warning(f"Reranker returned unrecognized shape: {list(data)[:5]}")
                return candidates

            for c, ns in zip(candidates, new_scores):
                if ns is not None:
                    c.score = ns
            candidates.sort(key=lambda x: x.score, reverse=True)

            cap = self._cfg("reranker_top_k", 0)
            try:
                cap_int = int(cap) if cap else 0
            except (TypeError, ValueError):
                cap_int = 0
            if cap_int > 0:
                candidates = candidates[:cap_int]
            logger.info(f"Reranker applied: {len(candidates)} candidates resorted")
            return candidates
        except Exception as e:
            logger.warning(f"Reranker failed: {e}. Keeping original order.")
            return candidates