import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ContextTools:
    """Implements the agent-accessible memory tools (Component 7)."""

    def __init__(self, store, indexer, router, graph, engine=None):
        self.store = store
        self.indexer = indexer
        self.router = router
        self.graph = graph
        self.engine = engine  # back-reference for in-memory state fallback

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "context_search",
                "description": (
                    "Semantic search over the persistent memory store. Returns past events "
                    "(user messages, assistant replies, tool outputs, decisions) ranked by "
                    "similarity to your query.\n\n"
                    "USE THIS WHEN:\n"
                    "  - you need to recall something from earlier in this conversation,\n"
                    "  - the user asks about something you discussed before that's no longer in your active context,\n"
                    "  - you want to check what was decided / what command was run / what error appeared,\n"
                    "  - the user references a past session ('remember when…', 'what did we decide last week').\n\n"
                    "CROSS-SESSION SEARCH:\n"
                    "  By default `segment='current'` (active topic only). To search across the entire memory "
                    "(all past sessions, all topics in this Hermes profile), pass `segment='all'` AND "
                    "`session='all'`. This is the right choice for any question about prior work that "
                    "wasn't in this session."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language search query."},
                        "k": {"type": "integer", "default": 5, "description": "Max results to return."},
                        "segment": {
                            "type": "string",
                            "default": "current",
                            "description": (
                                "'current' = active session segment (fast, narrow). "
                                "'all' = search across all topic segments. "
                                "Or a specific segment_id."
                            ),
                        },
                        "session": {
                            "type": ["string", "array"],
                            "default": "current",
                            "description": (
                                "'current' = this session only. "
                                "'all' = search across ALL past sessions in this Hermes profile "
                                "(use this for cross-session recall). "
                                "Or a specific session_id, or list of ids."
                            ),
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "fetch_event",
                "description": (
                    "Returns the full raw event (untruncated content + metadata) by event_id.\n\n"
                    "USE THIS WHEN:\n"
                    "  - a retrieved-context chunk or context_search result is truncated "
                    "(tool_output snippets are capped at ~300 chars) and you need the full text,\n"
                    "  - you have an event_id from expand_context / context_search and need the "
                    "complete payload (full tool output, full assistant message, etc.).\n\n"
                    "Returned payload includes: id, session_id, segment_id, timestamp, type, role, "
                    "content, tool_name, tool_input, token_estimate. The `segment_id` lets you "
                    "chain into expand_context(mode='segment', segment_id=...) without an "
                    "extra lookup.\n\n"
                    "By default content is truncated to ~1000 tokens. Pass full=true to get the original."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string", "description": "Event id from context_search / expand_context / retrieved chunk."},
                        "full": {"type": "boolean", "default": False, "description": "Return the original content without truncation."}
                    },
                    "required": ["event_id"]
                }
            },
            {
                "name": "expand_context",
                "description": (
                    "Walks the execution graph from a seed event OR returns the skeleton of an entire segment.\n\n"
                    "Two modes:\n"
                    "  • mode='neighbors' (default): return events directly linked to seed_event_id "
                    "in the execution graph (tool_call → tool_output, decision edges).\n"
                    "  • mode='segment': return the spine of the segment that contains seed_event_id "
                    "(or use segment_id directly): every user_message + every tool_call + last assistant_message, "
                    "capped at 15 events.\n\n"
                    "USE THIS WHEN:\n"
                    "  - you got an event_id from list_segments / context_search and want to see the segment's spine,\n"
                    "  - you need to recover a tool_call/tool_output chain that scrolled out of the active window,\n"
                    "  - the user asks 'why did we do X?' or 'what came of Y?' — walk from the seed.\n\n"
                    "depth applies only to mode='neighbors' (1..5)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "seed_event_id": {"type": "string", "description": "Event id to start from (required for neighbors; optional for segment if segment_id is given)."},
                        "segment_id": {"type": "string", "description": "Segment id (mode='segment' only)."},
                        "mode": {"type": "string", "enum": ["neighbors", "segment"], "default": "neighbors"},
                        "depth": {"type": "integer", "default": 1, "description": "BFS depth, 1-5 (mode='neighbors' only)."}
                    }
                }
            },
            {
                "name": "get_execution_state",
                "description": (
                    "Returns the current execution state JSON: goal, current_step, last_tool, "
                    "open_loops, decision_stack, active_entities, intent_label, topic_tags.\n\n"
                    "USE THIS WHEN:\n"
                    "  - you are unsure what the user was working on before the active window was compressed,\n"
                    "  - you need to check open_loops (unfinished tasks the user expects you to remember),\n"
                    "  - you want to confirm the goal/current_step before suggesting a next action.\n\n"
                    "Falls back to ancestor sessions on compression chains, so it stays usable after compress."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_segments",
                "description": (
                    "Returns the table of contents of the current session: per-segment event count, "
                    "time range, first/last user-message snippet. Cheap to call. Start here when you "
                    "need to navigate older parts of the session."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10, "description": "Max segments to return (most recent first)."}
                    }
                }
            },
            {
                "name": "get_goal_history",
                "description": (
                    "Returns the last N goals of the session (with timestamps and intent labels). "
                    "Use this when you need to remember what you were working on N turns ago and the "
                    "[GOAL TRAIL] in the system prompt only shows the latest 3."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20, "description": "Max history rows (most recent N)."}
                    }
                }
            },
            {
                "name": "recall_recent",
                "description": (
                    "Returns the last N turns of the current session in chronological order. "
                    "Each turn = one user_message plus all assistant/tool events that followed it "
                    "until the next user_message. Cheaper and more direct than the "
                    "list_segments → context_search → fetch_event chain when you just need "
                    "'what happened N turns ago'. Long contents are truncated to 600 chars; use "
                    "fetch_event(event_id) on a specific event for the full content."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "turns": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of most-recent user turns to return (1..50).",
                        }
                    }
                }
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], session_id: str) -> str:
        try:
            if name == "context_search":
                raw = self._context_search(args, session_id)
            elif name == "fetch_event":
                raw = self._fetch_event(args)
            elif name == "expand_context":
                raw = self._expand_context(args, session_id)
            elif name == "get_execution_state":
                raw = self._get_execution_state(session_id)
            elif name == "list_segments":
                raw = self._list_segments(args, session_id)
            elif name == "get_goal_history":
                raw = self._get_goal_history(args, session_id)
            elif name == "recall_recent":
                raw = self._recall_recent(args, session_id)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            logger.error(f"Tool call {name} failed: {e}")
            return json.dumps({"error": str(e)})
        # B8: annotate the response with an estimated token cost so the LLM
        # can budget further memory calls (especially fetch_event(full=true)).
        # Heuristic: chars / 4 ≈ tokens for cl100k-like tokenizers; cheap and
        # tokenizer-free. We only mutate dict payloads — list results
        # (context_search) keep their existing array shape so we don't break
        # callers that index results[i].
        try:
            est = max(1, len(raw) // 4)
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed["_estimated_tokens"] = est
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass
        return raw

    def _context_search(self, args: Dict, session_id: str) -> str:
        query = args.get("query", "")
        k = args.get("k", 5)
        segment = args.get("segment", "current")
        session = args.get("session", "current")

        if not query:
            # Fix C: surface the full args dict at WARNING level so an
            # operator can tell whether the LLM sent ``{}``, ``{"query":
            # null}``, or used a different key (``q`` / ``prompt`` /
            # ``text``). Without this, the only signal is the opaque
            # "Query is empty" error and there's no way to diagnose from
            # the outside.
            try:
                arg_keys = sorted((args or {}).keys())
            except Exception:
                arg_keys = []
            logger.warning(
                "context_search called with empty query — session=%s, "
                "args keys=%s, full args=%r",
                session_id, arg_keys, args,
            )
            return json.dumps({"error": "Query is empty",
                               "received_arg_keys": arg_keys})

        # Resolve session parameter.
        # Fix E: 'current' now broadens to the lineage chain of the current
        # session id — ancestors + descendants. After a long conversation
        # with many compression hops, the engine's self.session_id can
        # point at an empty middle node of the chain (events were
        # reassigned forward by reassign_session). Searching only the
        # literal current id misses the entire conversation in that case;
        # searching the chain gives the user the "everything we just
        # talked about" semantics they actually expect.
        if session == "current":
            chain = []
            try:
                chain = self.store.get_lineage_chain(session_id)
            except Exception as e:
                logger.warning(f"context_search: get_lineage_chain failed: {e}")
            if chain and len(chain) > 1:
                search_session_id = chain
                cross_session = False
            else:
                search_session_id = session_id
                cross_session = False
        elif session == "all":
            search_session_id = None
            cross_session = True
        elif isinstance(session, list):
            # Multiple session ids — handled via OR-filter below.
            search_session_id = list(session)
            cross_session = False
        else:
            search_session_id = session
            cross_session = False

        # Resolve segment parameter (A8: convert the magic strings to a clean
        # None contract before crossing the indexer boundary — historically
        # `'all'` was passed through as a literal string and only worked
        # because every indexer SQL path special-cased it).
        if segment in ("current", "all"):
            search_segment = None  # no segment filter
        else:
            search_segment = segment

        try:
            if cross_session:
                results = self.indexer.search_global(query, top_k=k)
            elif isinstance(search_session_id, list):
                results = self.indexer.search_sessions(
                    query, search_session_id, top_k=k, segment_id=search_segment
                )
            else:
                results = self.indexer.search(
                    query, search_session_id, segment_id=search_segment, top_k=k
                )

            output = []
            for r in results:
                event_id = r.get("event_id")
                # Fetch session_id + content snippet for the agent.
                conn = self.store._get_connection()
                row = conn.execute(
                    "SELECT session_id, type, content, timestamp FROM events WHERE id = ?",
                    (event_id,)
                ).fetchone()
                conn.close()
                if not row:
                    continue
                output.append({
                    "event_id": event_id,
                    "score": r.get("similarity"),
                    "type": row["type"],
                    "snippet": (row["content"] or "")[:300],
                    "timestamp": row["timestamp"],
                    "source_session_id": row["session_id"],
                })
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            logger.error(f"context_search failed: {e}")
            return json.dumps({"error": f"Search failed: {e}"})

    def _fetch_event(self, args: Dict) -> str:
        event_id = args.get("event_id")
        if not event_id:
            return json.dumps({"error": "event_id required"})
        full = bool(args.get("full", False))

        conn = self.store._get_connection()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.close()
        if not row:
            return json.dumps({"error": "Event not found"})
        payload = dict(row)
        content = payload.get("content") or ""
        # Default cap: ~1000 tokens ≈ 4000 chars. Cheap heuristic, no tokenizer
        # dependency. Pass full=true for the original.
        max_chars = 4000
        if not full and len(content) > max_chars:
            payload["content"] = content[:max_chars]
            payload["truncated"] = True
            payload["original_chars"] = len(content)
        else:
            payload["truncated"] = False
        return json.dumps(payload, ensure_ascii=False)

    def _expand_context(self, args: Dict, session_id: str) -> str:
        mode = (args.get("mode") or "neighbors").lower()
        seed_event_id = args.get("seed_event_id")
        segment_id = args.get("segment_id")

        if mode == "segment":
            # Resolve segment_id from seed if not given.
            if not segment_id:
                if not seed_event_id:
                    return json.dumps({"error": "segment mode requires segment_id or seed_event_id"})
                conn = self.store._get_connection()
                try:
                    row = conn.execute(
                        "SELECT segment_id, session_id FROM events WHERE id = ?",
                        (seed_event_id,)
                    ).fetchone()
                finally:
                    conn.close()
                if not row:
                    return json.dumps({"error": "seed_event_id not found"})
                segment_id = row["segment_id"]
                seed_session = row["session_id"]
            else:
                seed_session = session_id
            try:
                skeleton = self.store.get_segment_skeleton(seed_session, segment_id, max_events=15)
                return json.dumps({
                    "segment_id": segment_id,
                    "session_id": seed_session,
                    "events": skeleton,
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"segment skeleton failed: {e}"})

        depth = min(int(args.get("depth", 1) or 1), 5)
        if not seed_event_id:
            return json.dumps({"error": "seed_event_id required"})

        try:
            neighbors = self.graph.get_neighbors(seed_event_id, depth=depth)
            results = []
            for n in neighbors:
                conn = self.store._get_connection()
                row = conn.execute(
                    "SELECT id, type, content, timestamp FROM events WHERE id = ?",
                    (n["event_id"],)
                ).fetchone()
                conn.close()
                if row:
                    results.append({
                        "event_id": row["id"],
                        "type": row["type"],
                        "content": (row["content"] or "")[:200],
                        "timestamp": row["timestamp"],
                        "edge_type": n["edge_type"],
                    })
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"expand_context failed: {e}"})

    def _list_segments(self, args: Dict, session_id: str) -> str:
        limit = int(args.get("limit", 10) or 10)
        try:
            segments = self.store.list_segments(session_id, limit=limit)
            return json.dumps(segments, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"list_segments failed: {e}"})

    def _get_goal_history(self, args: Dict, session_id: str) -> str:
        limit = int(args.get("limit", 20) or 20)
        try:
            history = self.store.get_state_history(session_id, limit=limit)
            return json.dumps(history, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"get_goal_history failed: {e}"})

    def _recall_recent(self, args: Dict, session_id: str) -> str:
        turns = int(args.get("turns", 5) or 5)
        turns = max(1, min(turns, 50))
        try:
            recent = self.store.get_recent_turns(session_id, turns=turns)
            return json.dumps(recent, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"recall_recent failed: {e}"})

    def _get_execution_state(self, session_id: str) -> str:
        # B1: attach session stats so the LLM can see how much context lives
        # outside the active window (useful when deciding whether to spend
        # turns on memory tools).
        try:
            stats = self.store.get_session_stats(session_id)
        except Exception as e:
            logger.warning(f"get_execution_state: stats fetch failed: {e}")
            stats = None

        # 1. Try DB by current session_id.
        state = self.store.load_state(session_id)
        if state:
            if stats:
                state["_session_stats"] = stats
            return json.dumps(state, ensure_ascii=False)

        # 2. Inheritance fallback — DB row may live under an ancestor session_id
        # (compression chain). Walk lineage backwards.
        try:
            ancestor = self.store.get_latest_compressed_session(session_id)
            seen = {session_id}
            while ancestor and ancestor not in seen:
                seen.add(ancestor)
                state = self.store.load_state(ancestor)
                if state:
                    state["_loaded_from_ancestor_session_id"] = ancestor
                    if stats:
                        state["_session_stats"] = stats
                    return json.dumps(state, ensure_ascii=False)
                ancestor = self.store.get_latest_compressed_session(ancestor)
        except Exception as e:
            logger.warning(f"get_execution_state lineage walk failed: {e}")

        # 3. In-memory fallback — engine has the live state object even before
        # the first save_state() write of the session.
        if self.engine is not None:
            live = getattr(self.engine, "_current_state", None)
            if live:
                payload = dict(live)
                payload["_source"] = "in_memory_live"
                if stats:
                    payload["_session_stats"] = stats
                return json.dumps(payload, ensure_ascii=False)

        return json.dumps({"error": "No state found", "session_id": session_id})
