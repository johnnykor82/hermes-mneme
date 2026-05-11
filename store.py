import sqlite3
import uuid
import json
import hashlib
import datetime
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import sqlite_vec at module level
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None
    SQLITE_VEC_AVAILABLE = False

class ContextStore:
    """Manages the SQLite database for the context engine plugin."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self, load_vec: bool = False):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if load_vec and SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                pass
        return conn

    def _init_db(self):
        """Creates tables if they don't exist."""
        # First, create tables that don't depend on vec0
        conn = self._get_connection()
        cursor = conn.cursor()

        # Sessions table (Change 2.2)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            parent_id TEXT NULL,
            interface TEXT NOT NULL,
            label TEXT NULL,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
        """)

        # Session lineage — tracks compression chain (old_session → new_session)
        # When Hermes compresses, it creates a new session. This table lets us
        # find all ancestor sessions so retrieval can search across the full history.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_lineage (
            old_session_id TEXT NOT NULL,
            new_session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (old_session_id, new_session_id)
        )
        """)

        # Events table (Component 1)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL, -- user_message, assistant_message, tool_call, tool_output
            role TEXT NOT NULL, -- user, assistant, tool
            content TEXT,
            tool_name TEXT NULL,
            tool_input TEXT NULL, -- JSON
            token_estimate INTEGER DEFAULT 0
        )
        """)

        # Embedding Index (Component 2 + Change 1)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_index (
            event_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            embedding BLOB NOT NULL, -- stored as bytes
            token_count INTEGER,
            type TEXT,
            embedding_model_id TEXT NOT NULL,
            PRIMARY KEY (event_id, embedding_model_id)
        )
        """)

        # Vector index for fast similarity search (optional — requires sqlite-vec)
        if SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                    embedding float[1024]
                );
                """)
            except Exception as _vec_err:
                logger.warning(f"vec0 not available: {_vec_err}. Vector search will use Python fallback.")
        else:
            logger.info("sqlite-vec Python package not available. Skipping vec0 table creation.")

        # Mapping table to link vec_items.rowid with event_id
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vec_items_meta (
            rowid INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL,
            embedding_model_id TEXT NOT NULL,
            segment_id TEXT NOT NULL
        );
        """)

        # Execution State (Component 3)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_state (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL, -- JSON blob
            updated_at TEXT NOT NULL
        )
        """)

        # Execution Graph (Component 5b)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_graph (
            from_event_id TEXT NOT NULL,
            to_event_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (from_event_id, to_event_id)
        )
        """)

        # State history — append-only log of execution_state snapshots so the
        # agent can review past goals via get_goal_history(). Sized at ~100B per
        # row (goal + current_step strings); negligible even on long sessions.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            goal TEXT,
            current_step TEXT,
            intent_label TEXT,
            decisions_added_json TEXT
        )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_state_history_session "
            "ON state_history(session_id, timestamp)"
        )

        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")

    # --- Session Methods ---

    def create_session(self, session_id: str, interface: str, parent_id: Optional[str] = None):
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        conn = self._get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, parent_id, interface, created_at, last_active, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (session_id, parent_id, interface, now, now)
        )
        conn.commit()
        conn.close()

    # --- Event Methods ---

    @staticmethod
    def make_event_id(session_id: str, msg_index: int, role: str, content: str,
                      sub_index: int = 0) -> str:
        """Deterministic event_id derived from (session, message position, role, content).

        Same logical message produces the same id on re-ingest, so INSERT OR IGNORE
        on the events table makes ingest idempotent — critical for session resume
        after gateway restart, where Hermes may hand us the same buffer twice.

        sub_index disambiguates assistant messages with multiple tool_calls
        (one message → many events).
        """
        h = hashlib.sha256()
        h.update(session_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(msg_index).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(sub_index).encode("utf-8"))
        h.update(b"\x00")
        h.update((role or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((content or "").encode("utf-8"))
        return h.hexdigest()

    def session_exists(self, session_id: str) -> Optional[str]:
        """Returns the session's status string if present in the sessions table,
        else None. Used to distinguish a truly fresh session_id from one that's
        being resumed after gateway restart."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT status FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        return row["status"] if row else None

    def latest_segment_id(self, session_id: str) -> Optional[str]:
        """Returns segment_id of the most recent event in the given session,
        or None if the session has no events."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT segment_id FROM events WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        conn.close()
        return row["segment_id"] if row else None

    def reactivate_session(self, session_id: str) -> None:
        """Mark a session as 'active' again — used when resuming a session
        previously closed/compressed at gateway shutdown.

        After lazy session creation (A0), it is normal for the row to be
        absent here (e.g. fresh resume of a session_id that was never
        materialized). The UPDATE is a silent no-op in that case — we
        intentionally do not warn, because add_event() will UPSERT the row
        the moment a real event arrives.
        """
        conn = self._get_connection()
        conn.execute(
            "UPDATE sessions SET status='active', last_active=? WHERE session_id=?",
            (datetime.datetime.now(tz=datetime.timezone.utc).isoformat(), session_id)
        )
        conn.commit()
        conn.close()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """B1: aggregate cheap counters for a session — used by the adaptive
        memory hint and surfaced through get_execution_state so the LLM has
        a concrete sense of how much context lives outside the active window.

        Returns a dict with: total_events, total_segments, events_by_type,
        first_event_ts, last_event_ts, compressed_ancestors_count.
        Empty/zero values when the session has no events yet.
        """
        conn = self._get_connection()
        try:
            agg = conn.execute(
                "SELECT COUNT(*) AS n, COUNT(DISTINCT segment_id) AS segs, "
                "MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts "
                "FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            by_type_rows = conn.execute(
                "SELECT type, COUNT(*) AS n FROM events WHERE session_id = ? GROUP BY type",
                (session_id,),
            ).fetchall()
            ancestors = conn.execute(
                "SELECT COUNT(*) AS n FROM session_lineage WHERE new_session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        events_by_type = {r["type"]: int(r["n"]) for r in by_type_rows} if by_type_rows else {}
        return {
            "total_events": int(agg["n"]) if agg and agg["n"] else 0,
            "total_segments": int(agg["segs"]) if agg and agg["segs"] else 0,
            "events_by_type": events_by_type,
            "first_event_ts": agg["first_ts"] if agg else None,
            "last_event_ts": agg["last_ts"] if agg else None,
            "compressed_ancestors_count": int(ancestors["n"]) if ancestors else 0,
        }

    def get_lineage_chain(self, session_id: str) -> List[str]:
        """Return every session id in the same conversation thread as
        `session_id` — i.e. follow `session_lineage` backwards to the root
        AND forwards to the tip, then return the unique set as a list.

        Why this exists: after a long conversation hermes-mneme accumulates
        a chain S0 → S1 → ... → Sn where reassign_session has moved all
        events forward. `self.session_id` may point at any middle node
        (e.g. immediately after a compression boundary, before the next
        compress() lands new events under the latest node), so a query
        keyed on the literal session_id misses everything. The chain is
        the right unit of "the current conversation".

        Returns at least [session_id] even if the lineage table has no
        rows for this id. Order is unspecified; callers should not rely
        on it.
        """
        if not session_id:
            return []
        seen = {session_id}
        # Walk backwards to ancestors.
        frontier = [session_id]
        conn = self._get_connection()
        try:
            while frontier:
                next_frontier: List[str] = []
                for sid in frontier:
                    rows = conn.execute(
                        "SELECT old_session_id FROM session_lineage WHERE new_session_id = ?",
                        (sid,),
                    ).fetchall()
                    for r in rows:
                        ancestor = r["old_session_id"]
                        if ancestor and ancestor not in seen:
                            seen.add(ancestor)
                            next_frontier.append(ancestor)
                frontier = next_frontier
            # Walk forwards to descendants.
            frontier = [session_id]
            while frontier:
                next_frontier = []
                for sid in frontier:
                    rows = conn.execute(
                        "SELECT new_session_id FROM session_lineage WHERE old_session_id = ?",
                        (sid,),
                    ).fetchall()
                    for r in rows:
                        descendant = r["new_session_id"]
                        if descendant and descendant not in seen:
                            seen.add(descendant)
                            next_frontier.append(descendant)
                frontier = next_frontier
        finally:
            conn.close()
        return list(seen)

    def find_last_active_session(self, exclude_session_id: Optional[str] = None) -> Optional[str]:
        """Return the session_id of the most recently active session (status='active')
        whose id is not `exclude_session_id`. Used by drift recovery after a
        process restart: when the plugin starts cold (self.session_id is None)
        and Hermes hands it a brand-new session id, this lets us find the
        previous live session and reassign its events instead of orphaning them.

        Returns None if no other active session exists.
        """
        conn = self._get_connection()
        try:
            if exclude_session_id:
                row = conn.execute(
                    "SELECT session_id FROM sessions "
                    "WHERE status='active' AND session_id != ? "
                    "ORDER BY last_active DESC LIMIT 1",
                    (exclude_session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT session_id FROM sessions "
                    "WHERE status='active' "
                    "ORDER BY last_active DESC LIMIT 1"
                ).fetchone()
            return row["session_id"] if row else None
        finally:
            conn.close()

    def count_events(self, session_id: str) -> int:
        """Cheap count of events in a session (used by adaptive memory hint
        threshold and by external callers checking session emptiness)."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["n"]) if row else 0
        finally:
            conn.close()

    def delete_empty_sessions(self, min_age_seconds: int = 86400) -> int:
        """Remove phantom sessions: rows in `sessions` that have NO events,
        are not compression-rollover children (parent_id IS NULL), and were
        last touched more than `min_age_seconds` ago.

        The age cutoff protects:
          * sessions just created and about to receive their first event,
          * compression rollovers whose first reassign happens shortly after
            the row appears.

        Returns the number of rows deleted.
        """
        cutoff_iso = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(seconds=int(min_age_seconds))
        ).isoformat()
        conn = self._get_connection()
        try:
            cur = conn.execute(
                "DELETE FROM sessions "
                "WHERE parent_id IS NULL "
                "  AND last_active < ? "
                "  AND NOT EXISTS ("
                "      SELECT 1 FROM events e WHERE e.session_id = sessions.session_id"
                "  )",
                (cutoff_iso,),
            )
            conn.commit()
            return cur.rowcount or 0
        finally:
            conn.close()

    def add_event(self, session_id: str, segment_id: str, event_type: str, role: str,
                  content: str, tool_name: Optional[str] = None, tool_input: Optional[Dict] = None,
                  token_estimate: int = 0, event_id: Optional[str] = None) -> Tuple[str, bool]:
        """Insert an event. If event_id is provided (deterministic), uses
        INSERT OR IGNORE and returns (event_id, inserted_flag) — inserted_flag
        is False when the row already existed (idempotent re-ingest).

        Without event_id, falls back to a fresh uuid4 (legacy behavior) and
        always returns inserted_flag=True."""
        if event_id is None:
            event_id = str(uuid.uuid4())
            sql = """INSERT INTO events (id, session_id, segment_id, timestamp, type, role, content, tool_name, tool_input, token_estimate)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        else:
            sql = """INSERT OR IGNORE INTO events (id, session_id, segment_id, timestamp, type, role, content, tool_name, tool_input, token_estimate)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        conn = self._get_connection()
        # Lazy session row: materialize sessions row in the SAME transaction as
        # the first event. Guarantees the invariant "sessions row exists ⇔ at
        # least one event exists (or it is a compression rollover)".
        # Without this, callers that skipped create_session() would write events
        # whose session_id has no parent row.
        conn.execute(
            "INSERT OR IGNORE INTO sessions "
            "(session_id, parent_id, interface, created_at, last_active, status) "
            "VALUES (?, NULL, 'unknown', ?, ?, 'active')",
            (session_id, now, now),
        )
        # Refresh last_active on every event so resume picks the right tail.
        conn.execute(
            "UPDATE sessions SET last_active=? WHERE session_id=?",
            (now, session_id),
        )
        cur = conn.execute(
            sql,
            (event_id, session_id, segment_id, now, event_type, role, content, tool_name,
             json.dumps(tool_input) if tool_input else None, token_estimate)
        )
        inserted = cur.rowcount > 0
        conn.commit()
        conn.close()
        return event_id, inserted

    # --- State Methods ---

    def save_state(self, session_id: str, state_dict: Dict[str, Any]):
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO execution_state (session_id, state_json, updated_at) VALUES (?, ?, ?)""",
            (session_id, json.dumps(state_dict), now)
        )
        conn.commit()
        conn.close()

    def commit_state(self, session_id: str, state_dict: Dict[str, Any]) -> None:
        """Atomically write execution_state AND append a state_history snapshot
        in a single transaction (A4). Replaces the prior two-call pattern of
        save_state + append_state_history, which committed independently and
        could leave execution_state without a matching history row (or vice
        versa) on crash.

        History-append rule combines two conditions:

        1. **Diff-driven append** (legacy): write a row when (goal,
           current_step, intent_label) differ from the previous row. This
           still catches every real state change.

        2. **Keepalive append** (Fix B): in real sessions the goal is set
           from the first user_message and never changes, so condition (1)
           never fires after turn 1 and state_history stays empty for the
           rest of the conversation — get_goal_history returns ``[]`` and
           A3 crash-recovery has nothing to recover from. Force an append
           if the last row is older than `keepalive_seconds` (default 120s)
           OR if `last_tool` changed (a cheap proxy for "the agent did
           anything since the last snapshot").
        """
        if not state_dict:
            return
        goal = state_dict.get("goal") or None
        current_step = state_dict.get("current_step") or None
        enrichment = state_dict.get("enrichment") or {}
        intent_label = enrichment.get("intent_label") or None
        decisions = enrichment.get("decisions") or []
        decisions_json = (
            json.dumps(decisions, ensure_ascii=False) if decisions else None
        )
        last_tool = state_dict.get("last_tool") or None
        keepalive_seconds = 120
        now_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        now = now_dt.isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO execution_state "
                "(session_id, state_json, updated_at) VALUES (?, ?, ?)",
                (session_id, json.dumps(state_dict), now),
            )
            prev = conn.execute(
                "SELECT timestamp, goal, current_step, intent_label, "
                "       decisions_added_json "
                "FROM state_history "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            diff_changed = not (
                prev
                and prev["goal"] == goal
                and prev["current_step"] == current_step
                and prev["intent_label"] == intent_label
            )
            keepalive_due = False
            activity_since_prev = False
            if prev is not None and not diff_changed:
                # Time-based keepalive.
                try:
                    prev_ts = datetime.datetime.fromisoformat(prev["timestamp"])
                    age = (now_dt - prev_ts).total_seconds()
                    if age >= keepalive_seconds:
                        keepalive_due = True
                except Exception:
                    pass
                # Activity-based keepalive: any event landed in this session
                # since the previous snapshot. Catches tool_call, tool_output,
                # assistant_message — anything that proves progress was made.
                # state_history must reflect "did anything happen" granularly
                # enough for get_goal_history and A3 crash-recovery to be
                # useful; otherwise it stays at a single row for hours.
                try:
                    activity_row = conn.execute(
                        "SELECT 1 FROM events "
                        "WHERE session_id = ? AND timestamp > ? LIMIT 1",
                        (session_id, prev["timestamp"]),
                    ).fetchone()
                    if activity_row:
                        activity_since_prev = True
                except Exception:
                    pass
            should_append = diff_changed or keepalive_due or activity_since_prev
            if should_append:
                conn.execute(
                    "INSERT INTO state_history "
                    "(session_id, timestamp, goal, current_step, intent_label, decisions_added_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, now, goal, current_step, intent_label, decisions_json),
                )
            conn.commit()
        except Exception as e:
            logger.warning(f"commit_state failed for {session_id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT state_json FROM execution_state WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        if row:
            return json.loads(row['state_json'])
        return None

    # --- Embedding Methods ---
    # Note: add_embedding is now handled entirely by EmbeddingIndex to avoid duplication.
    # This method is kept for backward compatibility but delegates to the indexer.
    def add_embedding(self, event_id: str, segment_id: str, embedding: bytes, model_id: str, token_count: int, event_type: str):
        """Legacy delegate — embedding writes are handled by EmbeddingIndex."""
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO embedding_index
                (event_id, segment_id, embedding, embedding_model_id, token_count, type)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (event_id, segment_id, embedding, model_id, token_count, event_type)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store embedding metadata: {e}")
        finally:
            conn.close()

    def get_latest_compressed_session(self, current_session_id: str) -> Optional[str]:
        """Find the most recent ancestor session that was compressed into current.
        Walks the lineage chain backwards to find the root compressed session."""
        conn = self._get_connection()
        try:
            # Find direct parent
            row = conn.execute(
                "SELECT old_session_id FROM session_lineage WHERE new_session_id = ? ORDER BY created_at DESC LIMIT 1",
                (current_session_id,)
            ).fetchone()
            if row:
                return row[0]
            return None
        finally:
            conn.close()

    def reassign_session(self, old_session_id: str, new_session_id: str,
                         interface: str = "unknown") -> int:
        """Move all data from old_session_id to new_session_id (compression
        inheritance). Atomically also materializes the `sessions` row for
        new_session_id (parent_id=old_session_id) and marks old as
        'compressed'. Returns total number of updated rows across all tables.

        Atomicity (A1): the new sessions row, the lineage record, and the
        old-session status update are all in the SAME transaction as the
        events/graph/state UPDATEs. A SIGKILL between them no longer leaves
        events under a session_id with no parent row.

        Cycle rejection (Fix D): if a lineage row already exists in the
        opposite direction (new_session_id → old_session_id), this call is
        a rollback / round-trip — the events are already under
        new_session_id from the previous reassign, and writing the reverse
        lineage row would create a cycle. Return 0 and skip without
        modifying any table.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        # Fix D: refuse to create a reverse-edge cycle in session_lineage.
        # See tests/diagnostic/reproduce_state_bugs.py, scenario D.
        check_conn = self._get_connection()
        try:
            reverse = check_conn.execute(
                "SELECT 1 FROM session_lineage "
                "WHERE old_session_id = ? AND new_session_id = ? LIMIT 1",
                (new_session_id, old_session_id),
            ).fetchone()
        finally:
            check_conn.close()
        if reverse:
            logger.info(
                f"reassign_session: refusing rollback {old_session_id} -> {new_session_id} "
                f"(reverse lineage already exists)."
            )
            return 0
        conn = self._get_connection()
        total = 0
        try:
            now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
            # Materialize sessions row for new_session_id FIRST so any FK-like
            # check or subsequent reader sees a consistent state.
            conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_id, parent_id, interface, created_at, last_active, status) "
                "VALUES (?, ?, ?, ?, ?, 'active')",
                (new_session_id, old_session_id, interface, now, now),
            )
            # Update events.session_id
            cur = conn.execute("UPDATE events SET session_id = ? WHERE session_id = ?",
                             (new_session_id, old_session_id))
            total += cur.rowcount or 0
            # Rebase segment_id so retrieval that filters by current segment finds
            # inherited events (their segment_id was seg_{old_session}_N).
            conn.execute(
                "UPDATE events SET segment_id = REPLACE(segment_id, ?, ?) WHERE session_id = ? AND segment_id LIKE ?",
                (old_session_id, new_session_id, new_session_id, f"%{old_session_id}%")
            )
            # Mirror the rebase in vec_items_meta (its segment_id was set at insert time).
            conn.execute(
                "UPDATE vec_items_meta SET segment_id = REPLACE(segment_id, ?, ?) WHERE segment_id LIKE ?",
                (old_session_id, new_session_id, f"%{old_session_id}%")
            )
            # And in embedding_index.
            conn.execute(
                "UPDATE embedding_index SET segment_id = REPLACE(segment_id, ?, ?) WHERE segment_id LIKE ?",
                (old_session_id, new_session_id, f"%{old_session_id}%")
            )
            # embedding_index and vec_items_meta inherit implicitly via event_id —
            # no session_id column on them.
            # Update execution_graph
            cur = conn.execute("UPDATE execution_graph SET session_id = ? WHERE session_id = ?",
                             (new_session_id, old_session_id))
            total += cur.rowcount or 0
            # Update execution_state
            cur = conn.execute("UPDATE OR REPLACE execution_state SET session_id = ? WHERE session_id = ?",
                             (new_session_id, old_session_id))
            total += cur.rowcount or 0
            # Record lineage
            conn.execute("INSERT OR IGNORE INTO session_lineage (old_session_id, new_session_id, created_at) VALUES (?, ?, ?)",
                        (old_session_id, new_session_id, now))
            # Update sessions table — mark old as compressed, keep for reference
            conn.execute("UPDATE sessions SET status = 'compressed', last_active = ? WHERE session_id = ?",
                        (now, old_session_id))
            conn.commit()
            logger.info(f"Reassigned session {old_session_id} -> {new_session_id}: {total} rows updated")
        except Exception as e:
            logger.error(f"Failed to reassign session: {e}")
            conn.rollback()
        finally:
            conn.close()
        return total

    def get_recent_turns(self, session_id: str, turns: int = 5) -> List[Dict[str, Any]]:
        """B7: return the last N user turns plus all events that followed
        each user message until the next one. Each turn entry groups
        user_message + subsequent assistant/tool events in chronological
        order so the LLM gets a clean "what happened N turns ago" view
        without having to walk events one by one.
        """
        turns = max(1, int(turns))
        conn = self._get_connection()
        try:
            user_rows = conn.execute(
                "SELECT id, timestamp FROM events "
                "WHERE session_id = ? AND type = 'user_message' "
                "ORDER BY timestamp DESC LIMIT ?",
                (session_id, turns),
            ).fetchall()
            user_rows = list(reversed(user_rows))  # chronological order
            if not user_rows:
                return []
            user_ts = [r["timestamp"] for r in user_rows]
            # Boundaries: each turn spans [user_ts[i], user_ts[i+1]); the
            # last turn spans [user_ts[-1], +∞).
            out: List[Dict[str, Any]] = []
            for i, start_ts in enumerate(user_ts):
                end_ts = user_ts[i + 1] if i + 1 < len(user_ts) else None
                if end_ts is not None:
                    rows = conn.execute(
                        "SELECT id, type, role, content, tool_name, timestamp, segment_id "
                        "FROM events WHERE session_id = ? AND timestamp >= ? AND timestamp < ? "
                        "ORDER BY timestamp ASC",
                        (session_id, start_ts, end_ts),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, type, role, content, tool_name, timestamp, segment_id "
                        "FROM events WHERE session_id = ? AND timestamp >= ? "
                        "ORDER BY timestamp ASC",
                        (session_id, start_ts),
                    ).fetchall()
                events = []
                for r in rows:
                    content = r["content"] or ""
                    if len(content) > 600:
                        content = content[:600] + "…"
                    events.append({
                        "event_id": r["id"],
                        "type": r["type"],
                        "role": r["role"],
                        "tool_name": r["tool_name"],
                        "timestamp": r["timestamp"],
                        "segment_id": r["segment_id"],
                        "content": content,
                    })
                out.append({
                    "turn_index": i + 1,
                    "user_timestamp": start_ts,
                    "events": events,
                })
            return out
        finally:
            conn.close()

    def get_recent_events(self, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Fetch recent events for a session."""
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT id, type, role, content, tool_name, timestamp
            FROM events
            WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append({
                "id": row["id"],
                "type": row["type"],
                "role": row["role"],
                "content": row["content"],
                "tool_name": row["tool_name"],
                "timestamp": row["timestamp"]
            })
        return list(reversed(events)) # Return in chronological order

    # --- State History (append-only log of execution_state snapshots) ---

    def append_state_history(self, session_id: str, state_dict: Dict[str, Any]) -> None:
        """Append a snapshot row to state_history. Skips no-op writes where
        goal+current_step+intent_label are unchanged from the previous row —
        the enricher fires every N turns and would otherwise spam duplicates."""
        if not state_dict:
            return
        goal = state_dict.get("goal") or None
        current_step = state_dict.get("current_step") or None
        enrichment = state_dict.get("enrichment") or {}
        intent_label = enrichment.get("intent_label") or None
        decisions = enrichment.get("decisions") or []
        decisions_json = json.dumps(decisions, ensure_ascii=False) if decisions else None

        conn = self._get_connection()
        try:
            prev = conn.execute(
                "SELECT goal, current_step, intent_label FROM state_history "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,)
            ).fetchone()
            if prev and prev["goal"] == goal and prev["current_step"] == current_step \
                    and prev["intent_label"] == intent_label:
                return
            now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO state_history (session_id, timestamp, goal, current_step, "
                "intent_label, decisions_added_json) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, now, goal, current_step, intent_label, decisions_json)
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"append_state_history failed: {e}")
        finally:
            conn.close()

    def get_state_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent state_history rows for a session, oldest-first."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT timestamp, goal, current_step, intent_label, decisions_added_json "
                "FROM state_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, int(limit))
            ).fetchall()
        finally:
            conn.close()
        out = []
        for r in reversed(rows):
            decisions = []
            if r["decisions_added_json"]:
                try:
                    decisions = json.loads(r["decisions_added_json"])
                except Exception:
                    decisions = []
            out.append({
                "timestamp": r["timestamp"],
                "goal": r["goal"],
                "current_step": r["current_step"],
                "intent_label": r["intent_label"],
                "decisions": decisions,
            })
        return out

    def get_recent_unique_goals(self, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Last N distinct goals (most recent unique goal strings, chronological).
        Used to inject [GOAL TRAIL] into the system prompt every turn."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT timestamp, goal FROM state_history "
                "WHERE session_id = ? AND goal IS NOT NULL AND goal != '' "
                "ORDER BY id DESC",
                (session_id,)
            ).fetchall()
        finally:
            conn.close()
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in rows:
            g = r["goal"]
            if g in seen:
                continue
            seen.add(g)
            out.append({"timestamp": r["timestamp"], "goal": g})
            if len(out) >= int(limit):
                break
        return list(reversed(out))

    # --- Segment manifest (used by list_segments tool) ---

    def list_segments(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Aggregate per-segment stats for the given session.

        B3: for each segment we now return events_by_type (so the LLM can see
        whether a segment is conversation-heavy or tool-heavy at a glance),
        goal_at_end (the last goal recorded in state_history while this
        segment was active), and topic_tags from current execution_state
        enrichment (still session-wide in the current schema; per-segment
        topic_tags would need a schema change).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT segment_id, COUNT(*) AS n, MIN(timestamp) AS first_ts, "
                "MAX(timestamp) AS last_ts FROM events WHERE session_id = ? "
                "GROUP BY segment_id ORDER BY last_ts DESC LIMIT ?",
                (session_id, int(limit))
            ).fetchall()
            # Pull session-wide enrichment once (topic_tags live there).
            topic_tags: List[str] = []
            try:
                state_row = conn.execute(
                    "SELECT state_json FROM execution_state WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if state_row and state_row["state_json"]:
                    state_obj = json.loads(state_row["state_json"]) or {}
                    enrichment = state_obj.get("enrichment") or {}
                    topic_tags = list(enrichment.get("topic_tags") or [])
            except Exception as e:
                logger.warning(f"list_segments: topic_tags fetch failed: {e}")
            out = []
            for r in rows:
                seg_id = r["segment_id"]
                last_user = conn.execute(
                    "SELECT content FROM events WHERE session_id = ? AND segment_id = ? "
                    "AND type = 'user_message' ORDER BY timestamp DESC LIMIT 1",
                    (session_id, seg_id)
                ).fetchone()
                first_user = conn.execute(
                    "SELECT content FROM events WHERE session_id = ? AND segment_id = ? "
                    "AND type = 'user_message' ORDER BY timestamp ASC LIMIT 1",
                    (session_id, seg_id)
                ).fetchone()
                # Per-type breakdown for this segment.
                type_rows = conn.execute(
                    "SELECT type, COUNT(*) AS n FROM events "
                    "WHERE session_id = ? AND segment_id = ? GROUP BY type",
                    (session_id, seg_id),
                ).fetchall()
                events_by_type = {tr["type"]: int(tr["n"]) for tr in type_rows}
                # goal_at_end: the most recent state_history.goal whose
                # timestamp falls within this segment's time window. State
                # history is session-wide so we filter by timestamp.
                goal_at_end = None
                try:
                    goal_row = conn.execute(
                        "SELECT goal FROM state_history "
                        "WHERE session_id = ? AND timestamp <= ? AND goal IS NOT NULL "
                        "ORDER BY id DESC LIMIT 1",
                        (session_id, r["last_ts"]),
                    ).fetchone()
                    if goal_row and goal_row["goal"]:
                        goal_at_end = goal_row["goal"]
                except Exception as e:
                    logger.warning(f"list_segments: goal_at_end fetch failed: {e}")
                out.append({
                    "segment_id": seg_id,
                    "event_count": r["n"],
                    "events_by_type": events_by_type,
                    "first_ts": r["first_ts"],
                    "last_ts": r["last_ts"],
                    "first_user_snippet": (first_user["content"] or "")[:200] if first_user else None,
                    "last_user_snippet": (last_user["content"] or "")[:200] if last_user else None,
                    "goal_at_end": goal_at_end,
                    "topic_tags": topic_tags,  # session-wide for now
                })
            return out
        finally:
            conn.close()

    def get_segment_skeleton(self, session_id: str, segment_id: str,
                             max_events: int = 15) -> List[Dict[str, Any]]:
        """Pick the 'spine' of a segment for expand_context(mode='segment').
        Returns up to max_events: every user_message + every tool_call name +
        the last assistant_message. Long content snippets are truncated."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id, type, role, content, tool_name, timestamp "
                "FROM events WHERE session_id = ? AND segment_id = ? "
                "ORDER BY timestamp ASC",
                (session_id, segment_id)
            ).fetchall()
        finally:
            conn.close()
        users = [r for r in rows if r["type"] == "user_message"]
        tools = [r for r in rows if r["type"] == "tool_call"]
        last_asst = next((r for r in reversed(rows) if r["type"] == "assistant_message"), None)
        # Combine, dedup by id, sort, cap.
        combined = {r["id"]: r for r in users}
        for r in tools:
            combined.setdefault(r["id"], r)
        if last_asst is not None:
            combined.setdefault(last_asst["id"], last_asst)
        ordered = sorted(combined.values(), key=lambda r: r["timestamp"])
        if len(ordered) > max_events:
            # Keep first 5 and last (max-5) to preserve both ends of the segment.
            head = ordered[:5]
            tail = ordered[-(max_events - 5):]
            ordered = head + tail
        return [
            {
                "event_id": r["id"],
                "type": r["type"],
                "tool_name": r["tool_name"],
                "timestamp": r["timestamp"],
                "snippet": (r["content"] or "")[:200],
            }
            for r in ordered
        ]
