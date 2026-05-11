# Upstream Hermes PR: Resume picks empty phantom session over last non-empty one

> Этот файл — заготовка для issue/PR в репозиторий Hermes (upstream).
> Описанный баг находится в **Hermes core**, не в плагине hermes-mneme,
> поэтому фикс должен попасть в upstream-репозиторий, а не в плагин.
>
> Плагин hermes-mneme уже содержит mitigations со своей стороны
> (lazy session creation, atomic compression rollover, state recovery
> from history) — см. `docs/architecture.md` и комментарии с метками
> A0–A10 в `engine.py` / `store.py` / `index.py`.

## Title (для PR/issue)

`hermes -c` resumes the last empty (phantom) session instead of the last non-empty one

## Summary

When a Hermes session is created but the process dies (LLM hangs, terminal
killed, OOM, etc.) before the first message is committed, the corresponding
row in the SessionDB exists with `message_count = 0` and no `ended_at`.

On the next launch with `hermes -c` (`--continue`), `_resolve_last_session()`
picks this empty row because it sorts by `last_active DESC` / `started_at DESC`
without filtering on message presence. The user sees:

```
Session 20260510_191812_xxxxxx found but has no messages. Starting fresh.
```

…and the previous, valid session (with the work-in-progress conversation)
becomes effectively unreachable from `-c`.

## Reproducer

```bash
# Setup: create a real session with content
hermes -q "create file foo.txt"
# → session S1 has messages

# Trigger: open a fresh session, hang it, kill the process
hermes
# (don't type anything; let the prompt sit)
kill -9 $(pgrep -f "hermes")
# → session S2 has zero messages, no ended_at

# Symptom
hermes -c
# Output: "Session S2 found but has no messages. Starting fresh."
# Expected: resume S1 (the last non-empty session).
```

## Affected files

All paths relative to `~/.hermes/projects/context_engine/hermes-src/` in the
local install (real upstream paths in the Hermes repo will mirror this).

- `hermes_state.py:1449` — `SessionDB.resolve_resume_session_id()`
  Handles compression-chain forward-walk; does NOT handle phantom rows.
- `hermes_state.py:691-719` — `prune_empty_ghost_sessions()`
  Cleans up empty TUI sessions, but only those with `ended_at IS NOT NULL`
  AND older than 24 h. Phantoms from a hard kill have `ended_at = NULL` and
  are typically minutes old, so they slip through.
- `hermes_cli/main.py:617-634` — `_resolve_last_session(source)`
  Selects the most recently active session of the given source. Does not
  filter on `message_count > 0`.
- `cli.py:3591-3641` — `_init_agent` resume path
  Prints `"Session X found but has no messages. Starting fresh."` and lets
  the empty session win silently.
- `cli.py:3812-3887` — `_preload_resumed_session()`
  Mirror of the above for the early-display code path.
- `cli.py:5110-5152` — `_handle_resume_command`
  Same gap on `/resume <id>`.

## Proposed fix

### 1. New helper `SessionDB.resolve_to_nonempty_session(source)`

```python
def resolve_to_nonempty_session(
    self,
    source: str,
    exclude_subagents: bool = True,
) -> Optional[str]:
    """Return the id of the most recent session of `source` that has at
    least one message. Excludes subagent sessions (parent_session_id NOT
    NULL) by default so `-c` doesn't jump into a subagent.
    """
    where = "WHERE s.source = ? AND s.message_count > 0"
    if exclude_subagents:
        where += " AND s.parent_session_id IS NULL"
    row = self._conn.execute(
        f"SELECT s.id FROM sessions s {where} "
        "ORDER BY COALESCE(s.last_active, s.started_at) DESC LIMIT 1",
        (source,),
    ).fetchone()
    return row[0] if row else None
```

`message_count` is already maintained atomically alongside message INSERTs
(`hermes_state.py:1335-1342`), so the predicate is cheap.

### 2. Use it in `_resolve_last_session`

`hermes_cli/main.py:617-634`:

```python
def _resolve_last_session(source: str) -> Optional[str]:
    db = SessionDB()
    sid = db.resolve_to_nonempty_session(source)
    if sid:
        return sid
    # Optional: fall back to the existing search_sessions(...) for the
    # legacy behaviour when nothing non-empty exists.
    rows = db.search_sessions(source=source, limit=1)
    return rows[0]["id"] if rows else None
```

### 3. Friendlier behaviour in `_init_agent` / `_preload_resumed_session`

Instead of silently printing "Session X found but has no messages. Starting
fresh." when an explicit `--resume <id>` lands on a phantom, prompt the
user with the most-recent non-empty alternative:

```
Session 20260510_191812_xxxxxx is empty (likely interrupted before the
first message landed).
The most recent non-empty session of this source is
20260510_184523_yyyyyy (12 user messages, 2 hours ago).
Resume the non-empty one instead? [Y/n]
```

If the user agrees, switch `self.session_id` and continue normal resume.

### 4. Extend `prune_empty_ghost_sessions` for fresh phantoms

Add a second pass for sessions younger than 24 h but older than ~5 min:

```python
def prune_recent_phantom_sessions(self) -> int:
    cutoff = time.time() - 300  # 5 minutes — protects in-flight sessions
    def _do(conn):
        conn.execute("""
            DELETE FROM sessions
            WHERE source = 'tui'
              AND title IS NULL
              AND parent_session_id IS NULL
              AND started_at < ?
              AND NOT EXISTS (
                  SELECT 1 FROM messages WHERE messages.session_id = sessions.id
              )
        """, (cutoff,))
    return self._execute_write(_do) or 0
```

Call it from `cli.py:934` right next to the existing
`prune_empty_ghost_sessions` invocation.

## Tests

- `test_resolve_to_nonempty_session_picks_last_nonempty` — fixture: 3
  sessions, the most recent has 0 messages → method returns the second.
- `test_resolve_to_nonempty_session_excludes_subagents` — subagent (with
  `parent_session_id`) is skipped.
- Integration: full reproducer above (kill -9 between turns) wrapped in a
  pytest, asserting that `hermes -c` resumes the right session.

## Notes

- `resolve_resume_session_id` (compression-chain forward-walk) and
  `resolve_to_nonempty_session` (recency-with-message-filter) are
  semantically distinct — keep them as separate methods rather than fusing
  them.
- Plugin-side `hermes-mneme` already neutralises its half of the problem:
  `sessions` rows there are now created lazily inside `add_event()`
  (commit A0), so the plugin won't follow Hermes core down a RESUME path
  into an empty session. After the upstream fix lands, both halves agree.
- The interactive `[Y/n]` prompt in step 3 is optional — a non-interactive
  fallback (always pick the non-empty one) is also acceptable and arguably
  the better default.
