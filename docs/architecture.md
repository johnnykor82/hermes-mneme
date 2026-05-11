# Architecture Overview

This document provides a high-level overview of the Hermes Context Engine Plugin for contributors.

## System Components

### 1. Event Store (`store.py`)
- **Technology**: SQLite (single file `plugin.db`).
- **Purpose**: Immutable append-only log of all session events.
- **Tables**: `sessions`, `events`, `execution_state`, `state_history`,
  `embedding_index`, `execution_graph`.
- **Key Logic**: Written synchronously in `engine.py` during `compress()`.
  `state_history` is appended after every `save_state` so the goal
  trajectory is preserved (see Stage B in `DIVERGENCE.md`).

### 2. Embedding Index (`index.py`)
- **Technology**: `sqlite-vec` (mandatory), Jina-compatible API (default: `jina-embeddings-v5-text-small-retrieval-mlx` on port 8000).
- **Purpose**: Semantic retrieval layer.
- **Logic**: Generates embeddings via API, stores in `embedding_index` table as BLOBs.

### 3. Execution State (`engine.py` internal `_current_state`)
- **Purpose**: Tracks where the agent is in the current task (goal, step, open loops).
- **Source**: Derived from Event Store (can be rebuilt).
- **Update**: Updated deterministically from parsed messages.

### 4. Context Router (`router.py`)
- **Inputs**: `RouterInput` (message, state, segment, history).
- **Logic**:
    1.  **Classifier** (`classifier.py`): Detects intent (SWITCH, NEW_TASK, etc.) using signal functions.
    2.  **Query Builder**: Constructs search query from state.
    3.  **Retriever**: Calls `index.py` to search vector index.
    4.  **Scorer**: Ranks candidates using similarity, recency, dependency, and type weights.

### 5. Prompt Builder (`prompt_builder.py`)
- **Input**: Execution State, Retrieved Candidates, Recent Messages,
  optional `goal_trail`, `memory_access_hint`, `checkpoint_block`.
- **Output**: `list[dict]` (OpenAI format) ready for LLM.
- **System message structure** (Stage B):
  1. Original system prompt.
  2. `[MEMORY ACCESS]` — fixed hint about memory tools (when enabled).
  3. `[GOAL TRAIL]` — last N unique goals from `state_history`.
  4. `[EXECUTION STATE]` — JSON state.
  5. `[CHECKPOINT]` — only when the loop guard fires.
- **User messages**: `[RETRIEVED CONTEXT]` prepended before recent turns,
  current user message last.

### 6. Observability (`observability.py`)
- **Output**: `trace.jsonl` (one JSON object per turn).
- **Metrics**: `context_hit_rate`, `fallback_rate`, etc.

### 7. Agent Tools (`tools.py`)
- **Tools**: `context_search`, `fetch_event`, `expand_context`,
  `get_execution_state`, `list_segments`, `get_goal_history`.
- **Loop guard**: `engine.handle_tool_call` increments
  `_consecutive_memory_calls` for tools listed in
  `memory_tool_names`; when the counter reaches
  `checkpoint_after_n_memory_calls`, a `[CHECKPOINT]` block is
  appended to the next system message. Counter resets on any other
  tool call.
- **Integration**: Registered via `get_tool_schemas()` and handled by
  `handle_tool_call()`.

## Data Flow

```text
[User Message] 
    ↓
compress(messages) in engine.py
    ↓
1. Parse messages → store.add_event()
2. Update Execution State
3. Generate Embeddings → index.add_embedding()
    ↓
4. Classify Intent (Router)
5. Retrieve Candidates (Router)
6. Score & Rank (Router)
    ↓
7. Build Prompt (PromptBuilder)
    ↓
[LLM receives structured context]
```

## Multi-Session Isolation

- All queries filter by `session_id`.
- Sessions table tracks `parent_id` for subagents and compression chains.
- `session='current'` semantics in retrieval = **current session + all
  compression ancestors** (Stage A in `DIVERGENCE.md`).
- Cross-session access is opt-in via `context_search(session='all')`.

## Known Limitations (v1)

- **Segmentation**: Simplified (hard triggers only).
- **Execution Graph**: Edge writing not fully implemented in MVP.
- **Delta Extraction**: Disabled by default.
- **Cross-layer Conflict Resolver**: Not implemented.

## Divergences from spec

A point-by-point list of every place where the runtime code intentionally
differs from `~/.hermes/projects/context_engine/docs/speca.txt` lives in
[DIVERGENCE.md](DIVERGENCE.md). Reviewers should read it alongside the
spec to understand current behaviour.
