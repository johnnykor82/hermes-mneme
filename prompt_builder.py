import logging
import json
import copy
import tiktoken
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Token budget defaults (from spec)
STATE_CAP_TOKENS = 1000  # Execution state hard cap (Component 3, spec override)


class PromptBuilder:
    """Assembles the final structured context passed to the LLM.

    Responsibility boundary (strict):
    - Router returns ranked candidate list — unconstrained, untruncated.
    - Prompt Builder is the SOLE enforcer of all hard token limits.
    - No other component truncates content.
    """

    def __init__(self, total_budget: int = 16000, tokenizer_model: str = "cl100k_base",
                 protected_tail_turns: int = 6, state_budget_ratio: float = 0.05,
                 retrieved_budget_ratio: float = 0.45,
                 protected_tail_ratio: Optional[float] = None):
        self.total_budget = total_budget
        self.tokenizer_model = tokenizer_model
        self.protected_tail_turns = protected_tail_turns
        self.state_budget_ratio = state_budget_ratio
        self.retrieved_budget_ratio = retrieved_budget_ratio
        # Explicit tail ratio if provided; else derived (state+retrieved+tail=1.0).
        # Caller is expected to pass an explicit ratio that leaves headroom
        # (e.g. state=0.05, retrieved=0.30, tail=0.55, headroom=0.10) — see
        # spec section "Dynamic system prompt deduction".
        if protected_tail_ratio is None:
            self.protected_tail_ratio = max(
                0.0, 1.0 - state_budget_ratio - retrieved_budget_ratio
            )
        else:
            self.protected_tail_ratio = protected_tail_ratio
        self.tokenizer = self._initialize_tokenizer()
        self.system_prompt_tokens = 0  # Measured once at session start

    def _initialize_tokenizer(self):
        """Initialize tokenizer with fallback."""
        try:
            enc = tiktoken.get_encoding(self.tokenizer_model)
            return lambda text: len(enc.encode(text))
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}. Falling back to chars/4.")
            return lambda text: len(text) // 4

    def set_system_prompt_tokens(self, count: int):
        """Cache actual system prompt token count (called once at session start)."""
        self.system_prompt_tokens = count

    def build(
        self,
        execution_state: Dict[str, Any],
        retrieved_candidates: List[Any],  # List of RetrievalCandidate
        recent_messages: List[Dict[str, Any]],  # Last N turns (OpenAI format)
        current_user_message: str,
        system_prompt_override: str = None,
        goal_trail: Optional[List[Dict[str, Any]]] = None,
        memory_access_hint: Optional[str] = None,
        checkpoint_block: Optional[str] = None,
        cross_session_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assembles the final message list.

        Budget allocation (from spec):
        - system_prompt: measured once, deducted from total
        - execution_state: 5% of effective budget (hard cap 1000 tokens)
        - protected_tail: 50% of effective budget (last N turns, always included)
        - retrieved_context: 45% of effective budget (fill remaining, droppable)

        Output structure (list[dict] OpenAI format):
        1. System message: [EXECUTION STATE] + original system prompt
        2. Retrieved context prepended to first user message
        3. Recent turns preserved with original roles
        4. Current user message as final user turn

        Collision resolution (deterministic order):
        1. Drop lowest-scored retrieved chunks
        2. Truncate protected_tail from oldest (min 2 turns)
        3. Execution state is NEVER dropped
        """
        effective_budget = self.total_budget - self.system_prompt_tokens

        # Budget allocations
        state_budget = int(effective_budget * self.state_budget_ratio)
        tail_budget = int(effective_budget * self.protected_tail_ratio)
        retrieved_budget = int(effective_budget * self.retrieved_budget_ratio)

        # 1. Execution State (with hard cap)
        state_json_str = json.dumps(execution_state, ensure_ascii=False)
        state_tokens = self.tokenizer(state_json_str)

        if state_tokens > STATE_CAP_TOKENS:
            state_json_str = self._compress_execution_state(execution_state)
            state_tokens = self.tokenizer(state_json_str)

        # 2. Retrieved Context (budget-constrained, droppable)
        retrieved_chunks_text = []
        retrieved_tokens_used = 0
        # Truncation policy: take each candidate whole if it fits the
        # remaining retrieved budget. If it doesn't fit, fall back to a
        # head-only slice that DOES fit (so we keep at least the lead) and
        # then stop — subsequent candidates are lower-scored and would also
        # need slicing. Old behaviour capped tool_output at 300 chars
        # unconditionally, which left ~98 % of the retrieved budget unused
        # whenever the pool happened to be tool-output-heavy.
        # Lower bound for a partial slice; below this it's not worth it.
        MIN_PARTIAL_TOKENS = 200
        # B2: prefix every retrieved chunk with `id=<event_id>` so the LLM
        # can call fetch_event(event_id) when a snippet is truncated. Without
        # the id, fetch_event has no way to address the chunk.
        for cand in retrieved_candidates:
            content = cand.content or ""
            ev_id = getattr(cand, "event_id", None) or ""
            header = f"[{cand.type} id={ev_id}] " if ev_id else f"[{cand.type}] "
            chunk_text = f"{header}{content}"
            chunk_tokens = self.tokenizer(chunk_text)
            remaining = retrieved_budget - retrieved_tokens_used
            if chunk_tokens <= remaining:
                retrieved_chunks_text.append(chunk_text)
                retrieved_tokens_used += chunk_tokens
                continue
            # Doesn't fit whole. If we have meaningful room, slice the
            # content to fit and stop the loop. Char/token ratio uses the
            # tokenizer's own count to avoid drift on non-ASCII.
            if remaining < MIN_PARTIAL_TOKENS:
                break
            header_tokens = self.tokenizer(header)
            content_budget = max(0, remaining - header_tokens - 4)
            if content_budget < MIN_PARTIAL_TOKENS:
                break
            # Approximate slice: chars ≈ tokens * 4 (cl100k_base avg). Then
            # measure exactly and trim further if we overshoot.
            approx_chars = content_budget * 4
            sliced = content[:approx_chars]
            while sliced and self.tokenizer(f"{header}{sliced}") > remaining:
                sliced = sliced[: int(len(sliced) * 0.9)]
            if not sliced:
                break
            partial = f"{header}{sliced}"
            retrieved_chunks_text.append(partial)
            retrieved_tokens_used += self.tokenizer(partial)
            break

        # 3. Protected Tail
        # `protected_tail_turns` is the FLOOR (always keep at least these N
        # turns); the ratio `protected_tail_ratio` is the upper hint, not a
        # hard cap. After everything else is sized we'll try to add OLDER
        # turns until the rest of the window is full — otherwise on long
        # sessions with a thin retrieved pool 50%+ of the budget sits idle.
        # A7: avoid double-counting `current_user_message` if the last entry
        # of `recent_messages` is the same user turn (engine passes the full
        # binding tail; the last message is the current user turn). Without
        # this guard its tokens are counted both in `tail_tokens` and in
        # `current_msg_tokens`, inflating the budget estimate.
        recent_messages = list(recent_messages)
        if (
            recent_messages
            and current_user_message
            and recent_messages[-1].get("role") == "user"
            and str(recent_messages[-1].get("content", "")).strip()
                == str(current_user_message).strip()
        ):
            recent_messages = recent_messages[:-1]
        floor_n = self.protected_tail_turns
        if len(recent_messages) > floor_n:
            tail_messages = list(recent_messages[-floor_n:])
            extendable_messages = list(recent_messages[:-floor_n])
        else:
            tail_messages = list(recent_messages)
            extendable_messages = []
        tail_tokens = sum(self.tokenizer(str(m.get("content", ""))) + 4 for m in tail_messages)

        # If the floor itself blows past tail_budget AND we have more than 2
        # turns, drop oldest down to tail_budget (spec rule: never below 2).
        if tail_tokens > tail_budget and len(tail_messages) > 2:
            dropped = 0
            while len(tail_messages) > 2 and tail_tokens > tail_budget:
                removed = tail_messages.pop(0)
                tail_tokens -= self.tokenizer(str(removed.get("content", ""))) + 4
                dropped += 1
            logger.info(
                f"tail floor over tail_budget: dropped {dropped} oldest "
                f"(tail_tokens={tail_tokens}, tail_budget={tail_budget})"
            )

        # 4. Current user message
        current_msg_tokens = self.tokenizer(current_user_message)

        # 4b. Tail extension — fill unused headroom with OLDER turns.
        # The floor at `protected_tail_turns` is the minimum, not the cap. On
        # long sessions retrieved often returns a thin pool (e.g. 4–6 chunks
        # totalling 1–7 k tokens) and tail sits at ~half the window: 30 %+
        # of the budget goes unused. Pull additional older messages out of
        # `extendable_messages` (chronological, oldest first kept by slicing)
        # while there's still room, leaving a small safety margin so token
        # counting drift doesn't trigger collision resolution.
        SAFETY_MARGIN = 256
        used = state_tokens + retrieved_tokens_used + tail_tokens + current_msg_tokens
        added_back = 0
        while extendable_messages:
            candidate = extendable_messages[-1]
            cost = self.tokenizer(str(candidate.get("content", ""))) + 4
            if used + cost > effective_budget - SAFETY_MARGIN:
                break
            tail_messages.insert(0, extendable_messages.pop())
            tail_tokens += cost
            used += cost
            added_back += 1
        if added_back:
            logger.info(
                f"Tail extended by {added_back} older turns to fill headroom "
                f"(tail_tokens={tail_tokens}, used={used}/{effective_budget})"
            )

        # 5. Budget collision resolution
        total_used = state_tokens + retrieved_tokens_used + tail_tokens + current_msg_tokens

        if total_used > effective_budget:
            logger.warning(f"Budget collision: {total_used}/{effective_budget}. Resolving...")

            # Step 1: Drop retrieved chunks (lowest-scored first — already sorted by router)
            while retrieved_chunks_text and (state_tokens + retrieved_tokens_used + tail_tokens + current_msg_tokens) > effective_budget:
                removed = retrieved_chunks_text.pop()
                retrieved_tokens_used -= self.tokenizer(removed)

            # Step 2: Truncate protected tail (keep most recent, min 2 turns)
            while len(tail_messages) > 2 and (state_tokens + retrieved_tokens_used + tail_tokens + current_msg_tokens) > effective_budget:
                removed = tail_messages.pop(0)
                tail_tokens -= self.tokenizer(str(removed.get("content", ""))) + 4

            # Step 3: If still over budget, log critical and return minimal
            total_used = state_tokens + retrieved_tokens_used + tail_tokens + current_msg_tokens
            if total_used > effective_budget:
                logger.critical(f"Budget still exceeded after resolution: {total_used}/{effective_budget}")
                tail_messages = tail_messages[-2:] if len(tail_messages) >= 2 else tail_messages
                retrieved_chunks_text = []

        # 6. Assemble output
        # System message: original system prompt + memory hint + goal trail + execution state + checkpoint
        system_content_parts = []
        if system_prompt_override:
            system_content_parts.append(system_prompt_override)
        if memory_access_hint:
            system_content_parts.append(memory_access_hint)
        trail_block = self._format_goal_trail(goal_trail)
        if trail_block:
            system_content_parts.append(trail_block)
        system_content_parts.append(f"[EXECUTION STATE]\n{state_json_str}")
        if checkpoint_block:
            system_content_parts.append(checkpoint_block)
        # B5: cross-session bootstrap candidates (rendered once per new
        # session, only if the bootstrap probe found anything). The block
        # nudges the LLM toward likely-relevant prior conversations without
        # forcing it to call context_search itself.
        if cross_session_candidates:
            lines = [
                "[CROSS-SESSION CANDIDATES]",
                "Эти прошлые сессии похожи на текущий запрос — возможно, "
                "они уже содержат ответ. Посмотри через context_search("
                "query, session='all') если релевантно:",
            ]
            for c in cross_session_candidates[:5]:
                sid = c.get("session_id") or "?"
                snippet = (c.get("snippet") or c.get("content") or "")[:160]
                ev = c.get("event_id") or ""
                lines.append(f"  • session={sid} event={ev} :: {snippet}")
            system_content_parts.append("\n".join(lines))
        system_content = "\n\n".join(system_content_parts)

        # Build message list preserving original roles for recent turns
        final_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # Retrieved context as a synthetic user message before recent turns
        if retrieved_chunks_text:
            retrieved_block = "[RETRIEVED CONTEXT]\n" + "\n".join(retrieved_chunks_text)
            final_messages.append({"role": "user", "content": retrieved_block})

        # Recent turns with original roles preserved
        for msg in tail_messages:
            final_messages.append(msg)

        # Current user message
        final_messages.append({"role": "user", "content": current_user_message})

        final_total = self.tokenizer(system_content) + sum(
            self.tokenizer(str(m.get("content", ""))) + 4 for m in final_messages[1:]
        )
        logger.info(f"Prompt built. Total tokens: {final_total}/{self.total_budget}")

        return final_messages

    def _format_goal_trail(self, trail: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        """Render the [GOAL TRAIL] block from the last N unique goals.
        Returns None when trail is empty so build() can skip the block."""
        if not trail:
            return None
        lines = ["[GOAL TRAIL] (последние цели сессии)"]
        for i, item in enumerate(trail):
            ts = (item.get("timestamp") or "")[:16].replace("T", " ")
            goal = (item.get("goal") or "").strip()
            if not goal:
                continue
            marker = "  ← текущая" if i == len(trail) - 1 else ""
            lines.append(f"  • {ts} — {goal}{marker}")
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    def _compress_execution_state(self, state: Dict[str, Any]) -> str:
        """
        Lossy compression of execution state (Component 3 rules):
        1. Truncate enrichment fields entirely
        2. Truncate open_loops and decision_stack to most recent 3
        3. Truncate active_entities to most recent 5
        4. Summarize goal and current_step (keep first + last sentence)

        Operates on a deep copy so the caller's state dict is not mutated.
        """
        try:
            state = copy.deepcopy(state)
            # 1. Truncate enrichment
            state["enrichment"] = {"decision_summary": None, "intent_label": None, "topic_tags": []}

            # 2. Truncate lists
            for key in ("open_loops", "decision_stack"):
                if key in state and isinstance(state[key], list) and len(state[key]) > 3:
                    state[key] = state[key][-3:]

            # 3. Truncate active_entities
            if "active_entities" in state and isinstance(state["active_entities"], list) and len(state["active_entities"]) > 5:
                state["active_entities"] = state["active_entities"][-5:]

            # 4. Summarize goal and current_step
            for key in ("goal", "current_step"):
                if key in state and isinstance(state[key], str) and len(state[key]) > 200:
                    sentences = state[key].split('. ')
                    if len(sentences) > 2:
                        state[key] = sentences[0] + ". ... " + sentences[-1]
                    else:
                        state[key] = state[key][:200] + "..."

            return json.dumps(state, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State compression failed: {e}")
            return json.dumps(state, ensure_ascii=False)[:1000]
