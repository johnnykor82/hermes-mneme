"""Tool output compression (Component 1).

Large tool outputs (file dumps, command stdout, API responses) bloat the
embedding index and waste retrieval budget. This module produces a compact
head+tail summary used **for embedding only** — the original content stays
in events.content untouched, so fetch_event always returns the full text.

Why head+tail and not LLM summarization:
- Deterministic, no extra latency, no extra dependency.
- Head usually contains the most semantically loaded tokens (filename,
  command, first lines of output).
- Tail captures the result/error which is what queries usually target
  (stack trace, exit code, last value).
"""

from typing import Optional


def summarize_for_embedding(
    content: str,
    threshold_tokens: int,
    summary_tokens: int,
    chars_per_token: int = 4,
) -> str:
    """Return a compact representation suitable for embedding.

    If the content is below the threshold, return as-is. Otherwise return
    head + ellipsis marker + tail, capped to ~summary_tokens.

    chars_per_token=4 is the standard tiktoken approximation; we use
    char counts here to avoid a tokenizer round-trip on every event.
    """
    if not content:
        return content
    threshold_chars = threshold_tokens * chars_per_token
    if len(content) <= threshold_chars:
        return content
    summary_chars = summary_tokens * chars_per_token
    head_chars = summary_chars // 2
    tail_chars = summary_chars - head_chars
    head = content[:head_chars]
    tail = content[-tail_chars:]
    return f"{head}\n...[truncated {len(content) - head_chars - tail_chars} chars]...\n{tail}"
