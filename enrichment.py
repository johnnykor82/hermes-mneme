"""LLM-driven enrichment of execution_state (Stage 5.1).

Extracts higher-level signals from recent conversation history:
    intent_label, topic_tags, decisions

Endpoint resolution priority:
  1. Config keys (enricher_endpoint + enricher_model + enricher_api_key) →
     direct HTTP call to that OpenAI-compatible endpoint.
  2. Hermes auxiliary_client.call_llm(task="enrichment") — Hermes resolves
     provider/model from config.yaml (auxiliary.enrichment.*) or auto-fallback
     (main provider → OpenRouter → Nous → custom endpoint → Anthropic).
  3. Hermes LLM params captured via update_model() — last-resort direct call
     using Hermes' main provider model.

Same pattern as hermes-lcm/extraction.py.
"""

import json
import logging
import re
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are a context analyzer. Read the recent conversation between a user and an AI assistant. Extract structured signals as JSON.

Return ONLY valid JSON, no prose, no markdown fences. Schema:
{{
  "intent_label": "<short phrase, what the user is currently trying to accomplish, max 120 chars>",
  "topic_tags": ["<2-6 short topical keywords, lowercase>"],
  "decisions": [
    {{"decision": "<what was decided>", "rationale": "<why, max 200 chars>"}}
  ]
}}

Rules:
- Use the same language as the conversation (e.g. respond in Russian if the user writes in Russian).
- intent_label must reflect the LATEST user turn, not older ones.
- topic_tags: short nouns/noun phrases (e.g. "reranker", "vec0 knn", "compressor").
- decisions: only INCLUDE decisions that were explicitly made or confirmed; do not invent.
- If nothing was decided, return decisions: [].
- Output must be parseable by json.loads. No comments, no trailing commas.

CONVERSATION:
{history}
"""


@dataclass
class EnrichmentResult:
    intent_label: Optional[str] = None
    topic_tags: List[str] = field(default_factory=list)
    decisions: List[Dict[str, str]] = field(default_factory=list)
    raw: Optional[str] = None


class LLMEnricher:
    """Calls a chat-completion endpoint to extract enrichment signals."""

    def __init__(self, config, hermes_llm: Optional[Dict[str, str]] = None):
        self.config = config
        self.hermes_llm = hermes_llm or {}

    def _cfg(self, key, default=None):
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _resolve_endpoint(self) -> Dict[str, str]:
        """Endpoint/model/key — config first, Hermes as fallback."""
        endpoint = self._cfg("enricher_endpoint", "") or self.hermes_llm.get("base_url", "")
        model = self._cfg("enricher_model", "") or self.hermes_llm.get("model", "")
        api_key = self._cfg("enricher_api_key", "") or self.hermes_llm.get("api_key", "")
        return {"endpoint": endpoint, "model": model, "api_key": api_key}

    def is_ready(self) -> bool:
        """Either explicit config endpoint OR Hermes auxiliary_client OR Hermes LLM fallback."""
        cfg = self._resolve_endpoint()
        if cfg["endpoint"] and cfg["model"]:
            return True
        # auxiliary_client is the preferred path — same as hermes-lcm/extraction.py
        try:
            from agent.auxiliary_client import call_llm  # noqa: F401
            return True
        except Exception:
            pass
        # last resort: direct call using Hermes' captured main-provider params
        return bool(self.hermes_llm.get("base_url") and self.hermes_llm.get("model"))

    def _format_history(self, recent_events: List[Dict[str, Any]], max_turns: int) -> str:
        events = recent_events[-max_turns * 2:]  # rough: 1 turn ~= user + assistant
        lines = []
        for ev in events:
            role = ev.get("role") or ev.get("type") or "?"
            content = (ev.get("content") or "").strip()
            if not content:
                continue
            if len(content) > 800:
                content = content[:600] + " ...[truncated]... " + content[-200:]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def enrich(self, recent_events: List[Dict[str, Any]]) -> Optional[EnrichmentResult]:
        if not recent_events:
            return None

        max_turns = int(self._cfg("enricher_max_history_turns", 10) or 10)
        timeout = int(self._cfg("enricher_timeout_seconds", 30) or 30)

        history = self._format_history(recent_events, max_turns)
        if not history.strip():
            return None

        prompt = PROMPT_TEMPLATE.format(history=history)
        messages = [
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        cfg = self._resolve_endpoint()
        content: Optional[str] = None

        # Path 1: explicit config endpoint (direct HTTP).
        if cfg["endpoint"] and cfg["model"]:
            content = self._call_direct(cfg, messages, timeout)

        # Path 2: Hermes auxiliary_client — same as hermes-lcm.
        if content is None:
            content = self._call_via_auxiliary(messages, timeout)

        # Path 3: direct call using Hermes' main-provider params.
        if content is None and self.hermes_llm.get("base_url") and self.hermes_llm.get("model"):
            fallback_cfg = {
                "endpoint": self.hermes_llm["base_url"],
                "model": self.hermes_llm["model"],
                "api_key": self.hermes_llm.get("api_key", ""),
            }
            content = self._call_direct(fallback_cfg, messages, timeout)

        if content is None:
            logger.warning("Enricher: all LLM paths failed.")
            return None

        parsed = self._safe_parse_json(content)
        if not parsed:
            logger.warning(f"Enricher: failed to parse JSON from response: {content[:200]!r}")
            return EnrichmentResult(raw=content)

        return EnrichmentResult(
            intent_label=(parsed.get("intent_label") or None),
            topic_tags=[str(t).strip().lower() for t in (parsed.get("topic_tags") or []) if str(t).strip()][:6],
            decisions=[
                {"decision": str(d.get("decision", "")).strip(),
                 "rationale": str(d.get("rationale", "")).strip()}
                for d in (parsed.get("decisions") or [])
                if isinstance(d, dict) and d.get("decision")
            ][:5],
            raw=content,
        )

    def _call_direct(self, cfg: Dict[str, str], messages: List[Dict[str, str]], timeout: int) -> Optional[str]:
        """Direct OpenAI-compatible HTTP POST."""
        url = cfg["endpoint"].rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/v1/chat/completions" if "/v1" not in url else url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        # Configurable max_tokens. Default 1500: 600 was the original cap;
        # long Russian/Cyrillic sessions with non-trivial decisions arrays
        # (3-4 chars/token) hit it mid-string and produced truncated JSON
        # noise. 1500 closes the brace in every observed case. If you see
        # repeated "recovered partial JSON" info-logs, raise via config.
        max_tokens = int(self._cfg("enricher_max_tokens", 1500) or 1500)
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Enricher direct call failed ({url}): {e}")
            return None

    def _call_via_auxiliary(self, messages: List[Dict[str, str]], timeout: int) -> Optional[str]:
        """Use Hermes auxiliary_client.call_llm — picks up provider/model from
        config.yaml (auxiliary.enrichment.*) or auto-fallback chain. This is
        the same path hermes-lcm uses for extraction."""
        try:
            from agent.auxiliary_client import call_llm
        except Exception as e:
            logger.debug(f"Enricher: auxiliary_client unavailable: {e}")
            return None
        max_tokens = int(self._cfg("enricher_max_tokens", 1500) or 1500)
        try:
            response = call_llm(
                task="enrichment",
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            return str(content) if content else None
        except Exception as e:
            logger.warning(f"Enricher: auxiliary_client.call_llm failed: {e}")
            return None

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Three-tier JSON recovery (Fix F):

        1. Strict parse. Most healthy responses land here.
        2. Find the first ``{...}`` block. Catches models that add prose
           around the JSON despite the prompt instruction.
        3. Truncation recovery. Long Russian/Cyrillic responses can hit
           the LLM's max_tokens limit mid-string, producing
           ``{"intent_label": "…длинный текст``. Close the open string and
           any open structural delimiters in reverse order so we recover
           at least the partial intent_label / topic_tags. Always better
           than returning None and degrading downstream tools.
        """
        if not text:
            return None
        text = text.strip()
        # Strip markdown fences if the model added them despite the instruction.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            pass
        # Tier 2: find the first {...} block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Tier 3: truncation recovery. Walk forward, track open structures,
        # then synthesize the missing closers. We only handle the common
        # case: truncated inside a string value or after a partial array.
        start = text.find("{")
        if start < 0:
            return None
        body = text[start:]
        stack: List[str] = []   # tracks '{' / '[' / '"'
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            top = stack[-1] if stack else None
            if top == '"':
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == '"':
                    stack.pop()
            else:
                if ch == '"':
                    stack.append('"')
                elif ch == "{":
                    stack.append("{")
                elif ch == "[":
                    stack.append("[")
                elif ch == "}" and top == "{":
                    stack.pop()
                elif ch == "]" and top == "[":
                    stack.pop()
            i += 1
        # Build a closing tail that matches the still-open structures.
        repaired = body
        # If we ended inside a string, close it. Also drop a trailing comma
        # or partial key (e.g. "intent_label":) that json refuses.
        if stack and stack[-1] == '"':
            repaired += '"'
            stack.pop()
        # Strip trailing commas / dangling colons.
        repaired = re.sub(r"[,:\s]+$", "", repaired)
        for opener in reversed(stack):
            repaired += "}" if opener == "{" else "]"
        try:
            result = json.loads(repaired)
        except Exception:
            return None
        # Mark recovered keys so the operator can scan logs for how often
        # truncation actually fires. INFO (not WARNING) — this is graceful
        # degradation, not a failure; bump enricher_max_tokens if it spams.
        try:
            recovered_keys = sorted(k for k in result.keys()) if isinstance(result, dict) else []
        except Exception:
            recovered_keys = []
        logger.info(
            "Enricher: recovered partial JSON via tier-3 truncation parser "
            "(repaired_len=%d, original_len=%d, keys=%s)",
            len(repaired), len(text), recovered_keys,
        )
        return result
