import re
from dataclasses import dataclass
from typing import List, Optional

# Default patterns for switch phrases
DEFAULT_SWITCH_PATTERNS = [
    r"let's switch to",
    r"forget that",
    r"new topic",
    r"instead of that",
    r"давай переключимся",
    r"забудь это",
    r"новая тема"
]

# Question-word triggers (start of sentence). When a message starts with one of
# these, treat it as a question even without a trailing '?'. Real chat traffic
# routinely drops the punctuation.
EN_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will",
}
RU_QUESTION_WORDS = {
    "что", "почему", "как", "когда", "где", "кто", "откуда", "зачем", "какой",
    "какая", "какое", "какие", "сколько", "куда", "чей", "чья",
}

# Pre-compiled patterns for entity extraction.
_PROPER_NOUN_RE = re.compile(r"\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9_-]{2,}\b")
_BACKTICK_RE = re.compile(r"`([^`\n]{2,40})`")
_DQUOTE_RE = re.compile(r"\"([^\"\n]{2,40})\"")
_SQUOTE_RE = re.compile(r"'([^'\n]{2,40})'")

# Stopwords to drop from proper-noun matches (start-of-sentence common words).
_PROPER_NOUN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that",
    "these", "those", "it", "its", "we", "you", "your", "i", "my",
    "и", "или", "но", "это", "эта", "этот", "эти", "тот", "та", "те",
    "мы", "вы", "ваш", "я", "мой", "он", "она", "они",
}


@dataclass
class ClassifierSignals:
    explicit_switch: bool = False
    entity_contradiction: bool = False
    embedding_drift: float = 0.0
    question_about_output: bool = False


def classify_explicit_switch(message: str, patterns: List[str] = None) -> bool:
    """Check for explicit switch phrases."""
    if patterns is None:
        patterns = DEFAULT_SWITCH_PATTERNS
    for p in patterns:
        if re.search(p, message, re.IGNORECASE):
            return True
    return False


def classify_entity_contradiction(message: str, state_entities: List[str]) -> bool:
    """Check if message negates an entity in current goal."""
    # Simple negation check
    negations = ["not", "no", "never", "don't", "не", "нет"]
    msg_lower = message.lower()

    for entity in state_entities:
        if entity.lower() in msg_lower:
            # Check if negation word is nearby
            for neg in negations:
                if neg in msg_lower:
                    # Very simple check: if both exist, assume contradiction possible
                    return True
    return False


def classify_embedding_drift(message: str, segment_centroid: Optional[List[float]]) -> float:
    """
    Placeholder for embedding drift calculation.
    Returns a score 0.0 to 1.0.
    Requires embedding comparison, skipping for MVP.
    """
    # TODO: Implement cosine distance to segment centroid
    return 0.0


def is_question(message: str) -> bool:
    """Question detection that doesn't require a trailing '?'.

    True if either:
      * the message ends with '?' (after trimming), or
      * the first non-trivial token is an EN/RU question word.
    """
    if not message:
        return False
    stripped = message.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    # First token (lowercase, strip punctuation).
    first = re.split(r"\s+", stripped, maxsplit=1)[0].lower().strip(".,!:;-—()[]{}'\"")
    if not first:
        return False
    return first in EN_QUESTION_WORDS or first in RU_QUESTION_WORDS


def extract_entities(text: str, max_entities: int = 30) -> List[str]:
    """Deterministic, no-LLM entity extraction from a free-form text.

    Captures three kinds of tokens that tend to be referenced in follow-up
    questions:
      * Proper nouns / CamelCase identifiers (RU + EN, capitalized, ≥3 chars).
      * Backtick-quoted code-style tokens.
      * Quoted strings (single + double quotes), 2..40 chars.

    Returns a deduplicated list (case-insensitive dedup, original case kept
    for the first occurrence), bounded by max_entities.
    """
    if not text:
        return []

    found: List[str] = []
    seen_lower = set()

    def _push(tok: str) -> None:
        tok = tok.strip().strip(".,;:!?()[]{}<>")
        if not tok or len(tok) < 2:
            return
        low = tok.lower()
        if low in _PROPER_NOUN_STOPWORDS:
            return
        if low in seen_lower:
            return
        seen_lower.add(low)
        found.append(tok)

    for m in _BACKTICK_RE.findall(text):
        _push(m)
    for m in _DQUOTE_RE.findall(text):
        _push(m)
    for m in _SQUOTE_RE.findall(text):
        _push(m)
    for m in _PROPER_NOUN_RE.findall(text):
        _push(m)

    return found[:max_entities]


def classify_question_about_output(message: str, last_assistant_entities: List[str]) -> bool:
    """Message is a question (per is_question) AND references at least one
    entity that appeared in the last assistant turn.

    Per spec (lines 439-440, 473-475): the entity list must be from the LAST
    assistant message specifically — not the running aggregate of session
    entities. The caller is responsible for passing the right list.
    """
    if not last_assistant_entities:
        return False
    if not is_question(message):
        return False
    msg_lower = message.lower()
    for entity in last_assistant_entities:
        if not entity:
            continue
        if entity.lower() in msg_lower:
            return True
    return False


# --- Policy Resolver ---

# Intent types
INTENT_CONTINUATION = "CONTINUATION"
INTENT_SWITCH = "SWITCH"
INTENT_NEW_TASK = "NEW_TASK"
INTENT_CLARIFICATION = "CLARIFICATION"


def resolve_intent(signals: ClassifierSignals) -> str:
    """
    Combines signals using priority chain.
    Priority:
    1. Explicit switch
    2. Entity contradiction
    3. Embedding drift (if high)
    4. Question about output
    5. Default: Continuation
    """
    if signals.explicit_switch:
        return INTENT_SWITCH
    if signals.entity_contradiction:
        return INTENT_SWITCH
    if signals.embedding_drift > 0.35:  # Threshold from spec
        return INTENT_NEW_TASK
    if signals.question_about_output:
        return INTENT_CLARIFICATION

    return INTENT_CONTINUATION
