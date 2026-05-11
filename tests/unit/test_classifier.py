"""Unit tests for classifier signal functions (Stage 6)."""

import pytest

from hermes_mneme import classifier as C


# --- is_question -----------------------------------------------------------

def test_is_question_trailing_qmark():
    assert C.is_question("what is happening?") is True
    assert C.is_question("ok.  ") is False
    assert C.is_question("") is False
    assert C.is_question(None) is False


def test_is_question_en_question_word_no_qmark():
    assert C.is_question("Why did it crash") is True
    assert C.is_question("How does this work") is True
    assert C.is_question("What about the cat") is True
    # not a question word at start
    assert C.is_question("the cat ate breakfast") is False


def test_is_question_ru_question_word_no_qmark():
    assert C.is_question("почему упало") is True
    assert C.is_question("Что произошло") is True
    assert C.is_question("откуда это взялось") is True
    assert C.is_question("просто рассказ без вопроса") is False


# --- extract_entities ------------------------------------------------------

def test_extract_entities_proper_nouns():
    ents = C.extract_entities("The Hermes plugin uses Mnemosyne for storage.")
    low = {e.lower() for e in ents}
    assert "hermes" in low
    assert "mnemosyne" in low


def test_extract_entities_backticks_and_quotes():
    text = "Calling `memory_recall` with query \"Барсик кот\" returned 'duplicates'."
    ents = C.extract_entities(text)
    assert "memory_recall" in ents
    assert "Барсик кот" in ents
    assert "duplicates" in ents


def test_extract_entities_dedup_case_insensitive():
    ents = C.extract_entities("Hermes is great. Look at Hermes again.")
    low = [e.lower() for e in ents]
    assert low.count("hermes") == 1


def test_extract_entities_filters_stopwords():
    ents = C.extract_entities("The quick brown Fox jumps.")
    low = {e.lower() for e in ents}
    # 'the' is a stopword (capitalized at start), 'fox' is a real proper-ish noun
    assert "the" not in low
    assert "fox" in low


# --- classify_question_about_output ----------------------------------------

def test_clarification_requires_question_and_entity():
    last_entities = ["Hermes", "memory_recall"]
    # question + entity match → True
    assert C.classify_question_about_output(
        "why did Hermes return so many duplicates?", last_entities
    ) is True
    # question but no entity match → False
    assert C.classify_question_about_output(
        "why is the sky blue?", last_entities
    ) is False
    # entity match but not a question → False
    assert C.classify_question_about_output(
        "Hermes is fine.", last_entities
    ) is False
    # empty entities → False
    assert C.classify_question_about_output("why?", []) is False


def test_clarification_works_without_qmark():
    """Real chat traffic drops the '?'. Question words at sentence start
    still mark the message as a question."""
    assert C.classify_question_about_output(
        "Почему Hermes удалил столько записей",
        ["Hermes"]
    ) is True


# --- resolve_intent priority chain ----------------------------------------

def test_resolve_intent_explicit_switch_wins():
    sig = C.ClassifierSignals(
        explicit_switch=True, entity_contradiction=True,
        embedding_drift=0.9, question_about_output=True
    )
    assert C.resolve_intent(sig) == C.INTENT_SWITCH


def test_resolve_intent_clarification_below_drift():
    sig = C.ClassifierSignals(embedding_drift=0.4, question_about_output=True)
    # drift > 0.35 → NEW_TASK wins over clarification
    assert C.resolve_intent(sig) == C.INTENT_NEW_TASK


def test_resolve_intent_clarification_when_no_other_signal():
    sig = C.ClassifierSignals(embedding_drift=0.1, question_about_output=True)
    assert C.resolve_intent(sig) == C.INTENT_CLARIFICATION


def test_resolve_intent_default_continuation():
    sig = C.ClassifierSignals()
    assert C.resolve_intent(sig) == C.INTENT_CONTINUATION
