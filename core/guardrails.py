"""
Guardrails: lightweight, dependency-free safety and reliability checks that
run BEFORE a query reaches the LLM (input guardrails) and AFTER a response
is generated (output guardrails), reducing prompt injection, unsafe
content, and hallucinated / ungrounded answers.

These are intentionally simple, transparent, rule-based checks rather than
a black-box moderation model, so behaviour is easy to audit and extend.
"""
from __future__ import annotations

import re
from typing import List

from core.config import settings
from core.exceptions import GuardrailViolation

# --- Patterns that indicate an attempt to override system instructions ---
_PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|above|prior) instructions",
    r"disregard (all|any|the)?\s*(previous|above|prior)",
    r"you are now",
    r"act as (an?|the) (unfiltered|jailbroken|dan)",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions)",
    r"pretend (you|to) (are|be)",
    r"override (your|the) (rules|guardrails|policy)",
]

# --- Minimal profanity / unsafe-request keyword list (illustrative, not exhaustive) ---
_UNSAFE_KEYWORDS = [
    "hack into", "malware", "ransomware", "bomb making", "kill someone",
    "child sexual", "credit card dump", "steal password",
]


def validate_question(question: str) -> str:
    """Validate and sanitize a user question. Raises GuardrailViolation on failure."""
    if not question or not question.strip():
        raise GuardrailViolation("Question cannot be empty.")

    question = question.strip()

    if len(question) > settings.MAX_QUESTION_LENGTH:
        raise GuardrailViolation(
            f"Question too long ({len(question)} chars). "
            f"Limit is {settings.MAX_QUESTION_LENGTH} characters."
        )

    if settings.ENABLE_PROMPT_INJECTION_FILTER:
        lowered = question.lower()
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                raise GuardrailViolation(
                    "Your question looks like it is trying to override system "
                    "instructions, which isn't allowed."
                )

    if settings.ENABLE_PROFANITY_FILTER:
        lowered = question.lower()
        for word in _UNSAFE_KEYWORDS:
            if word in lowered:
                raise GuardrailViolation(
                    "This question was blocked by the safety filter."
                )

    return question


def check_groundedness(answer: str, source_chunks: List[str]) -> bool:
    """
    Heuristic hallucination guard: verifies that a meaningful proportion of
    the answer's vocabulary overlaps with the retrieved source chunks. This
    is a cheap proxy for "is the answer actually grounded in the retrieved
    context" and flags answers that look fabricated / off-topic.
    """
    if not source_chunks:
        return False

    def tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z]{4,}", text.lower()))

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return False

    context_tokens = set()
    for chunk in source_chunks:
        context_tokens |= tokenize(chunk)

    overlap = answer_tokens & context_tokens
    overlap_ratio = len(overlap) / max(len(answer_tokens), 1)

    return overlap_ratio >= settings.GROUNDEDNESS_MIN_OVERLAP


def apply_output_guardrails(answer: str, source_chunks: List[str]) -> str:
    """
    Apply post-generation guardrails. If the answer doesn't look grounded
    in the retrieved context, append a transparency disclaimer rather than
    silently presenting a possibly hallucinated answer as fact.
    """
    if not answer or not answer.strip():
        return (
            "I wasn't able to generate a response. Please try rephrasing "
            "your question."
        )

    no_answer_markers = ["i don't know", "not mentioned", "no information", "cannot find"]
    if any(marker in answer.lower() for marker in no_answer_markers):
        return answer

    if not check_groundedness(answer, source_chunks):
        answer += (
            "\n\n*Note: This answer could not be fully verified against the "
            "retrieved document content. Please double-check against the "
            "source document.*"
        )

    return answer
