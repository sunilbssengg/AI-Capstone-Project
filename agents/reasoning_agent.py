"""
Reasoning Agent — responsible for the "generate" step of the RAG flow:
takes retrieved context + the user question, calls the Gemini 2.5 LLM to
produce a grounded answer, and applies output guardrails before returning
the final answer to the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agents.retrieval_agent import RetrievedChunk
from core.exceptions import LLMGenerationError
from core.guardrails import apply_output_guardrails
from services.llm_service import generate_answer
from utils.logger import logger


@dataclass
class ReasonedAnswer:
    answer: str
    sources: List[str]
    grounded: bool


class ReasoningAgent:
    """A focused agent whose only job is grounded answer generation."""

    name = "reasoning_agent"
    description = (
        "Generates a grounded, factual answer to a question using retrieved "
        "enterprise document context and the Gemini 2.5 LLM, applying "
        "guardrails to reduce hallucination."
    )

    def run(self, question: str, chunks: List[RetrievedChunk]) -> ReasonedAnswer:
        context_texts = [c.text for c in chunks]
        try:
            raw_answer = generate_answer(question, context_texts)
        except LLMGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMGenerationError("Reasoning step failed.", str(exc)) from exc

        final_answer = apply_output_guardrails(raw_answer, context_texts)
        grounded = final_answer == raw_answer  # guardrail appends a note if ungrounded

        sources = sorted({c.source for c in chunks})
        logger.info(f"ReasoningAgent produced answer (grounded={grounded}, "
                    f"sources={sources}).")
        return ReasonedAnswer(answer=final_answer, sources=sources, grounded=grounded)
