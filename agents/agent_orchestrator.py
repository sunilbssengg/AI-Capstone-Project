"""
Agent Orchestrator — coordinates the autonomous multi-agent RAG pipeline.

Flow (matches the project spec):
  1. PLAN     - validate the question, decide retrieval parameters
  2. RETRIEVE - RetrievalAgent embeds the query + runs cosine similarity search
  3. REASON   - ReasoningAgent calls Gemini 2.5 with retrieved context
  4. RESPOND  - package the grounded answer + sources + trace for the UI

Two execution modes are provided:
  - `run()`            : deterministic pipeline (fast, predictable, used by
                          default in the Streamlit app).
  - `run_as_tool_agent`: wraps each step as a LangChain Tool and lets a
                          Gemini-powered LangChain agent autonomously decide
                          whether/how to call them (true agentic reasoning,
                          useful for more complex/multi-hop questions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.exceptions import AppBaseException, GuardrailViolation
from core.guardrails import validate_question
from utils.logger import logger

from agents.retrieval_agent import RetrievalAgent, RetrievedChunk
from agents.reasoning_agent import ReasoningAgent, ReasonedAnswer


@dataclass
class AgentTraceStep:
    step: str
    detail: str


@dataclass
class OrchestratorResult:
    answer: str
    sources: List[str]
    grounded: bool
    trace: List[AgentTraceStep] = field(default_factory=list)


class AgentOrchestrator:
    """Coordinates PLAN -> RETRIEVE -> REASON -> RESPOND across agents."""

    def __init__(self) -> None:
        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = ReasoningAgent()

    def run(self, question: str) -> OrchestratorResult:
        trace: List[AgentTraceStep] = []

        # 1. PLAN — validate + sanitize the input (guardrail)
        try:
            clean_question = validate_question(question)
            trace.append(AgentTraceStep("plan", "Question validated by guardrails."))
        except GuardrailViolation as exc:
            trace.append(AgentTraceStep("plan", f"Blocked: {exc.message}"))
            return OrchestratorResult(answer=f"⚠️ {exc.message}", sources=[], grounded=False, trace=trace)

        # 2. RETRIEVE
        try:
            chunks: List[RetrievedChunk] = self.retrieval_agent.run(clean_question)
            trace.append(AgentTraceStep(
                "retrieve",
                f"Retrieved {len(chunks)} chunk(s) via cosine similarity search."
            ))
        except AppBaseException as exc:
            logger.error(f"Retrieval failed: {exc.message}")
            trace.append(AgentTraceStep("retrieve", f"Failed: {exc.message}"))
            return OrchestratorResult(
                answer="⚠️ I couldn't search the document store. Please try again.",
                sources=[], grounded=False, trace=trace,
            )

        if not chunks:
            trace.append(AgentTraceStep("respond", "No relevant content found."))
            return OrchestratorResult(
                answer="I couldn't find anything relevant to your question in the "
                       "uploaded documents. Try rephrasing, or upload a document "
                       "that covers this topic.",
                sources=[], grounded=False, trace=trace,
            )

        # 3. REASON
        try:
            result: ReasonedAnswer = self.reasoning_agent.run(clean_question, chunks)
            trace.append(AgentTraceStep(
                "reason",
                f"Generated answer with Gemini 2.5 (grounded={result.grounded})."
            ))
        except AppBaseException as exc:
            logger.error(f"Reasoning failed: {exc.message}")
            trace.append(AgentTraceStep("reason", f"Failed: {exc.message}"))
            return OrchestratorResult(
                answer="⚠️ I couldn't generate an answer right now. Please try again shortly.",
                sources=[], grounded=False, trace=trace,
            )

        # 4. RESPOND
        trace.append(AgentTraceStep("respond", "Final answer packaged with sources."))
        return OrchestratorResult(
            answer=result.answer,
            sources=result.sources,
            grounded=result.grounded,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Optional: true tool-calling autonomous agent (LangChain AgentExecutor)
    # ------------------------------------------------------------------
    def run_as_tool_agent(self, question: str) -> str:
        """
        Let a Gemini-powered LangChain agent autonomously decide how to use
        the retrieval tool (useful for multi-hop or exploratory questions).
        Falls back gracefully if agent construction fails.
        """
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import Tool
        from services.llm_service import get_llm

        clean_question = validate_question(question)

        retrieve_tool = Tool.from_function(
            func=self.retrieval_agent.as_tool_fn(),
            name="search_enterprise_documents",
            description=(
                "Search the uploaded enterprise documents for content relevant "
                "to a question. Input should be a natural language query."
            ),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an autonomous enterprise document assistant. "
                       "Use the search_enterprise_documents tool to find grounded "
                       "context before answering. Never answer from memory alone. "
                       "If the tool returns nothing relevant, say so plainly."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(get_llm(), [retrieve_tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[retrieve_tool], verbose=False,
                                  max_iterations=4, handle_parsing_errors=True)

        result = executor.invoke({"input": clean_question})
        return result.get("output", "I couldn't produce an answer.")
