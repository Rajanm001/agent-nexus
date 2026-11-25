"""
AgentNexus Orchestrator — LangGraph-based stateful agent router.
Routes tasks to specialized agents, manages state, handles retries.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState
from src.agents.research_agent import ResearchAgent
from src.agents.document_agent import DocumentAgent
from src.agents.factcheck_agent import FactCheckAgent
from src.agents.writer_agent import WriterAgent
from src.agents.review_agent import ReviewAgent

logger = logging.getLogger(__name__)

MAX_REVISIONS = 3
QUALITY_THRESHOLD = 7.5


class Orchestrator:
    """
    LangGraph orchestrator that routes tasks through the agent pipeline.

    Flow:
        research → document_retrieval → fact_check → write → review
        If review score < threshold → back to write (max 3 revisions)
        If review passes → output final report
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.research_agent = ResearchAgent()
        self.document_agent = DocumentAgent()
        self.factcheck_agent = FactCheckAgent()
        self.writer_agent = WriterAgent()
        self.review_agent = ReviewAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for agent orchestration."""

        workflow = StateGraph(AgentState)

        # Add nodes (each node = one agent)
        workflow.add_node("research", self._run_research)
        workflow.add_node("retrieve_docs", self._run_document_retrieval)
        workflow.add_node("fact_check", self._run_fact_check)
        workflow.add_node("write", self._run_writer)
        workflow.add_node("review", self._run_review)

        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "fact_check")
        workflow.add_edge("fact_check", "write")
        workflow.add_edge("write", "review")

        # Conditional edge: review → END or review → write (revision)
        workflow.add_conditional_edges(
            "review",
            self._should_revise,
            {
                "revise": "write",
                "accept": END,
            },
        )

        # Compile with checkpointing for state persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _run_research(self, state: AgentState) -> AgentState:
        """Execute the research agent — web search + data extraction."""
        logger.info(f"[Research] Searching for: {state['query']}")
        results = await self.research_agent.execute(state["query"])
        return {**state, "research_results": results}

    async def _run_document_retrieval(self, state: AgentState) -> AgentState:
        """Execute the document agent — RAG retrieval from indexed docs."""
        logger.info(f"[DocAgent] Retrieving relevant documents")
        documents = await self.document_agent.retrieve(
            query=state["query"],
            context=state.get("research_results", []),
        )
        return {**state, "documents": documents}

    async def _run_fact_check(self, state: AgentState) -> AgentState:
        """Execute the fact-check agent — cross-verify claims."""
        logger.info(f"[FactCheck] Verifying claims from research")
        all_claims = state.get("research_results", []) + state.get("documents", [])
        verified = await self.factcheck_agent.verify(claims=all_claims)
        return {**state, "verified_claims": verified}

    async def _run_writer(self, state: AgentState) -> AgentState:
        """Execute the writer agent — generate cited report via Claude."""
        iteration = state.get("iteration", 0) + 1
        logger.info(f"[Writer] Generating report (iteration {iteration})")

        review_feedback = state.get("review_feedback")
        draft = await self.writer_agent.write(
            query=state["query"],
            verified_claims=state["verified_claims"],
            documents=state.get("documents", []),
            previous_draft=state.get("draft_report"),
            feedback=review_feedback,
        )
        return {**state, "draft_report": draft, "iteration": iteration}

    async def _run_review(self, state: AgentState) -> AgentState:
        """Execute the review agent — quality gate with scoring."""
        logger.info(f"[Review] Evaluating draft quality")
        review = await self.review_agent.evaluate(
            draft=state["draft_report"],
            query=state["query"],
            sources=state.get("verified_claims", []),
        )
        return {
            **state,
            "review_score": review.score,
            "review_feedback": review.feedback,
            "final_report": state["draft_report"] if review.score >= QUALITY_THRESHOLD else None,
            "citations": review.citations,
        }

    def _should_revise(self, state: AgentState) -> Literal["revise", "accept"]:
        """Decide whether to revise the draft or accept it."""
        score = state.get("review_score", 0)
        iteration = state.get("iteration", 0)

        if score >= QUALITY_THRESHOLD:
            logger.info(f"[Review] ACCEPTED — score {score}/10")
            return "accept"

        if iteration >= MAX_REVISIONS:
            logger.warning(f"[Review] Max revisions reached — accepting score {score}/10")
            return "accept"

        logger.info(f"[Review] REVISION NEEDED — score {score}/10, iteration {iteration}")
        return "revise"

    async def run(self, query: str, thread_id: str = "default") -> AgentState:
        """Execute the full agent pipeline for a given query."""
        initial_state: AgentState = {
            "query": query,
            "research_results": [],
            "documents": [],
            "claims": [],
            "verified_claims": [],
            "draft_report": "",
            "review_score": 0.0,
            "review_feedback": None,
            "final_report": None,
            "citations": [],
            "iteration": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.graph.ainvoke(initial_state, config=config)

        logger.info(
            f"[Orchestrator] Complete — score: {final_state['review_score']}/10, "
            f"iterations: {final_state['iteration']}, "
            f"citations: {len(final_state.get('citations', []))}"
        )
        return final_state
