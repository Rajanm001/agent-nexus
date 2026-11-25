"""Shared state definition for AgentNexus multi-agent pipeline."""

from __future__ import annotations
from typing import TypedDict
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float


@dataclass
class Document:
    content: str
    source: str
    chunk_id: str
    similarity_score: float


@dataclass
class Claim:
    text: str
    source: str
    confidence: float


@dataclass
class VerifiedClaim:
    claim: Claim
    verified: bool
    verification_sources: list[str]
    confidence: float


@dataclass
class Citation:
    text: str
    source_url: str
    source_title: str
    claim_id: int


@dataclass
class ReviewResult:
    score: float
    feedback: str | None
    citations: list[Citation]


class AgentState(TypedDict):
    """Shared state passed between all agents in the pipeline."""
    query: str
    research_results: list[SearchResult]
    documents: list[Document]
    claims: list[Claim]
    verified_claims: list[VerifiedClaim]
    draft_report: str
    review_score: float
    review_feedback: str | None
    final_report: str | None
    citations: list[Citation]
    iteration: int
