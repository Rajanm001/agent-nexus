"""Research endpoint — accepts a query, runs the agent pipeline, returns a report."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class ResearchRequest(BaseModel):
    query: str
    thread_id: str = "default"

class ResearchResponse(BaseModel):
    report: str | None
    score: float
    iterations: int
    citations_count: int

@router.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest, request: Request):
    orchestrator = request.app.state.orchestrator
    result = await orchestrator.run(query=req.query, thread_id=req.thread_id)
    return ResearchResponse(
        report=result.get("final_report"),
        score=result.get("review_score", 0),
        iterations=result.get("iteration", 0),
        citations_count=len(result.get("citations", [])),
    )
