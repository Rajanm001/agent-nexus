"""FastAPI application — HTTP + WebSocket gateway for AgentNexus."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.agents.orchestrator import Orchestrator
from src.api.routes.research import router as research_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgentNexus starting up")
    app.state.orchestrator = Orchestrator()
    yield
    logger.info("AgentNexus shutting down")

app = FastAPI(
    title="AgentNexus",
    description="Multi-agent research system with MCP backbone",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "running", "agents": 6, "mcp_tools": 6}

@app.websocket("/ws/status")
async def agent_status(websocket: WebSocket):
    """Real-time agent status updates via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"status": "connected"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
