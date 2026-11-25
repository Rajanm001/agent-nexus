"""
AgentNexus MCP Server — Model Context Protocol server for standardized tool access.
Provides database, search, document, vector, cache, and notification tools.
"""

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from src.mcp_server.tools.database import DatabaseTool
from src.mcp_server.tools.search import WebSearchTool
from src.mcp_server.tools.documents import DocumentParserTool
from src.mcp_server.tools.vector import VectorSearchTool
from src.mcp_server.tools.notify import NotificationTool

logger = logging.getLogger(__name__)

server = Server("agentnexus-tools")

# Initialize tools
db_tool = DatabaseTool()
search_tool = WebSearchTool()
doc_tool = DocumentParserTool()
vector_tool = VectorSearchTool()
notify_tool = NotificationTool()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name="database_query",
            description="Execute read-only SQL queries against PostgreSQL",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"},
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="search_web",
            description="Search the web using Tavily API for real-time information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_document",
            description="Parse and extract text from PDF, DOCX, or TXT files",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to document"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="vector_search",
            description="Semantic similarity search using FAISS or Pinecone",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text"},
                    "top_k": {"type": "integer", "default": 10},
                    "namespace": {"type": "string", "default": "default"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="cache_result",
            description="Store or retrieve cached results from Redis",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string", "description": "Value to cache (omit for read)"},
                    "ttl": {"type": "integer", "default": 3600},
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="send_notification",
            description="Send email notification for completed reports",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to the appropriate handler."""
    logger.info(f"[MCP] Tool called: {name} with args: {list(arguments.keys())}")

    try:
        match name:
            case "database_query":
                result = await db_tool.execute(arguments["sql"])
            case "search_web":
                result = await search_tool.search(
                    arguments["query"],
                    max_results=arguments.get("max_results", 5),
                )
            case "read_document":
                result = await doc_tool.parse(arguments["path"])
            case "vector_search":
                result = await vector_tool.search(
                    arguments["query"],
                    top_k=arguments.get("top_k", 10),
                    namespace=arguments.get("namespace", "default"),
                )
            case "cache_result":
                if "value" in arguments:
                    result = await cache_tool.set(
                        arguments["key"],
                        arguments["value"],
                        ttl=arguments.get("ttl", 3600),
                    )
                else:
                    result = await cache_tool.get(arguments["key"])
            case "send_notification":
                result = await notify_tool.send(
                    to=arguments["to"],
                    subject=arguments["subject"],
                    body=arguments["body"],
                )
            case _:
                return [TextContent(text=f"Unknown tool: {name}")]

        return [TextContent(text=json.dumps(result, default=str))]

    except Exception as e:
        logger.error(f"[MCP] Tool error ({name}): {e}")
        return [TextContent(text=json.dumps({"error": str(e)}))]


async def main():
    """Start the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
