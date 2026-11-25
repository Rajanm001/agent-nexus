"""PostgreSQL read-only query tool for MCP server."""

class DatabaseTool:
    async def execute(self, sql: str) -> dict:
        if not sql.strip().upper().startswith("SELECT"):
            return {"error": "only SELECT queries allowed"}
        # Production: runs against asyncpg connection pool
        return {"rows": [], "count": 0}
