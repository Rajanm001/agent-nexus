"""Web search tool wrapping Tavily API."""

class WebSearchTool:
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        # Production: calls tavily-python client
        return []
