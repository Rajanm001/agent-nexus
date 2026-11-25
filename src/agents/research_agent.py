"""Web search and data extraction agent using Tavily API."""

import logging
from anthropic import AsyncAnthropic
from src.agents.state import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research specialist. Your job is to search the web,
extract relevant facts, and return structured results. Focus on recent, credible
sources. Always note the source URL for citation."""

class ResearchAgent:
    def __init__(self):
        self.client = AsyncAnthropic()
        self.model = "claude-sonnet-4-20250514"

    async def execute(self, query: str) -> list[SearchResult]:
        """Search the web and extract structured results."""
        logger.info(f"Researching: {query}")

        # Step 1: generate search queries from the user's question
        search_queries = await self._generate_search_queries(query)

        # Step 2: run each query through Tavily
        all_results = []
        for sq in search_queries:
            results = await self._search(sq)
            all_results.extend(results)

        # Step 3: deduplicate and rank by relevance
        ranked = self._deduplicate_and_rank(all_results, query)
        logger.info(f"Found {len(ranked)} unique results")
        return ranked[:15]

    async def _generate_search_queries(self, query: str) -> list[str]:
        """Break a complex question into 3-4 targeted search queries."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Generate 3-4 specific web search queries for: {query}\n"
                           f"Return one query per line, nothing else."
            }],
        )
        lines = response.content[0].text.strip().split("\n")
        return [line.strip("- ").strip() for line in lines if line.strip()]

    async def _search(self, query: str) -> list[SearchResult]:
        """Execute a single search via MCP tool call."""
        # In production this calls the MCP server's search_web tool
        # Stubbed here for the repository skeleton
        return []

    def _deduplicate_and_rank(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """Remove duplicate URLs and sort by relevance score."""
        seen_urls = set()
        unique = []
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)
        return sorted(unique, key=lambda x: x.relevance_score, reverse=True)
