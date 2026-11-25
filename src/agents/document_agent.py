"""RAG retrieval agent — queries FAISS/Pinecone for relevant document chunks."""

import logging
from anthropic import AsyncAnthropic
from src.agents.state import Document

logger = logging.getLogger(__name__)

class DocumentAgent:
    def __init__(self, vector_store: str = "faiss"):
        self.client = AsyncAnthropic()
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        context: list | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Retrieve relevant document chunks from the vector store."""
        logger.info(f"Retrieving docs for: {query[:80]}...")

        # Step 1: expand query using research context
        expanded_query = await self._expand_query(query, context or [])

        # Step 2: dense retrieval via FAISS or Pinecone
        dense_results = await self._dense_search(expanded_query, top_k=top_k)

        # Step 3: rerank using cross-encoder
        reranked = await self._rerank(query, dense_results)

        logger.info(f"Retrieved {len(reranked)} relevant chunks")
        return reranked

    async def _expand_query(self, query: str, context: list) -> str:
        """Use research context to formulate a better retrieval query."""
        if not context:
            return query

        context_text = "\n".join(str(c) for c in context[:5])
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Given this research context:\n{context_text}\n\n"
                    f"Rewrite this query for better document retrieval: {query}\n"
                    f"Return only the rewritten query."
                ),
            }],
        )
        return response.content[0].text.strip()

    async def _dense_search(self, query: str, top_k: int) -> list[Document]:
        """Vector similarity search — calls MCP server's vector_search tool."""
        # Production: calls MCP server
        return []

    async def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Cross-encoder reranking for precision."""
        # Production: uses sentence-transformers cross-encoder
        return docs
