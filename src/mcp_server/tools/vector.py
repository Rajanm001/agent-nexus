"""Vector similarity search tool — FAISS or Pinecone."""

class VectorSearchTool:
    async def search(self, query: str, top_k: int = 10, namespace: str = "default"):
        # Production: embeds query → searches FAISS/Pinecone → returns chunks
        return []
