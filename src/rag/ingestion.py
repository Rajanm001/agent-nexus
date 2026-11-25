"""Document ingestion pipeline — chunking, embedding, and indexing."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest(self, file_path: str, namespace: str = "default") -> int:
        """Ingest a document: parse → chunk → embed → store."""
        path = Path(file_path)
        logger.info(f"Ingesting {path.name} into namespace '{namespace}'")

        text = await self._parse(path)
        chunks = self._chunk(text)
        embeddings = await self._embed(chunks)
        count = await self._store(embeddings, namespace)

        logger.info(f"Indexed {count} chunks from {path.name}")
        return count

    async def _parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            import docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return path.read_text(encoding="utf-8")

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    async def _embed(self, chunks: list[str]) -> list[dict]:
        # Production: calls OpenAI text-embedding-3-large
        return [{"text": c, "embedding": []} for c in chunks]

    async def _store(self, embeddings: list[dict], namespace: str) -> int:
        # Production: writes to FAISS index or Pinecone
        return len(embeddings)
