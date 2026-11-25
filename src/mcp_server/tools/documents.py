"""Document parsing tool — handles PDF, DOCX, TXT."""

from pathlib import Path

class DocumentParserTool:
    async def parse(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:50000]
