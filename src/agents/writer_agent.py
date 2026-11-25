"""Report writer agent — generates cited research reports using Claude API."""

import logging
from anthropic import AsyncAnthropic
from src.agents.state import VerifiedClaim, Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research writer. Write clear, well-structured reports
based on verified facts. Every factual statement must have an inline citation.
Use markdown formatting. Be direct — no filler, no hedging, no unnecessary
qualifiers. If a fact was flagged as unverified, say so explicitly."""

class WriterAgent:
    def __init__(self):
        self.client = AsyncAnthropic()
        self.model = "claude-sonnet-4-20250514"

    async def write(
        self,
        query: str,
        verified_claims: list[VerifiedClaim],
        documents: list[Document],
        previous_draft: str | None = None,
        feedback: str | None = None,
    ) -> str:
        """Generate a cited research report from verified claims."""
        claims_text = self._format_claims(verified_claims)
        docs_text = self._format_documents(documents)

        revision_context = ""
        if previous_draft and feedback:
            revision_context = (
                f"\n\nPREVIOUS DRAFT (needs revision):\n{previous_draft}\n\n"
                f"REVIEWER FEEDBACK:\n{feedback}\n\n"
                f"Revise the draft to address the feedback. Keep what works."
            )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Research question: {query}\n\n"
                    f"VERIFIED FACTS:\n{claims_text}\n\n"
                    f"SUPPORTING DOCUMENTS:\n{docs_text}"
                    f"{revision_context}\n\n"
                    f"Write a comprehensive research report with inline citations."
                ),
            }],
        )
        return response.content[0].text

    def _format_claims(self, claims: list[VerifiedClaim]) -> str:
        lines = []
        for i, vc in enumerate(claims, 1):
            status = "VERIFIED" if vc.verified else "UNVERIFIED"
            sources = ", ".join(vc.verification_sources) if vc.verification_sources else "pending"
            lines.append(f"[{i}] [{status}] {vc.claim.text} (sources: {sources})")
        return "\n".join(lines) or "No claims available."

    def _format_documents(self, docs: list[Document]) -> str:
        lines = []
        for d in docs[:10]:
            lines.append(f"- [{d.source}] {d.content[:300]}...")
        return "\n".join(lines) or "No documents retrieved."
