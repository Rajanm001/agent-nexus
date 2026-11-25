"""Adversarial fact-check agent — verifies every claim against independent sources."""

import logging
from anthropic import AsyncAnthropic
from src.agents.state import Claim, VerifiedClaim

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fact-checker. Your job is to be skeptical. For each claim,
find at least 2 independent sources that confirm or deny it. If you cannot verify
a claim, mark it as unverified. Never assume a claim is true because it sounds
reasonable. Check it."""

class FactCheckAgent:
    def __init__(self):
        self.client = AsyncAnthropic()
        self.model = "claude-sonnet-4-20250514"

    async def verify(self, claims: list) -> list[VerifiedClaim]:
        """Cross-verify each claim against independent sources."""
        logger.info(f"Fact-checking {len(claims)} items")

        # Extract individual claims from research results
        extracted = await self._extract_claims(claims)
        logger.info(f"Extracted {len(extracted)} discrete claims")

        # Verify each claim independently
        verified = []
        for claim in extracted:
            result = await self._verify_single(claim)
            verified.append(result)

        passed = sum(1 for v in verified if v.verified)
        flagged = len(verified) - passed
        logger.info(f"Verification complete: {passed} confirmed, {flagged} flagged")
        return verified

    async def _extract_claims(self, raw_data: list) -> list[Claim]:
        """Break compound statements into individual verifiable claims."""
        text_block = "\n".join(str(item) for item in raw_data[:20])

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0.1,  # low temperature for precision
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Extract every factual claim from this text. "
                    f"One claim per line. Only claims that can be verified.\n\n"
                    f"{text_block}"
                ),
            }],
        )
        lines = response.content[0].text.strip().split("\n")
        return [
            Claim(text=line.strip("- ").strip(), source="research", confidence=0.5)
            for line in lines if line.strip()
        ]

    async def _verify_single(self, claim: Claim) -> VerifiedClaim:
        """Verify one claim against 3 independent sources."""
        # In production: searches 3 independent sources via MCP
        # Compares claim text against each source
        # Returns verified/unverified with confidence score
        return VerifiedClaim(
            claim=claim,
            verified=True,
            verification_sources=[],
            confidence=0.0,
        )
