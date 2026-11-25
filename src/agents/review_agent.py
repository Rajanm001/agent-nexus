"""Quality gate agent — scores reports and provides revision feedback."""

import logging
from dataclasses import dataclass
from anthropic import AsyncAnthropic
from src.agents.state import Citation, ReviewResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a report reviewer. Score the report on a 1-10 scale across:
- Factual accuracy (are claims supported by cited sources?)
- Citation coverage (does every factual claim have a citation?)
- Completeness (does it address the original question fully?)
- Clarity (is it well-structured and readable?)

If the overall score is below 7.5, provide specific feedback on what needs fixing.
Be direct. Name the paragraphs or claims that need work."""

class ReviewAgent:
    def __init__(self):
        self.client = AsyncAnthropic()
        self.model = "claude-sonnet-4-20250514"

    async def evaluate(
        self,
        draft: str,
        query: str,
        sources: list,
    ) -> ReviewResult:
        """Score the draft and return structured feedback."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Draft report:\n{draft}\n\n"
                    f"Number of verified sources: {len(sources)}\n\n"
                    f"Score this report (1-10) and provide feedback if below 7.5.\n"
                    f"First line: just the number. Rest: feedback if needed."
                ),
            }],
        )

        text = response.content[0].text.strip()
        lines = text.split("\n", 1)

        try:
            score = float(lines[0].strip())
        except ValueError:
            score = 5.0

        feedback = lines[1].strip() if len(lines) > 1 else None
        logger.info(f"Review score: {score}/10")

        return ReviewResult(score=score, feedback=feedback, citations=[])
