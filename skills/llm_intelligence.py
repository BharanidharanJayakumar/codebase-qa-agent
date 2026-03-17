"""
LLM-powered intelligence for retrieval reranking.

Provides LLM-based reranking of BM25 code chunks for better relevance.
Used as a fallback path when the navigator + targeted file reading isn't enough.
Rate-limited for Groq free tier with exponential backoff.
"""

import asyncio
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ── Pydantic Schema ─────────────────────────────────────────────────────────

class RerankedChunks(BaseModel):
    """LLM output: reranked code chunks by relevance to a question."""
    ranked_indices: list[int] = Field(description="Indices of the chunks, ordered from most relevant to least relevant")
    reasoning: str = Field(description="Brief explanation of ranking")


# ── LLM Call Helper ──────────────────────────────────────────────────────────

async def _llm_call_with_retry(router, system: str, user: str, schema, max_tokens: int = 500):
    """Make an LLM call with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            result = await router.ai(
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                wait = (2 ** attempt) * 5
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            else:
                logger.error(f"LLM call failed: {e}")
                raise
    raise RuntimeError("Max retries exceeded for LLM call")


# ── LLM Retrieval Reranking ─────────────────────────────────────────────────

RERANK_SYSTEM = """You are a code search expert. Given a user's question and a list of code chunks, rank them from MOST to LEAST relevant.

Consider:
- How directly the chunk answers the question
- Whether it contains the specific code/logic being asked about
- Whether it's the definition vs just a usage/import
- Prefer actual implementation over tests or config

Return the indices (0-based) in order of relevance. Only include chunks that are actually relevant — omit completely irrelevant ones."""


async def rerank_chunks_llm(
    question: str,
    chunks: list[dict],
    router,
) -> list[int]:
    """
    LLM-powered reranking of code chunks for a question.

    Takes BM25-retrieved chunks and reranks them by actual relevance.
    Returns ordered list of chunk indices (0-based, most relevant first).
    Falls back to original order on failure.
    """
    if not chunks or len(chunks) <= 1:
        return list(range(len(chunks)))

    # Cap at 15 chunks to stay within token limits
    capped = chunks[:15]

    # Build prompt
    parts = [f"Question: {question}\n"]
    for i, chunk in enumerate(capped):
        file_path = chunk.get("file", chunk.get("file_path", "unknown"))
        symbol = chunk.get("symbol", "")
        content = chunk.get("content", "")[:800]
        sym_label = f" ({symbol})" if symbol else ""
        parts.append(f"[{i}] {file_path}{sym_label}:\n```\n{content}\n```\n")

    try:
        result = await _llm_call_with_retry(
            router,
            system=RERANK_SYSTEM,
            user="\n".join(parts),
            schema=RerankedChunks,
            max_tokens=200,
        )

        # Validate indices
        valid_indices = [i for i in result.ranked_indices if 0 <= i < len(capped)]

        # Add any missing indices at the end (in case LLM omitted some)
        seen = set(valid_indices)
        for i in range(len(capped)):
            if i not in seen:
                valid_indices.append(i)

        return valid_indices

    except Exception as e:
        logger.warning(f"Reranking failed: {e}, using original order")
        return list(range(len(capped)))
