"""
LLM-powered intelligence for indexing — replaces hardcoded regex/heuristic logic.

Provides:
1. Symbol categorization — LLM reads code context to categorize symbols
2. Concept extraction — LLM extracts semantic concepts from code
3. Retrieval reranking — LLM picks best code chunks for a question

All functions include graceful fallback to the old heuristic methods.
Rate-limited for Groq free tier with exponential backoff.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Rate limiting (Groq free tier: ~30 req/min)
CALL_DELAY_SECONDS = 1
MAX_RETRIES = 3
# Max files per LLM batch (to stay within token limits)
BATCH_SIZE = 8
# Max code chars per file in batch
MAX_CODE_PER_FILE = 1500


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class FileCategories(BaseModel):
    """LLM output: categories for symbols in a batch of files."""
    files: list["FileCategoryEntry"] = Field(description="One entry per file analyzed")


class FileCategoryEntry(BaseModel):
    file_path: str = Field(description="Relative path of the file")
    symbols: list["SymbolCategory"] = Field(description="Categorized symbols in this file")


class SymbolCategory(BaseModel):
    name: str = Field(description="Symbol name")
    category: str = Field(description="One of: dto, route, service, test, config, middleware, utility, component, hook, store, repository, migration, seed, validator, guard, pipe, interceptor, decorator, factory, provider, event, job, command, query, handler")
    detail: str = Field(description="Brief description of what this symbol does (10 words max)")


class FileConceptsOutput(BaseModel):
    """LLM output: semantic concepts extracted from code."""
    files: list["FileConceptEntry"] = Field(description="One entry per file analyzed")


class FileConceptEntry(BaseModel):
    file_path: str = Field(description="Relative path of the file")
    concepts: list[str] = Field(description="5-15 semantic concepts/keywords that describe what this file does. Include domain terms, patterns, and key behaviors.")


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


# ── 1. LLM Symbol Categorization ────────────────────────────────────────────

CATEGORIZE_SYSTEM = """You are an expert code analyst. Given a batch of source code files with their symbols, categorize each symbol.

Categories (use exactly these strings):
- dto: Data transfer objects, models, schemas, entities, request/response types
- route: API endpoints, controllers, route handlers, HTTP request handlers
- service: Business logic services, use cases, interactors
- test: Test files, test functions, test classes, specs
- config: Configuration, settings, environment setup
- middleware: Request/response middleware, interceptors, filters
- utility: Helper functions, utilities, shared tools
- component: UI components (React, Vue, Angular, Svelte, etc.)
- hook: React hooks, Vue composables, custom hooks
- store: State management (Redux, Zustand, Vuex, etc.)
- repository: Data access layer, database queries, ORM models
- migration: Database migrations, schema changes
- validator: Input validation, schema validation
- guard: Auth guards, permission checks, route guards
- event: Event handlers, listeners, emitters, subscribers
- job: Background jobs, workers, scheduled tasks, queues
- handler: General handlers (command handlers, message handlers, etc.)

Rules:
- Categorize based on what the code DOES, not just file/folder naming
- A class named "UserService" in a controllers folder is still a "service" if it contains business logic
- If a symbol doesn't fit any category, use "utility"
- Be concise in detail — max 10 words
- ONLY include symbols that are meaningful (skip generic things like Ok, BadRequest, NotFound etc.)"""


def _build_categorize_prompt(file_batch: list[dict]) -> str:
    """Build a prompt for categorizing symbols in a batch of files."""
    parts = []
    for entry in file_batch:
        rel_path = entry["rel_path"]
        symbols = entry["symbols"]
        code_snippet = entry.get("code_snippet", "")

        sym_list = ", ".join(f"{s['name']} ({s['type']})" for s in symbols[:20])
        parts.append(f"File: {rel_path}")
        parts.append(f"Symbols: [{sym_list}]")
        if code_snippet:
            parts.append(f"Code:\n```\n{code_snippet}\n```")
        parts.append("")

    return "\n".join(parts)


async def categorize_symbols_llm(
    file_index: dict,
    symbol_map: dict,
    project_root: str,
    router,
) -> list[tuple[str, str, str, str]]:
    """
    LLM-powered symbol categorization. Sends batches of files with their
    symbols and code snippets to the LLM for accurate categorization.

    Returns list of (rel_path, symbol_name, category, detail) tuples.
    Falls back to regex categorization on failure.
    """
    # Build batches of files with symbols
    files_with_symbols = []
    for rel_path, meta in file_index.items():
        symbols_raw = meta.get("symbols", [])
        if not symbols_raw:
            continue

        # Get symbol details from symbol_map
        symbols = []
        for sym_name in symbols_raw:
            if sym_name in symbol_map:
                for loc in symbol_map[sym_name]:
                    if loc["file"] == rel_path:
                        symbols.append({"name": sym_name, "type": loc["type"], "line": loc["line"]})
                        break
            else:
                symbols.append({"name": sym_name, "type": "unknown", "line": 0})

        if not symbols:
            continue

        # Get a code snippet (first chunk or read from disk)
        code_snippet = ""
        chunks = meta.get("chunks", [])
        if chunks:
            # Get the first meaningful chunk (skip headers)
            for chunk in chunks[:3]:
                if chunk.get("symbol"):
                    code_snippet = chunk["content"][:MAX_CODE_PER_FILE]
                    break
            if not code_snippet and chunks:
                code_snippet = chunks[0]["content"][:MAX_CODE_PER_FILE]

        files_with_symbols.append({
            "rel_path": rel_path,
            "symbols": symbols,
            "code_snippet": code_snippet,
        })

    if not files_with_symbols:
        return []

    # Process in batches
    all_categories = []
    batches = [files_with_symbols[i:i + BATCH_SIZE] for i in range(0, len(files_with_symbols), BATCH_SIZE)]

    logger.info(f"Categorizing symbols in {len(files_with_symbols)} files ({len(batches)} LLM batches)")

    for batch_idx, batch in enumerate(batches):
        prompt = _build_categorize_prompt(batch)

        try:
            result = await _llm_call_with_retry(
                router,
                system=CATEGORIZE_SYSTEM,
                user=prompt,
                schema=FileCategories,
                max_tokens=800,
            )

            for file_entry in result.files:
                for sym in file_entry.symbols:
                    all_categories.append((
                        file_entry.file_path,
                        sym.name,
                        sym.category,
                        sym.detail,
                    ))

            logger.info(f"  Batch {batch_idx + 1}/{len(batches)}: categorized {sum(len(f.symbols) for f in result.files)} symbols")

        except Exception as e:
            logger.warning(f"  Batch {batch_idx + 1} failed: {e}, falling back to regex for this batch")
            # Fallback: use the old regex categorizer for this batch
            from skills.aggregator import categorize_symbols as regex_categorize
            for entry in batch:
                rel_path = entry["rel_path"]
                chunks = file_index.get(rel_path, {}).get("chunks", [])
                content = "\n".join(c.get("content", "") for c in chunks)
                regex_results = regex_categorize(rel_path, content, entry["symbols"])
                all_categories.extend(regex_results)

        # Rate limiting between batches
        if batch_idx < len(batches) - 1:
            await asyncio.sleep(CALL_DELAY_SECONDS)

    return all_categories


# ── 2. LLM Concept Extraction ───────────────────────────────────────────────

CONCEPTS_SYSTEM = """You are an expert code analyst. Given source code files, extract the key semantic concepts from each file.

A "concept" is a meaningful term that describes WHAT the code does — not syntactic elements.

Good concepts: "user authentication", "jwt token", "password hashing", "match scoring", "live updates", "websocket", "CRUD operations", "pagination", "file upload", "caching", "rate limiting"

Bad concepts (too generic): "function", "class", "variable", "import", "export", "return", "string", "number"

Rules:
- Extract 5-15 concepts per file
- Include domain-specific terms (e.g., "cricket match", "innings", "ball-by-ball")
- Include design patterns (e.g., "repository pattern", "middleware chain", "observer")
- Include technology-specific concepts (e.g., "SignalR hub", "Entity Framework", "React hooks")
- Split compound concepts into individual terms (e.g., "user authentication" AND "authentication")
- Include both the specific AND general form when relevant"""


def _build_concepts_prompt(file_batch: list[dict]) -> str:
    """Build a prompt for extracting concepts from a batch of files."""
    parts = []
    for entry in file_batch:
        parts.append(f"File: {entry['rel_path']}")
        code = entry.get("code_snippet", "")
        if code:
            parts.append(f"```\n{code}\n```")
        else:
            symbols = entry.get("symbols", [])
            sym_list = ", ".join(s["name"] for s in symbols[:15])
            parts.append(f"Symbols: [{sym_list}]")
        parts.append("")

    return "\n".join(parts)


async def extract_concepts_llm(
    file_index: dict,
    project_root: str,
    router,
) -> dict[str, list[str]]:
    """
    LLM-powered concept extraction. Returns {rel_path: [concepts]} dict.
    These concepts replace the old keyword extraction for building the keyword_map.

    Falls back to heuristic keyword extraction on failure.
    """
    # Build batches
    files_to_process = []
    for rel_path, meta in file_index.items():
        code_snippet = ""
        chunks = meta.get("chunks", [])
        if chunks:
            # Combine first few chunks for context
            combined = []
            chars = 0
            for chunk in chunks:
                content = chunk.get("content", "")
                if chars + len(content) > MAX_CODE_PER_FILE:
                    combined.append(content[:MAX_CODE_PER_FILE - chars])
                    break
                combined.append(content)
                chars += len(content)
            code_snippet = "\n".join(combined)

        symbols_raw = meta.get("symbols", [])
        symbols = [{"name": s} for s in symbols_raw[:15]]

        files_to_process.append({
            "rel_path": rel_path,
            "code_snippet": code_snippet,
            "symbols": symbols,
        })

    if not files_to_process:
        return {}

    # Process in batches
    all_concepts: dict[str, list[str]] = {}
    batches = [files_to_process[i:i + BATCH_SIZE] for i in range(0, len(files_to_process), BATCH_SIZE)]

    logger.info(f"Extracting concepts from {len(files_to_process)} files ({len(batches)} LLM batches)")

    for batch_idx, batch in enumerate(batches):
        prompt = _build_concepts_prompt(batch)

        try:
            result = await _llm_call_with_retry(
                router,
                system=CONCEPTS_SYSTEM,
                user=prompt,
                schema=FileConceptsOutput,
                max_tokens=600,
            )

            for file_entry in result.files:
                all_concepts[file_entry.file_path] = file_entry.concepts

            logger.info(f"  Concepts batch {batch_idx + 1}/{len(batches)}: extracted for {len(result.files)} files")

        except Exception as e:
            logger.warning(f"  Concepts batch {batch_idx + 1} failed: {e}, falling back to heuristic")
            from skills.extractor import extract_keywords
            for entry in batch:
                content = entry.get("code_snippet", "")
                if content:
                    all_concepts[entry["rel_path"]] = extract_keywords(content, top_n=15)

        # Rate limiting
        if batch_idx < len(batches) - 1:
            await asyncio.sleep(CALL_DELAY_SECONDS)

    return all_concepts


# ── 3. LLM Retrieval Reranking ──────────────────────────────────────────────

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
