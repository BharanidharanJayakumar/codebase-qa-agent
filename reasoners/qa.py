import json
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])


class Answer(BaseModel):
    """Structured answer returned to the caller."""
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low — based on how much relevant context was found")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask next")


class FileMatch(BaseModel):
    """Result from find_relevant_files."""
    files: list[str] = Field(description="File paths ranked by relevance, most relevant first")
    reasoning: str = Field(description="Brief explanation of why these files are relevant")


def _retrieve_context(query: str, file_summaries: dict, keyword_map: dict, symbol_map: dict) -> dict:
    """
    Pure Python retrieval — no LLM needed to find relevant context.

    Strategy:
    1. Extract keywords from the question
    2. Look up which files contain those keywords in the keyword_map
    3. Check if any function/class names in the question match symbol_map
    4. Score files by how many keyword hits they have
    5. Return top 5 file summaries as context

    This is a lightweight keyword-based retrieval (like early search engines).
    Good enough for codebase Q&A — the LLM then reasons over the results.
    """
    query_keywords = extract_keywords(query, top_n=10)

    # Also check if any words in the query match symbol names directly
    query_words = query.lower().split()
    symbol_hits = {
        word: symbol_map[word]
        for word in query_words
        if word in symbol_map
    }

    # Score files: +1 for each keyword that mentions them
    file_scores: dict[str, int] = {}
    for kw in query_keywords:
        for file_path in keyword_map.get(kw, []):
            file_scores[file_path] = file_scores.get(file_path, 0) + 1

    # Boost files that contain directly matched symbols (strong signal)
    for sym_info in symbol_hits.values():
        file_path = sym_info["file"]
        file_scores[file_path] = file_scores.get(file_path, 0) + 5

    # Pick top 5 most relevant files
    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_files = [path for path, _ in ranked]

    # Build context block: file path + its summary
    context_parts = []
    for file_path in top_files:
        meta = file_summaries.get(file_path, {})
        if meta:
            context_parts.append(
                f"File: {file_path}\n"
                f"Purpose: {meta.get('purpose', 'unknown')}\n"
                f"Summary: {meta.get('summary', '')}\n"
                f"Dependencies: {', '.join(meta.get('dependencies', []))}"
            )

    # If symbol was directly found, add that too
    for sym_name, sym_info in symbol_hits.items():
        context_parts.append(
            f"Symbol `{sym_name}` defined in {sym_info['file']} at line {sym_info['line']} ({sym_info['type']})"
        )

    return {
        "context": "\n\n".join(context_parts),
        "top_files": top_files,
        "symbol_hits": symbol_hits,
        "confidence": "high" if len(top_files) >= 3 else "medium" if top_files else "low",
    }


@qa_router.reasoner()
async def answer_question(question: str) -> dict:
    """
    Answer any question about the indexed codebase.

    Flow:
    1. Load index from memory (instant, no filesystem reads)
    2. Retrieve relevant file summaries using keyword matching
    3. Pass only the relevant context to the LLM
    4. Return structured answer with source files

    This is the endpoint Claude Code will call instead of reading files.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_answer_question \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"question": "How does authentication work?"}}'
    """
    # Load the full index from memory
    file_summaries_raw = await qa_router.app.memory.get("file_summaries")
    keyword_map_raw = await qa_router.app.memory.get("keyword_map")
    symbol_map_raw = await qa_router.app.memory.get("symbol_map")
    project_root = await qa_router.app.memory.get("project_root") or "unknown"

    if not file_summaries_raw:
        return {
            "answer": "No index found. Please run index_project first pointing at your codebase.",
            "relevant_files": [],
            "confidence": "low",
            "follow_up": [],
        }

    file_summaries = json.loads(file_summaries_raw)
    keyword_map = json.loads(keyword_map_raw or "{}")
    symbol_map = json.loads(symbol_map_raw or "{}")

    # Retrieve relevant context without touching the filesystem
    retrieved = _retrieve_context(question, file_summaries, keyword_map, symbol_map)

    # LLM answers using only the retrieved summaries as context
    # The LLM never sees raw file content — only structured summaries
    # This keeps the prompt small and the answers fast
    result = await qa_router.ai(
        system=(
            "You are an expert software engineer helping a developer understand a codebase. "
            f"The project is located at: {project_root}\n"
            "Answer questions using only the file summaries provided. "
            "Be specific about file names and line numbers when you know them. "
            "If you don't have enough context, say so clearly."
        ),
        user=(
            f"Question: {question}\n\n"
            f"Relevant context from the codebase index:\n\n"
            f"{retrieved['context']}"
        ),
        schema=Answer,
    )

    qa_router.app.note(
        f"Q: {question[:80]} | Confidence: {retrieved['confidence']} | Files: {retrieved['top_files']}",
        tags=["qa", "query"]
    )

    # Merge confidence from retrieval (LLM might override, but retrieval is more honest)
    return {
        **result.model_dump(),
        "relevant_files": retrieved["top_files"],
        "confidence": retrieved["confidence"],
    }


@qa_router.reasoner()
async def find_relevant_files(query: str) -> dict:
    """
    Return the most relevant files for a given topic or task — without a full answer.

    Useful when you want to know *where to look* before diving in.
    Lighter than answer_question — no LLM call, pure keyword retrieval.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_find_relevant_files \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"query": "database connection pooling"}}'
    """
    file_summaries_raw = await qa_router.app.memory.get("file_summaries")
    keyword_map_raw = await qa_router.app.memory.get("keyword_map")
    symbol_map_raw = await qa_router.app.memory.get("symbol_map")

    if not file_summaries_raw:
        return {"files": [], "reasoning": "No index found. Run index_project first."}

    file_summaries = json.loads(file_summaries_raw)
    keyword_map = json.loads(keyword_map_raw or "{}")
    symbol_map = json.loads(symbol_map_raw or "{}")

    retrieved = _retrieve_context(query, file_summaries, keyword_map, symbol_map)

    # For this endpoint we do use a lightweight LLM call just for the reasoning explanation
    result = await qa_router.ai(
        system="You are a code navigation assistant. Explain briefly why these files are relevant to the query.",
        user=f"Query: {query}\n\nRelevant files found:\n{retrieved['context']}",
        schema=FileMatch,
    )

    return {
        "files": retrieved["top_files"],
        "reasoning": result.reasoning,
    }
