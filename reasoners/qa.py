from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import load_index

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])


class Answer(BaseModel):
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


class FileMatch(BaseModel):
    files: list[str] = Field(description="File paths ranked by relevance, most relevant first")
    reasoning: str = Field(description="Brief explanation of why these files are relevant")


def _retrieve_context(query: str, file_index: dict, keyword_map: dict, symbol_map: dict) -> dict:
    """
    Pure Python retrieval — find the most relevant files for a query.
    No LLM needed here. Returns actual file content for the top matches.
    """
    query_keywords = extract_keywords(query, top_n=10)
    query_words = query.lower().split()

    # Direct symbol name matches (strongest signal)
    symbol_hits = {
        word: symbol_map[word]
        for word in query_words
        if word in symbol_map
    }

    # Score files by keyword frequency
    file_scores: dict[str, int] = {}
    for kw in query_keywords:
        for file_path in keyword_map.get(kw, []):
            file_scores[file_path] = file_scores.get(file_path, 0) + 1

    # Boost files with direct symbol hits
    for sym_info in symbol_hits.values():
        file_path = sym_info["file"]
        file_scores[file_path] = file_scores.get(file_path, 0) + 5

    # Top 5 most relevant files
    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_files = [path for path, _ in ranked]

    # Build context from ACTUAL file content — not pre-baked summaries
    # This is why indexing is now instant: we stored raw content, not LLM outputs
    # The LLM sees real code at query time, giving much more accurate answers
    context_parts = []
    for file_path in top_files:
        meta = file_index.get(file_path, {})
        if meta:
            context_parts.append(
                f"=== File: {file_path} ===\n"
                f"{meta.get('content_chunk', '')}"
            )

    # Add symbol location info if directly matched
    for sym_name, sym_info in symbol_hits.items():
        context_parts.append(
            f"\n[Symbol `{sym_name}` defined in {sym_info['file']} "
            f"at line {sym_info['line']} ({sym_info['type']})]"
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
    1. Load index from memory (instant)
    2. Keyword retrieval finds top 5 relevant files (instant, no LLM)
    3. Pass actual file content to Ollama — LLM answers from real code
    4. Return structured answer with source files

    LLM is only called ONCE per question, on real code, not summaries.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_answer_question \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"question": "How does authentication work?"}}'
    """
    stored = load_index()
    if not stored:
        return {
            "answer": "No index found. Please run index_project first.",
            "relevant_files": [],
            "confidence": "low",
            "follow_up": [],
        }

    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    retrieved = _retrieve_context(question, file_index, keyword_map, symbol_map)

    # Single LLM call with actual code content — happens only at query time
    result = await qa_router.ai(
        system=(
            "You are an expert software engineer helping a developer understand a codebase. "
            f"The project is at: {project_root}\n"
            "Answer questions using the actual source code provided. "
            "Be specific — mention file names, function names, and line numbers when you know them. "
            "If the context is insufficient, say so clearly."
        ),
        user=(
            f"Question: {question}\n\n"
            f"Relevant source code from the codebase:\n\n"
            f"{retrieved['context']}"
        ),
        schema=Answer,
    )

    qa_router.app.note(
        f"Q: {question[:80]} | Files: {retrieved['top_files']} | Confidence: {retrieved['confidence']}",
        tags=["qa", "query"]
    )

    return {
        **result.model_dump(),
        "relevant_files": retrieved["top_files"],
        "confidence": retrieved["confidence"],
    }


@qa_router.reasoner()
async def find_relevant_files(query: str) -> dict:
    """
    Return the most relevant files for a topic — pure keyword retrieval, no LLM.
    Fastest endpoint — instant response.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_find_relevant_files \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"query": "authentication jwt token"}}'
    """
    stored = load_index()
    if not stored:
        return {"files": [], "reasoning": "No index found. Run index_project first."}

    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]

    retrieved = _retrieve_context(query, file_index, keyword_map, symbol_map)

    result = await qa_router.ai(
        system="You are a code navigation assistant. Briefly explain why these files are relevant.",
        user=f"Query: {query}\n\nTop matching files:\n" + "\n".join(retrieved["top_files"]),
        schema=FileMatch,
    )

    return {
        "files": retrieved["top_files"],
        "reasoning": result.reasoning,
    }
