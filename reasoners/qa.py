import math
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import load_index

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])

MIN_SCORE = 0.5  # Minimum BM25 score to consider a file relevant


class Answer(BaseModel):
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


def _retrieve_context(query: str, file_index: dict, keyword_map: dict, symbol_map: dict) -> dict:
    """
    Retrieve relevant chunks using BM25 IDF scoring + symbol boosting.
    Returns formatted context with actual source code for the LLM.
    """
    query_keywords = extract_keywords(query, top_n=10)
    query_words = query.lower().split()
    total_files = len(file_index)

    # Direct symbol name matches (strongest signal)
    # symbol_map is now one-to-many: each name → list of locations
    symbol_hits = {}
    for word in query_words:
        if word in symbol_map:
            symbol_hits[word] = symbol_map[word]  # list of {file, line, type}

    # BM25 IDF scoring: rare keywords score higher than common ones
    file_scores: dict[str, float] = {}
    for kw in query_keywords:
        files_with_kw = keyword_map.get(kw, [])
        if not files_with_kw:
            continue
        df = len(files_with_kw)
        idf = math.log((total_files - df + 0.5) / (df + 0.5) + 1)
        for file_path in files_with_kw:
            file_scores[file_path] = file_scores.get(file_path, 0) + idf

    # Boost all files with direct symbol hits (one-to-many)
    for sym_name, locations in symbol_hits.items():
        for loc in locations:
            file_path = loc["file"]
            file_scores[file_path] = file_scores.get(file_path, 0) + 5

    # Filter by minimum score, then take top 5
    ranked = [(p, s) for p, s in file_scores.items() if s >= MIN_SCORE]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_files = [path for path, _ in ranked[:5]]

    # Build context from semantic chunks, not the truncated 4000-char blob
    context_parts = []
    for file_path in top_files:
        meta = file_index.get(file_path, {})
        chunks = meta.get("chunks", [])
        if not chunks:
            continue
        for chunk in chunks:
            sym_label = f" ({chunk['symbol']})" if chunk.get("symbol") else ""
            context_parts.append(
                f"=== {file_path} [lines {chunk['start_line']}-{chunk['end_line']}]{sym_label} ===\n"
                f"{chunk['content']}"
            )

    # Add symbol location hints
    for sym_name, locations in symbol_hits.items():
        for loc in locations:
            context_parts.append(
                f"\n[Symbol `{sym_name}` defined in {loc['file']} "
                f"at line {loc['line']} ({loc['type']})]"
            )

    # Meaningful confidence based on score quality, not file count
    top_score = ranked[0][1] if ranked else 0
    max_possible = len(query_keywords) * math.log(total_files + 1) + 5 if total_files else 1
    ratio = top_score / max_possible if max_possible > 0 else 0
    confidence = "high" if ratio >= 0.3 else "medium" if ratio >= 0.1 else "low"

    return {
        "context": "\n\n".join(context_parts),
        "top_files": top_files,
        "symbol_hits": symbol_hits,
        "confidence": confidence,
        "top_score": top_score,
    }


@qa_router.reasoner()
async def answer_question(question: str) -> dict:
    """
    Answer any question about the indexed codebase.

    Flow:
    1. Load index from disk (instant)
    2. BM25 keyword retrieval finds top 5 relevant files (instant, no LLM)
    3. Pass actual source code chunks to LLM — answers from real code
    4. Return structured answer with source files

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

    # Guard: if no files matched, don't call the LLM with empty context
    if not retrieved["top_files"]:
        return {
            "answer": (
                "No relevant files found for this question. "
                "Try using specific function names, class names, or file names from the codebase."
            ),
            "relevant_files": [],
            "confidence": "low",
            "follow_up": [],
        }

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
    Genuinely instant response.

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

    # Pure retrieval — no LLM call
    symbol_names = list(retrieved["symbol_hits"].keys())
    return {
        "files": retrieved["top_files"],
        "symbol_hits": symbol_names,
        "confidence": retrieved["confidence"],
        "reasoning": f"Matched {len(retrieved['top_files'])} files via BM25 keyword scoring and symbol lookup.",
    }
