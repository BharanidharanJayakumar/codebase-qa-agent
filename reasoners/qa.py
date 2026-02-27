import math
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import load_index, load_session, save_session_turn, list_indexed_projects

# Embeddings are optional — degrade gracefully if not installed
try:
    from skills.embeddings import load_and_search
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])

MIN_SCORE = 0.5  # Minimum BM25 score to consider a file relevant
MAX_CONTEXT_CHARS = 24_000  # ~6000 tokens — fits comfortably in 8K context window


class Answer(BaseModel):
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


def _retrieve_context(query: str, file_index: dict, keyword_map: dict, symbol_map: dict,
                      project_root: str = "") -> dict:
    """
    Hybrid retrieval: BM25 IDF + symbol boosting + optional semantic similarity.
    Returns formatted context with actual source code for the LLM.
    """
    query_keywords = extract_keywords(query, top_n=10)
    query_words = query.lower().split()
    total_files = len(file_index)

    # Direct symbol name matches (strongest signal)
    symbol_hits = {}
    for word in query_words:
        if word in symbol_map:
            symbol_hits[word] = symbol_map[word]

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

    # Semantic search boost (loads pre-computed embeddings from SQLite)
    if EMBEDDINGS_AVAILABLE and total_files > 0:
        try:
            sem_results = load_and_search(query, project_root=project_root, top_k=10)
            for rel_path, chunk_idx, score in sem_results:
                if score > 0.3:  # Only boost meaningfully similar chunks
                    file_scores[rel_path] = file_scores.get(rel_path, 0) + score * 3
        except Exception:
            pass  # Graceful degradation — BM25 still works

    # Filter by minimum score, then take top 5
    ranked = [(p, s) for p, s in file_scores.items() if s >= MIN_SCORE]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_files = [path for path, _ in ranked[:5]]

    # Build context from semantic chunks with token budget enforcement.
    # Pack chunks in order of file relevance until we hit MAX_CONTEXT_CHARS.
    context_parts = []
    chars_used = 0
    for file_path in top_files:
        meta = file_index.get(file_path, {})
        chunks = meta.get("chunks", [])
        if not chunks:
            continue
        for chunk in chunks:
            sym_label = f" ({chunk['symbol']})" if chunk.get("symbol") else ""
            part = (
                f"=== {file_path} [lines {chunk['start_line']}-{chunk['end_line']}]{sym_label} ===\n"
                f"{chunk['content']}"
            )
            if chars_used + len(part) > MAX_CONTEXT_CHARS:
                break  # Budget exhausted — stop packing
            context_parts.append(part)
            chars_used += len(part)
        if chars_used >= MAX_CONTEXT_CHARS:
            break

    # Add symbol location hints (small, always fits)
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
async def answer_question(question: str, session_id: str = "", project_path: str = "") -> dict:
    """
    Answer any question about the indexed codebase.
    Pass a session_id to enable follow-up questions with conversation memory.
    Pass a project_path to query a specific project (defaults to most recently indexed).

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_answer_question \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"question": "How does authentication work?", "session_id": "s1"}}'
    """
    stored = load_index(project_path)
    if not stored:
        return {
            "answer": "No index found. Please run index_project first.",
            "relevant_files": [],
            "confidence": "low",
            "follow_up": [],
            "session_id": session_id,
        }

    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    # Load conversation history if session_id provided
    history = load_session(session_id) if session_id else []

    # For follow-up questions, enrich the query with context from previous turns
    # so BM25 retrieval can find relevant files even for vague follow-ups like "what about its tests?"
    enriched_query = question
    if history:
        prev_files = []
        prev_keywords = []
        for turn in history[-2:]:  # last 2 turns for keyword enrichment
            prev_files.extend(turn.get("relevant_files", []))
            prev_keywords.extend(extract_keywords(turn["question"], top_n=5))
        enriched_query = f"{question} {' '.join(prev_keywords)}"

    retrieved = _retrieve_context(enriched_query, file_index, keyword_map, symbol_map, project_root)

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
            "session_id": session_id,
        }

    # Build conversation history for the LLM prompt
    history_block = ""
    if history:
        parts = []
        for turn in history[-3:]:  # last 3 turns max
            parts.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        history_block = (
            "Previous conversation:\n"
            + "\n---\n".join(parts)
            + "\n\n---\nNow answer the follow-up question below.\n\n"
        )

    result = await qa_router.ai(
        system=(
            "You are an expert software engineer helping a developer understand a codebase. "
            f"The project is at: {project_root}\n"
            "Answer questions using the actual source code provided. "
            "Be specific — mention file names, function names, and line numbers when you know them. "
            "If the context is insufficient, say so clearly."
        ),
        user=(
            f"{history_block}"
            f"Question: {question}\n\n"
            f"Relevant source code from the codebase:\n\n"
            f"{retrieved['context']}"
        ),
        schema=Answer,
    )

    # Save this turn to the session
    if session_id:
        save_session_turn(
            session_id, question,
            result.answer, retrieved["top_files"]
        )

    qa_router.app.note(
        f"Q: {question[:80]} | Files: {retrieved['top_files']} | Confidence: {retrieved['confidence']}",
        tags=["qa", "query"]
    )

    return {
        **result.model_dump(),
        "relevant_files": retrieved["top_files"],
        "confidence": retrieved["confidence"],
        "session_id": session_id,
        "project_id": stored.get("project_id", ""),
    }


@qa_router.reasoner()
async def find_relevant_files(query: str, project_path: str = "") -> dict:
    """
    Return the most relevant files for a topic — pure keyword retrieval, no LLM.
    Genuinely instant response.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_find_relevant_files \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"query": "authentication jwt token"}}'
    """
    stored = load_index(project_path)
    if not stored:
        return {"files": [], "reasoning": "No index found. Run index_project first."}

    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    retrieved = _retrieve_context(query, file_index, keyword_map, symbol_map, project_root)

    # Pure retrieval — no LLM call
    symbol_names = list(retrieved["symbol_hits"].keys())
    return {
        "files": retrieved["top_files"],
        "symbol_hits": symbol_names,
        "confidence": retrieved["confidence"],
        "reasoning": f"Matched {len(retrieved['top_files'])} files via BM25 keyword scoring and symbol lookup.",
    }


@qa_router.reasoner()
async def list_projects() -> dict:
    """
    List all indexed projects with slug and project_id. Pure retrieval, no LLM.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_list_projects \\
      -H "Content-Type: application/json" \\
      -d '{"input": {}}'
    """
    projects = list_indexed_projects()
    return {
        "projects": projects,
        "total": len(projects),
    }


@qa_router.reasoner()
async def get_file_content(file_path: str, project_path: str = "") -> dict:
    """
    Get the source code of a specific file from the index.
    file_path is the relative path within the project (e.g. 'src/main.py').
    project_path accepts path, slug, or project_id.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.qa_get_file_content \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"file_path": "skills/storage.py"}}'
    """
    stored = load_index(project_path)
    if not stored:
        return {"error": "No index found. Run index_project first.", "content": ""}

    file_index = stored["file_index"]
    project_root = stored["project_root"]

    if file_path not in file_index:
        return {
            "error": f"File not found in index: {file_path}",
            "content": "",
            "available_files": sorted(file_index.keys())[:20],
        }

    meta = file_index[file_path]
    chunks = meta.get("chunks", [])

    # Reassemble content from chunks
    content = "\n".join(chunk["content"] for chunk in chunks)

    # Try to read fresh from disk if the project is still accessible
    full_path = Path(project_root) / file_path
    if full_path.exists():
        try:
            from skills.scanner import read_file
            fresh = read_file(str(full_path))
            if fresh.get("content"):
                content = fresh["content"]
        except Exception:
            pass  # Fall back to indexed chunks

    return {
        "file_path": file_path,
        "project_id": stored.get("project_id", ""),
        "content": content,
        "symbols": meta.get("symbols", []),
        "keywords": meta.get("keywords", []),
        "extension": meta.get("extension", ""),
        "size_bytes": meta.get("size_bytes", 0),
        "chunks_count": len(chunks),
    }
