"""
Retrieval agent — file relevance scoring, code search, and context building.
Moved from qa.py to its own agent for separation of concerns.
No LLM calls; pure keyword/BM25/embedding retrieval.
"""
import math
import re
from pathlib import Path as _Path

from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import load_index

# Embeddings are optional
try:
    from skills.embeddings import load_and_search
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

retrieval_router = AgentRouter(prefix="retrieval", tags=["retrieval"])

MIN_SCORE = 0.2
MAX_CONTEXT_CHARS = 24_000


def retrieve_context(query: str, file_index: dict, keyword_map: dict, symbol_map: dict,
                     project_root: str = "") -> dict:
    """
    Hybrid retrieval: BM25 IDF + symbol boosting + optional semantic similarity.
    Returns formatted context with actual source code for the LLM.
    """
    query_keywords = extract_keywords(query, top_n=10)
    query_words = query.lower().split()
    total_files = len(file_index)

    # Direct symbol name matches
    symbol_lower = {k.lower(): v for k, v in symbol_map.items()}
    symbol_hits = {}
    for word in query_words:
        if word in symbol_lower:
            symbol_hits[word] = symbol_lower[word]
    for raw_word in query.split():
        if raw_word in symbol_map and raw_word.lower() not in symbol_hits:
            symbol_hits[raw_word.lower()] = symbol_map[raw_word]

    # BM25 IDF scoring
    file_scores: dict[str, float] = {}
    for kw in query_keywords:
        files_with_kw = keyword_map.get(kw, [])
        if not files_with_kw:
            continue
        df = len(files_with_kw)
        idf = math.log((total_files - df + 0.5) / (df + 0.5) + 1)
        for file_path in files_with_kw:
            file_scores[file_path] = file_scores.get(file_path, 0) + idf

    # Boost files with direct symbol hits
    for sym_name, locations in symbol_hits.items():
        for loc in locations:
            file_path = loc["file"]
            file_scores[file_path] = file_scores.get(file_path, 0) + 5

    # Semantic search boost
    if EMBEDDINGS_AVAILABLE and total_files > 0:
        try:
            sem_results = load_and_search(query, project_root=project_root, top_k=10)
            for rel_path, chunk_idx, score in sem_results:
                if score > 0.3:
                    file_scores[rel_path] = file_scores.get(rel_path, 0) + score * 3
        except Exception:
            pass

    ranked = [(p, s) for p, s in file_scores.items() if s >= MIN_SCORE]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_files = [path for path, _ in ranked[:5]]

    # Build context from chunks with token budget
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
                break
            context_parts.append(part)
            chars_used += len(part)
        if chars_used >= MAX_CONTEXT_CHARS:
            break

    # Add symbol location hints
    for sym_name, locations in symbol_hits.items():
        for loc in locations:
            context_parts.append(
                f"\n[Symbol `{sym_name}` defined in {loc['file']} "
                f"at line {loc['line']} ({loc['type']})]"
            )

    # Confidence calculation
    top_score = ranked[0][1] if ranked else 0
    max_possible = len(query_keywords) * math.log(total_files + 1) + 5 if total_files else 1
    ratio = top_score / max_possible if max_possible > 0 else 0

    has_symbol_hits = len(symbol_hits) > 0
    matching_file_count = len(ranked)

    if has_symbol_hits and matching_file_count >= 1:
        confidence = "high"
    elif ratio >= 0.15 or (matching_file_count >= 3 and ratio >= 0.08):
        confidence = "high"
    elif ratio >= 0.05 or matching_file_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "context": "\n\n".join(context_parts),
        "top_files": top_files,
        "symbol_hits": symbol_hits,
        "confidence": confidence,
        "top_score": top_score,
    }


@retrieval_router.reasoner()
async def find_relevant_files(query: str, project_path: str = "") -> dict:
    """
    Return the most relevant files for a topic — pure keyword retrieval, no LLM.
    """
    stored = load_index(project_path)
    if not stored:
        return {"files": [], "reasoning": "No index found. Run index_project first."}

    retrieved = retrieve_context(
        query, stored["file_index"], stored["keyword_map"],
        stored["symbol_map"], stored["project_root"],
    )

    symbol_names = list(retrieved["symbol_hits"].keys())
    return {
        "files": retrieved["top_files"],
        "symbol_hits": symbol_names,
        "confidence": retrieved["confidence"],
        "reasoning": f"Matched {len(retrieved['top_files'])} files via BM25 keyword scoring and symbol lookup.",
    }


@retrieval_router.reasoner()
async def search_code(query: str, project_path: str = "") -> dict:
    """
    Grep-like code search across indexed files. Returns matching lines with file path and line numbers.
    """
    stored = load_index(project_path)
    if not stored:
        return {"matches": [], "total": 0, "error": "No index found."}

    file_index = stored["file_index"]
    project_root = stored["project_root"]
    matches = []

    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error:
        return {"matches": [], "total": 0, "error": "Invalid search pattern."}

    for rel_path, meta in file_index.items():
        chunks = meta.get("chunks", [])
        content = "\n".join(c["content"] for c in chunks)

        full_path = _Path(project_root) / rel_path
        if full_path.exists():
            try:
                from skills.scanner import read_file
                fresh = read_file(str(full_path))
                if fresh.get("content"):
                    content = fresh["content"]
            except Exception:
                pass

        for i, line in enumerate(content.split("\n"), 1):
            if pattern.search(line):
                matches.append({
                    "file": rel_path,
                    "line": i,
                    "text": line.strip(),
                })
                if len(matches) >= 50:
                    return {"matches": matches, "total": len(matches), "truncated": True}

    return {"matches": matches, "total": len(matches)}
