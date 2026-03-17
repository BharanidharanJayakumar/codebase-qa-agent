"""
QA orchestrator agent — Navigator-driven, question-aware code exploration.

Flow: Navigate → Read targeted files → Answer (with optional drill-down).
The LLM decides what to read based on folder summaries + dependency graph.
"""
import json
import logging
import re
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import (
    load_index, load_session, save_session_turn, list_indexed_projects,
    load_project_summary, load_symbol_categories, load_file_imports,
    load_module_summaries, load_semantic_summary,
)
from pathlib import Path as _Path
from reasoners.retrieval import retrieve_context
from reasoners.navigator import (
    navigate, navigate_fallback, read_targeted_files, read_files_by_paths,
    build_summary_context, AnswerWithDrilldown, ANSWER_SYSTEM,
)

logger = logging.getLogger(__name__)

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])


class Answer(BaseModel):
    """Final answer schema (used for drill-down final round)."""
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


# ── Aggregate Detection ──────────────────────────────────────────────────────

AGGREGATE_WORDS = ["how many", "count", "list all", "list every", "show all", "show every"]


def _is_aggregate(question: str) -> bool:
    q_lower = question.lower()
    return any(w in q_lower for w in AGGREGATE_WORDS)


# ── Aggregate Formatting (kept for counting/listing questions) ───────────────

def _format_aggregate(question: str, project_path: str) -> tuple[str, dict]:
    """Format aggregate data and return (context, extra_data)."""
    question_lower = question.lower()
    category_map = {
        "dto": "dto", "dtos": "dto", "model": "dto", "models": "dto",
        "schema": "dto", "entity": "dto", "entities": "dto",
        "route": "route", "routes": "route", "endpoint": "route", "endpoints": "route",
        "api": "route", "controller": "route", "controllers": "route",
        "test": "test", "tests": "test", "spec": "test", "specs": "test",
        "service": "service", "services": "service",
        "config": "config", "configuration": "config",
        "middleware": "middleware",
    }

    target_category = None
    for keyword, cat in category_map.items():
        if keyword in question_lower:
            target_category = cat
            break

    rows = load_symbol_categories(project_path, category=target_category)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        cat = row["category"] if isinstance(row, dict) else row[2]
        entry = {
            "file": row["rel_path"] if isinstance(row, dict) else row[0],
            "symbol": row["symbol_name"] if isinstance(row, dict) else row[1],
            "detail": row["detail"] if isinstance(row, dict) else row[3],
        }
        grouped.setdefault(cat, []).append(entry)

    parts = ["=== CATEGORIZED SYMBOLS ==="]
    for cat, items in grouped.items():
        # Group by file for cleaner output — shows all files with their symbols
        by_file: dict[str, list[str]] = {}
        for item in items:
            by_file.setdefault(item["file"], []).append(item["symbol"])

        unique_files = len(by_file)
        parts.append(f"\n{cat.upper()} ({len(items)} symbols across {unique_files} files):")
        for filepath, symbols in sorted(by_file.items()):
            # Deduplicate and filter out generic symbols (Ok, BadRequest, NotFound, etc.)
            generic = {"Ok", "BadRequest", "NotFound", "NoContent", "Unauthorized",
                       "CreatedAtAction", "ArgumentNullException"}
            meaningful = [s for s in symbols if s not in generic]
            sym_str = ", ".join(meaningful[:10])
            if len(meaningful) > 10:
                sym_str += f" (+{len(meaningful) - 10} more)"
            parts.append(f"  {filepath}: [{sym_str}]")

    total = sum(len(v) for v in grouped.values())
    unique_total = sum(len(set(item["file"] for item in items)) for items in grouped.values())
    extra = {"aggregate_total": total, "aggregate_filter": target_category, "unique_files": unique_total}
    return "\n".join(parts), extra


# ── Main Answer Endpoint ──────────────────────────────────────────────────────

@qa_router.reasoner()
async def answer_question(question: str, session_id: str = "", project_path: str = "") -> dict:
    """
    Answer any question about the indexed codebase.

    Flow: Navigate → Read targeted files → Answer (with optional drill-down).
    The navigator LLM reads the project map and decides what to look at.
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

    # Load conversation history
    history = load_session(session_id) if session_id else []

    # ── Step 1: Navigate (1 LLM call) ──
    # The navigator sees the full project map and decides what files to read
    try:
        decision = await navigate(question, project_root, file_index, symbol_map, qa_router)
    except Exception as e:
        logger.warning(f"Navigator failed: {e}, using fallback")
        decision = navigate_fallback(question, file_index)

    # ── Step 2: Build context (0 LLM calls — reads from disk/DB) ──
    context_parts = []
    top_files = []
    extra_data = {}

    if decision.needs_code:
        # Read the specific files the navigator selected
        code_context = read_targeted_files(decision, file_index, project_root)
        if code_context:
            context_parts.append(code_context)
        top_files = decision.target_files[:10]

        # Supplement with BM25 if navigator found very few files
        if len(decision.target_files) < 2 and decision.search_terms:
            bm25_ctx = retrieve_context(
                " ".join(decision.search_terms), file_index, keyword_map, symbol_map, project_root
            )
            if bm25_ctx["context"]:
                context_parts.append("=== ADDITIONAL SEARCH RESULTS ===\n" + bm25_ctx["context"])
                top_files.extend(bm25_ctx["top_files"])
    else:
        # Summaries are sufficient — no file reading needed
        summary_ctx = build_summary_context(project_root)
        if summary_ctx:
            context_parts.append(summary_ctx)

    # Add categories for aggregate questions
    if _is_aggregate(question):
        agg_ctx, agg_extra = _format_aggregate(question, project_root)
        if agg_ctx:
            context_parts.append(agg_ctx)
            extra_data.update(agg_extra)

    # Join context
    retrieved_context = "\n\n".join(p for p in context_parts if p.strip())

    # Guard: if no context found, try BM25 fallback
    if not retrieved_context.strip():
        bm25_ctx = retrieve_context(question, file_index, keyword_map, symbol_map, project_root)
        if bm25_ctx["context"]:
            retrieved_context = "=== SOURCE CODE ===\n" + bm25_ctx["context"]
            top_files = bm25_ctx["top_files"]
        else:
            return {
                "answer": (
                    "No relevant information found for this question. "
                    "Try using specific function names, class names, or file names from the codebase."
                ),
                "relevant_files": [],
                "confidence": "low",
                "follow_up": [],
                "session_id": session_id,
            }

    # Build conversation history block
    history_block = ""
    if history:
        parts = []
        for turn in history[-3:]:
            parts.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        history_block = (
            "Previous conversation:\n"
            + "\n---\n".join(parts)
            + "\n\n---\nNow answer the follow-up question below.\n\n"
        )

    # ── Step 3: Answer with drill-down option (1 LLM call) ──
    try:
        result = await qa_router.ai(
            system=ANSWER_SYSTEM,
            user=(
                f"{history_block}"
                f"Question: {question}\n\n"
                f"Context from the codebase:\n\n"
                f"{retrieved_context}"
            ),
            schema=AnswerWithDrilldown,
        )
    except Exception as e:
        logger.error(f"LLM answer synthesis failed: {e}")
        return {
            "answer": "The AI model is temporarily rate-limited. Please wait a moment and try again.",
            "relevant_files": top_files,
            "confidence": "low",
            "follow_up": [],
            "session_id": session_id,
        }

    # ── Step 4: Optional drill-down (0-1 LLM calls) ──
    if result.needs_more_context and result.additional_files:
        logger.info(f"Drill-down: reading {len(result.additional_files)} additional files")
        extra_context = read_files_by_paths(result.additional_files, file_index, project_root)
        if extra_context:
            try:
                result = await qa_router.ai(
                    system=ANSWER_SYSTEM,
                    user=(
                        f"Question: {question}\n\n"
                        f"Previous context:\n{retrieved_context}\n\n"
                        f"Additional files you requested:\n{extra_context}"
                    ),
                    schema=Answer,
                )
                top_files.extend(result.relevant_files)
            except Exception as e:
                logger.warning(f"Drill-down LLM call failed: {e}, using first-round answer")

    # Save this turn
    if session_id:
        save_session_turn(session_id, question, result.answer, top_files)

    # Determine confidence
    confidence = result.confidence if hasattr(result, "confidence") else "medium"

    qa_router.app.note(
        f"Q: {question[:80]} | Nav: needs_code={decision.needs_code}, files={len(decision.target_files)} | Top: {top_files[:3]}",
        tags=["qa", "query"]
    )

    return {
        **result.model_dump(),
        "relevant_files": list(dict.fromkeys(top_files))[:10],  # deduplicate
        "confidence": confidence,
        "session_id": session_id,
        "project_id": stored.get("project_id", ""),
        **extra_data,
    }


# ── Other Endpoints (unchanged) ──────────────────────────────────────────────

@qa_router.reasoner()
async def find_relevant_files(query: str, project_path: str = "") -> dict:
    """Return the most relevant files for a topic — pure keyword retrieval, no LLM."""
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


@qa_router.reasoner()
async def list_projects() -> dict:
    """List all indexed projects."""
    projects = list_indexed_projects()
    return {"projects": projects, "total": len(projects)}


@qa_router.reasoner()
async def get_file_content(file_path: str, project_path: str = "") -> dict:
    """Get the source code of a specific file from the index."""
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
    content = "\n".join(chunk["content"] for chunk in chunks)

    full_path = _Path(project_root) / file_path
    if full_path.exists():
        try:
            from skills.scanner import read_file
            fresh = read_file(str(full_path))
            if fresh.get("content"):
                content = fresh["content"]
        except Exception:
            pass

    return {
        "file_path": file_path,
        "project_id": stored.get("project_id", ""),
        "content": content,
        "line_count": content.count("\n") + 1 if content else 0,
        "symbols": meta.get("symbols", []),
        "keywords": meta.get("keywords", []),
        "extension": meta.get("extension", ""),
        "size_bytes": meta.get("size_bytes", 0),
        "chunks_count": len(chunks),
    }


@qa_router.reasoner()
async def list_project_files(project_path: str = "") -> dict:
    """List ALL files in an indexed project for file explorer UI."""
    stored = load_index(project_path)
    if not stored:
        return {"files": [], "total": 0, "error": "No index found."}

    file_index = stored["file_index"]
    files = [
        {
            "relative_path": rel_path,
            "extension": meta.get("extension", ""),
            "size_bytes": meta.get("size_bytes", 0),
        }
        for rel_path, meta in sorted(file_index.items())
    ]
    return {"files": files, "total": len(files)}


@qa_router.reasoner()
async def get_session_history(session_id: str) -> dict:
    """Load conversation history for a session."""
    if not session_id:
        return {"turns": [], "session_id": ""}
    turns = load_session(session_id, max_turns=50)
    return {"turns": turns, "session_id": session_id}


@qa_router.reasoner()
async def search_code(query: str, project_path: str = "") -> dict:
    """Grep-like code search across indexed files."""
    import re as _re
    stored = load_index(project_path)
    if not stored:
        return {"matches": [], "total": 0, "error": "No index found."}

    file_index = stored["file_index"]
    project_root = stored["project_root"]
    matches = []

    try:
        pattern = _re.compile(_re.escape(query), _re.IGNORECASE)
    except _re.error:
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
                matches.append({"file": rel_path, "line": i, "text": line.strip()})
                if len(matches) >= 50:
                    return {"matches": matches, "total": len(matches), "truncated": True}

    return {"matches": matches, "total": len(matches)}
