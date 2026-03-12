"""
QA orchestrator agent — classifies query intent, delegates to retrieval or summary agents,
then synthesizes answers via LLM.
"""
import json
import re
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.storage import (
    load_index, load_session, save_session_turn, list_indexed_projects,
    load_project_summary, load_symbol_categories, load_file_imports,
)
from pathlib import Path as _Path
from reasoners.retrieval import retrieve_context

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])


class Answer(BaseModel):
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


# ── Intent classification ────────────────────────────────────────────────────

AGGREGATE_PATTERNS = [
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bcount\b.*\b(dto|route|test|service|model|endpoint|controller|middleware|config)", re.IGNORECASE),
    re.compile(r"\blist\s+(all|every)\b", re.IGNORECASE),
    re.compile(r"\ball\s+(dto|route|test|service|model|endpoint|controller|middleware)", re.IGNORECASE),
    re.compile(r"\bwhat\s+(dto|route|test|service|model|endpoint|api)\s*s?\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(me\s+)?(all|every)\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+are\s+(the\s+)?(all|every)\b", re.IGNORECASE),
]

OVERVIEW_PATTERNS = [
    re.compile(r"\bwhat\s+(is|does)\s+this\s+(project|app|repo|codebase)\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bsummar(y|ize)\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\btech\s*stack\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+language", re.IGNORECASE),
    re.compile(r"\bwhat\s+framework", re.IGNORECASE),
    re.compile(r"\bdirectory\s+structure\b", re.IGNORECASE),
    re.compile(r"\bproject\s+structure\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+this\b", re.IGNORECASE),
]


def _classify_query(question: str) -> str:
    """Classify query intent: 'aggregate', 'overview', or 'code_specific'."""
    for pattern in AGGREGATE_PATTERNS:
        if pattern.search(question):
            return "aggregate"
    for pattern in OVERVIEW_PATTERNS:
        if pattern.search(question):
            return "overview"
    return "code_specific"


def _retrieve_aggregate(question: str, project_path: str) -> dict:
    """Retrieve aggregated data from symbol_categories for aggregate queries."""
    question_lower = question.lower()

    # Detect which category the user is asking about
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

    # Group by category
    grouped: dict[str, list[dict]] = {}
    for rel_path, symbol_name, cat, detail in rows:
        grouped.setdefault(cat, [])
        grouped[cat].append({"file": rel_path, "symbol": symbol_name, "detail": detail})

    # Build context string for the LLM
    parts = []
    for cat, items in grouped.items():
        parts.append(f"\n=== {cat.upper()} ({len(items)} found) ===")
        for item in items[:30]:  # cap per category
            parts.append(f"  {item['symbol']} in {item['file']} ({item['detail']})")

    return {
        "context": "\n".join(parts),
        "categories": grouped,
        "total": sum(len(v) for v in grouped.values()),
        "filter": target_category,
    }


def _retrieve_overview(project_path: str) -> dict:
    """Retrieve project summary for overview queries."""
    raw_summary = load_project_summary(project_path)
    if not raw_summary:
        return {"context": "", "summary": {}}

    summary = {}
    for key, value in raw_summary.items():
        try:
            summary[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            summary[key] = value

    # Build context string for the LLM
    parts = []
    if summary.get("project_description"):
        parts.append(f"Project description: {summary['project_description']}")
    if summary.get("languages"):
        lang_str = ", ".join(f"{k}: {v} files" for k, v in summary["languages"].items())
        parts.append(f"Languages: {lang_str}")
    if summary.get("framework_hints"):
        parts.append(f"Frameworks: {', '.join(summary['framework_hints'])}")
    if summary.get("total_symbols"):
        sym_str = ", ".join(f"{v} {k}s" for k, v in summary["total_symbols"].items())
        parts.append(f"Symbols: {sym_str}")
    if summary.get("total_lines"):
        parts.append(f"Total lines of code: {summary['total_lines']}")
    if summary.get("dependency_files"):
        parts.append(f"Dependency files: {', '.join(summary['dependency_files'])}")
    if summary.get("directory_tree"):
        tree = summary["directory_tree"]
        dirs = ", ".join(f"{k}/ ({v['files']} files)" for k, v in list(tree.items())[:15])
        parts.append(f"Top directories: {dirs}")
    if summary.get("readme_content"):
        parts.append(f"\nREADME:\n{summary['readme_content'][:2000]}")

    return {
        "context": "\n".join(parts),
        "summary": summary,
    }


@qa_router.reasoner()
async def answer_question(question: str, session_id: str = "", project_path: str = "") -> dict:
    """
    Answer any question about the indexed codebase.
    Classifies intent (aggregate/overview/code-specific) and retrieves accordingly.
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

    # Classify query intent
    intent = _classify_query(question)

    # Enrich query for follow-ups
    enriched_query = question
    if history:
        prev_keywords = []
        for turn in history[-2:]:
            prev_keywords.extend(extract_keywords(turn["question"], top_n=5))
        enriched_query = f"{question} {' '.join(prev_keywords)}"

    # Route retrieval based on intent
    retrieved_context = ""
    top_files = []
    confidence = "medium"
    extra_data = {}

    if intent == "aggregate":
        agg = _retrieve_aggregate(enriched_query, project_root)
        retrieved_context = agg["context"]
        confidence = "high" if agg["total"] > 0 else "low"
        extra_data["aggregate_total"] = agg["total"]
        extra_data["aggregate_filter"] = agg["filter"]
        # Also get some code context for richer answers
        code_ctx = retrieve_context(enriched_query, file_index, keyword_map, symbol_map, project_root)
        top_files = code_ctx["top_files"]
        if code_ctx["context"]:
            retrieved_context += "\n\nRelevant source code:\n" + code_ctx["context"][:8000]

    elif intent == "overview":
        overview = _retrieve_overview(project_root)
        retrieved_context = overview["context"]
        confidence = "high" if overview["context"] else "low"
        extra_data["summary"] = overview.get("summary", {})

    else:  # code_specific
        code_ctx = retrieve_context(enriched_query, file_index, keyword_map, symbol_map, project_root)
        retrieved_context = code_ctx["context"]
        top_files = code_ctx["top_files"]
        confidence = code_ctx["confidence"]

    # Guard: if no context found
    if not retrieved_context.strip():
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

    # Build intent-aware system prompt
    intent_hints = {
        "aggregate": (
            "The user is asking an aggregate/counting question. "
            "Use the categorized symbol data provided to give precise counts and lists. "
            "Be specific about file locations."
        ),
        "overview": (
            "The user is asking about the project overview/architecture. "
            "Use the project summary data to give a comprehensive overview. "
            "Cover languages, frameworks, structure, and purpose."
        ),
        "code_specific": (
            "The user is asking about specific code. "
            "Use the source code context to give a detailed, accurate answer. "
            "Mention file names, function names, and line numbers."
        ),
    }

    result = await qa_router.ai(
        system=(
            "You are an expert software engineer helping a developer understand a codebase. "
            f"The project is at: {project_root}\n"
            f"{intent_hints.get(intent, '')}\n"
            "Answer questions using the data and source code provided. "
            "Be specific and accurate. If the context is insufficient, say so clearly."
        ),
        user=(
            f"{history_block}"
            f"Question: {question}\n\n"
            f"Data from the codebase:\n\n"
            f"{retrieved_context}"
        ),
        schema=Answer,
    )

    # Save this turn
    if session_id:
        save_session_turn(session_id, question, result.answer, top_files)

    qa_router.app.note(
        f"Q: {question[:80]} | Intent: {intent} | Files: {top_files} | Confidence: {confidence}",
        tags=["qa", "query"]
    )

    return {
        **result.model_dump(),
        "relevant_files": top_files,
        "confidence": confidence,
        "session_id": session_id,
        "project_id": stored.get("project_id", ""),
        "intent": intent,
        **extra_data,
    }


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
