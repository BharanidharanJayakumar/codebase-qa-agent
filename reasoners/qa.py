"""
QA orchestrator agent — LLM-driven query planning, multi-source retrieval,
and intelligent answer synthesis.
"""
import json
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

qa_router = AgentRouter(prefix="qa", tags=["question-answering"])


class Answer(BaseModel):
    answer: str = Field(description="Clear, direct answer to the question")
    relevant_files: list[str] = Field(description="Files most relevant to this question")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")


# ── Query Planning ────────────────────────────────────────────────────────────

class QueryPlan(BaseModel):
    intent: str = Field(description="One of: aggregate, overview, code_specific, architecture")
    data_sources: list[str] = Field(description="Sources to query: semantic_summary, module_summaries, code_chunks, categories, imports, metadata")
    search_terms: list[str] = Field(description="Key terms for code retrieval if code_chunks is selected")
    reasoning: str = Field(description="Brief reason for this plan")


QUERY_PLANNER_SYSTEM = """You are a query planner for a codebase Q&A system. Given the user's question, decide what data sources to query.

Available data sources:
- **semantic_summary**: LLM-generated understanding of the entire project (purpose, architecture, patterns, domain, data flow). Use for "what does this project do?", "what's the theme?", "describe the architecture".
- **module_summaries**: LLM-generated summaries per directory/module (purpose, patterns, domain concepts). Use for questions about specific modules or when understanding project structure.
- **code_chunks**: Actual source code retrieved via keyword search. Use for specific code questions, "how does X work?", "what does function Y do?".
- **categories**: Categorized symbols (DTOs, routes, tests, services, config, middleware). Use for "how many routes?", "list all DTOs", "show me the services".
- **imports**: File dependency graph. Use for "what depends on X?", "show import relationships".
- **metadata**: Basic stats (languages, file counts, frameworks, directory tree). Use as supplementary data.

Rules:
- For overview/theme/purpose questions: always include semantic_summary + module_summaries
- For code-specific questions: always include code_chunks
- For counting/listing questions: always include categories
- You can select multiple sources for richer answers
- Include search_terms only when code_chunks is selected"""


async def _plan_query(question: str) -> QueryPlan:
    """Use LLM to create a query plan — decides what data sources to fetch."""
    try:
        result = await qa_router.ai(
            system=QUERY_PLANNER_SYSTEM,
            user=question,
            schema=QueryPlan,
            max_tokens=200,
        )
        # Validate intent
        valid_intents = {"aggregate", "overview", "code_specific", "architecture"}
        if result.intent not in valid_intents:
            result.intent = "code_specific"
        # Validate data sources
        valid_sources = {"semantic_summary", "module_summaries", "code_chunks", "categories", "imports", "metadata"}
        result.data_sources = [s for s in result.data_sources if s in valid_sources]
        if not result.data_sources:
            result.data_sources = ["code_chunks"]
        return result
    except Exception:
        return _plan_query_fallback(question)


def _plan_query_fallback(question: str) -> QueryPlan:
    """Regex fallback for query planning when LLM is unavailable."""
    q_lower = question.lower()

    # Aggregate patterns
    aggregate_words = ["how many", "count", "list all", "list every", "show all", "show every"]
    if any(w in q_lower for w in aggregate_words):
        return QueryPlan(
            intent="aggregate",
            data_sources=["categories", "metadata"],
            search_terms=[],
            reasoning="Aggregate question detected via keywords",
        )

    # Overview patterns
    overview_words = ["overview", "summary", "summarize", "what is this", "what does this",
                      "theme", "purpose", "about this", "describe this", "this project",
                      "this codebase", "this repo", "tech stack", "what language", "what framework"]
    if any(w in q_lower for w in overview_words):
        return QueryPlan(
            intent="overview",
            data_sources=["semantic_summary", "module_summaries", "metadata"],
            search_terms=[],
            reasoning="Overview question detected via keywords",
        )

    # Default: code-specific
    terms = extract_keywords(question, top_n=5)
    return QueryPlan(
        intent="code_specific",
        data_sources=["code_chunks"],
        search_terms=terms,
        reasoning="Defaulting to code-specific retrieval",
    )


# ── Multi-Source Retrieval ────────────────────────────────────────────────────

def _format_semantic_summary(data: dict) -> str:
    """Format semantic summary for LLM context."""
    if not data:
        return ""
    parts = ["=== PROJECT UNDERSTANDING ==="]
    for key in ["purpose", "architecture", "domain", "data_flow", "tech_decisions"]:
        if data.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: {data[key]}")
    if data.get("key_patterns"):
        try:
            patterns = json.loads(data["key_patterns"]) if isinstance(data["key_patterns"], str) else data["key_patterns"]
            parts.append(f"Key Patterns: {', '.join(patterns)}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"Key Patterns: {data['key_patterns']}")
    return "\n".join(parts)


def _format_module_summaries(modules: list[dict], search_terms: list[str] | None = None) -> str:
    """Format module summaries for LLM context, optionally filtered by relevance."""
    if not modules:
        return ""

    # If search terms provided, prioritize relevant modules
    if search_terms:
        def relevance(mod):
            text = f"{mod['summary']} {' '.join(mod.get('domain_concepts', []))} {' '.join(mod.get('key_abstractions', []))}".lower()
            return sum(1 for t in search_terms if t.lower() in text)
        modules = sorted(modules, key=relevance, reverse=True)

    parts = ["=== MODULE SUMMARIES ==="]
    for mod in modules[:10]:  # cap at 10 modules
        parts.append(f"\n{mod['module_path']}/: {mod['summary']}")
        if mod.get("key_patterns"):
            parts.append(f"  Patterns: {', '.join(mod['key_patterns'][:5])}")
        if mod.get("domain_concepts"):
            parts.append(f"  Domain: {', '.join(mod['domain_concepts'][:5])}")
        if mod.get("key_abstractions"):
            parts.append(f"  Key: {', '.join(mod['key_abstractions'][:5])}")
    return "\n".join(parts)


def _format_metadata(project_path: str) -> str:
    """Format basic project metadata for LLM context."""
    raw = load_project_summary(project_path)
    if not raw:
        return ""

    parts = ["=== PROJECT METADATA ==="]
    if raw.get("languages"):
        langs = raw["languages"]
        if isinstance(langs, str):
            try:
                langs = json.loads(langs)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(langs, dict):
            lang_str = ", ".join(f"{k}: {v} files" for k, v in langs.items())
            parts.append(f"Languages: {lang_str}")
    if raw.get("framework_hints"):
        hints = raw["framework_hints"]
        if isinstance(hints, str):
            try:
                hints = json.loads(hints)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(hints, list) and hints:
            parts.append(f"Frameworks: {', '.join(hints)}")
    if raw.get("total_lines"):
        parts.append(f"Total lines of code: {raw['total_lines']}")
    if raw.get("total_symbols"):
        syms = raw["total_symbols"]
        if isinstance(syms, str):
            try:
                syms = json.loads(syms)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(syms, dict):
            sym_str = ", ".join(f"{v} {k}s" for k, v in syms.items())
            parts.append(f"Symbols: {sym_str}")
    if raw.get("directory_tree"):
        tree = raw["directory_tree"]
        if isinstance(tree, str):
            try:
                tree = json.loads(tree)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(tree, dict):
            dirs = ", ".join(f"{k}/ ({v['files']} files)" for k, v in list(tree.items())[:15] if isinstance(v, dict))
            parts.append(f"Directories: {dirs}")
    if raw.get("readme_content"):
        parts.append(f"\nREADME:\n{str(raw['readme_content'])[:1500]}")
    return "\n".join(parts)


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
        parts.append(f"\n{cat.upper()} ({len(items)} found):")
        for item in items[:30]:
            parts.append(f"  {item['symbol']} in {item['file']} ({item['detail']})")

    total = sum(len(v) for v in grouped.values())
    extra = {"aggregate_total": total, "aggregate_filter": target_category}
    return "\n".join(parts), extra


def _format_imports(project_path: str) -> str:
    """Format import graph for LLM context."""
    imports = load_file_imports(project_path)
    if not imports:
        return ""

    parts = ["=== IMPORT GRAPH ==="]
    by_source: dict[str, list[str]] = {}
    for imp in imports[:100]:  # cap
        source = imp["source_path"] if isinstance(imp, dict) else imp[0]
        target = imp.get("imported_name", "") if isinstance(imp, dict) else imp[1]
        by_source.setdefault(source, []).append(target)

    for source, targets in list(by_source.items())[:20]:
        parts.append(f"  {source} imports: {', '.join(targets[:8])}")
    return "\n".join(parts)


MAX_CONTEXT_CHARS = 24_000


async def _execute_query_plan(
    plan: QueryPlan,
    question: str,
    project_root: str,
    file_index: dict,
    keyword_map: dict,
    symbol_map: dict,
) -> tuple[str, list[str], str, dict]:
    """
    Execute a query plan by fetching from specified data sources.
    Returns (context, top_files, confidence, extra_data).
    """
    context_parts = []
    top_files = []
    confidence = "medium"
    extra_data = {}

    if "semantic_summary" in plan.data_sources:
        sem = load_semantic_summary(project_root)
        fmt = _format_semantic_summary(sem)
        if fmt:
            context_parts.append(fmt)
            confidence = "high"

    if "module_summaries" in plan.data_sources:
        mods = load_module_summaries(project_root)
        fmt = _format_module_summaries(mods, plan.search_terms or None)
        if fmt:
            context_parts.append(fmt)

    if "categories" in plan.data_sources:
        fmt, agg_extra = _format_aggregate(question, project_root)
        if fmt:
            context_parts.append(fmt)
            extra_data.update(agg_extra)
            confidence = "high" if agg_extra.get("aggregate_total", 0) > 0 else "low"

    if "imports" in plan.data_sources:
        fmt = _format_imports(project_root)
        if fmt:
            context_parts.append(fmt)

    if "metadata" in plan.data_sources:
        fmt = _format_metadata(project_root)
        if fmt:
            context_parts.append(fmt)

    if "code_chunks" in plan.data_sources:
        search_query = " ".join(plan.search_terms) if plan.search_terms else question
        code_ctx = retrieve_context(search_query, file_index, keyword_map, symbol_map, project_root)
        if code_ctx["context"]:
            context_parts.append("=== SOURCE CODE ===\n" + code_ctx["context"])
        top_files = code_ctx["top_files"]
        if not confidence or confidence == "medium":
            confidence = code_ctx["confidence"]

    # Fallback: if no context gathered, try code retrieval
    if not any(p.strip() for p in context_parts):
        code_ctx = retrieve_context(question, file_index, keyword_map, symbol_map, project_root)
        if code_ctx["context"]:
            context_parts.append("=== SOURCE CODE ===\n" + code_ctx["context"])
        top_files = code_ctx["top_files"]
        confidence = code_ctx["confidence"]

    # Join and cap total context
    full_context = "\n\n".join(p for p in context_parts if p.strip())
    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS] + "\n... (context truncated)"

    return full_context, top_files, confidence, extra_data


# ── Intent-Aware System Prompts ───────────────────────────────────────────────

INTENT_PROMPTS = {
    "aggregate": (
        "The user is asking an aggregate/counting question. "
        "Use the categorized symbol data to give precise counts and lists. "
        "Be specific about file locations."
    ),
    "overview": (
        "The user is asking about the project overview, theme, or purpose. "
        "Use the project understanding and module summaries to give a comprehensive, "
        "insightful answer about what this project does, its architecture, and design. "
        "Go beyond just listing languages — explain the purpose and domain."
    ),
    "code_specific": (
        "The user is asking about specific code. "
        "Use the source code context to give a detailed, accurate answer. "
        "Mention file names, function names, and line numbers."
    ),
    "architecture": (
        "The user is asking about the project's architecture and design. "
        "Use the module summaries and semantic understanding to explain how "
        "the system is structured, the design patterns used, and how modules interact."
    ),
}


# ── Main Answer Endpoint ──────────────────────────────────────────────────────

@qa_router.reasoner()
async def answer_question(question: str, session_id: str = "", project_path: str = "") -> dict:
    """
    Answer any question about the indexed codebase.
    Uses LLM-driven query planning and multi-source retrieval.
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

    # Step 1: Plan the query (LLM decides what data to fetch)
    plan = await _plan_query(question)

    # Enrich search terms with conversation context
    if history and plan.search_terms:
        for turn in history[-2:]:
            plan.search_terms.extend(extract_keywords(turn["question"], top_n=3))

    # Step 2: Execute the plan — multi-source retrieval
    retrieved_context, top_files, confidence, extra_data = await _execute_query_plan(
        plan, question, project_root, file_index, keyword_map, symbol_map
    )

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

    # Step 3: Synthesize answer (LLM call)
    result = await qa_router.ai(
        system=(
            "You are an expert software engineer helping a developer understand a codebase. "
            f"The project is at: {project_root}\n"
            f"{INTENT_PROMPTS.get(plan.intent, INTENT_PROMPTS['code_specific'])}\n"
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
        f"Q: {question[:80]} | Plan: {plan.intent}/{plan.data_sources} | Files: {top_files}",
        tags=["qa", "query"]
    )

    return {
        **result.model_dump(),
        "relevant_files": top_files,
        "confidence": confidence,
        "session_id": session_id,
        "project_id": stored.get("project_id", ""),
        "intent": plan.intent,
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
