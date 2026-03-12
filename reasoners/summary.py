"""
Summary agent — serves project-level intelligence (summary, categories, imports).
No LLM calls; reads from pre-computed SQLite tables populated at index time.
"""
import json
from pydantic import BaseModel
from agentfield import AgentRouter

from skills.storage import (
    load_project_summary,
    load_symbol_categories,
    load_file_imports,
    get_project_db_path,
)

summary_router = AgentRouter(prefix="summary", tags=["summary"])


@summary_router.reasoner()
async def get_project_overview(project_path: str) -> dict:
    """
    Return full project summary: languages, frameworks, dir tree, README, stats.
    """
    db_path = get_project_db_path(project_path)
    if not db_path:
        return {"error": "Project not indexed. Run index_project first."}

    summary = load_project_summary(project_path)
    if not summary:
        return {"error": "No project summary found. Re-index to generate."}

    # Parse JSON values back to Python objects
    result = {}
    for key, value in summary.items():
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            result[key] = value

    return {"status": "ok", "summary": result}


@summary_router.reasoner()
async def query_categories(
    project_path: str,
    category: str | None = None,
) -> dict:
    """
    Query categorized symbols. Optional category filter: dto, route, test, service, config, middleware.
    """
    db_path = get_project_db_path(project_path)
    if not db_path:
        return {"error": "Project not indexed."}

    rows = load_symbol_categories(project_path, category=category)
    # Group by category
    grouped: dict[str, list[dict]] = {}
    for rel_path, symbol_name, cat, detail in rows:
        grouped.setdefault(cat, [])
        grouped[cat].append({
            "file": rel_path,
            "symbol": symbol_name,
            "detail": detail,
        })

    return {
        "status": "ok",
        "filter": category,
        "categories": grouped,
        "total": sum(len(v) for v in grouped.values()),
    }


@summary_router.reasoner()
async def query_imports(
    project_path: str,
    file_path: str | None = None,
) -> dict:
    """
    Query import graph. Optional file_path filter to get imports for a specific file.
    """
    db_path = get_project_db_path(project_path)
    if not db_path:
        return {"error": "Project not indexed."}

    rows = load_file_imports(project_path, source_path=file_path)

    imports = []
    internal_count = 0
    external_count = 0
    for source, name, target in rows:
        is_internal = target is not None
        if is_internal:
            internal_count += 1
        else:
            external_count += 1
        imports.append({
            "source": source,
            "imported": name,
            "target": target,
            "internal": is_internal,
        })

    return {
        "status": "ok",
        "filter_file": file_path,
        "imports": imports,
        "internal_count": internal_count,
        "external_count": external_count,
        "total": len(imports),
    }
