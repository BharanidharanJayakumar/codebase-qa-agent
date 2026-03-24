"""
Summary agent — serves project-level intelligence (summary, categories, imports).
Enhanced with comprehensive project analysis: architecture detection, complexity
metrics, test coverage, infrastructure hints, and developer-focused insights.
"""
import json
from pydantic import BaseModel
from agentfield import AgentRouter

from skills.storage import (
    load_index,
    load_project_summary,
    load_symbol_categories,
    load_file_imports,
    load_module_summaries,
    load_semantic_summary,
    get_project_db_path,
)
from skills.aggregator import (
    build_comprehensive_report,
    detect_architecture_pattern,
    find_entry_points,
    compute_complexity_metrics,
    detect_infrastructure,
    analyze_test_coverage,
    detect_external_dependencies,
    find_top_connected_files,
)

summary_router = AgentRouter(prefix="summary", tags=["summary"])


@summary_router.reasoner()
async def get_project_overview(project_path: str) -> dict:
    """
    Return a comprehensive project overview with developer-focused insights.

    Includes: project description, architecture pattern, language breakdown,
    framework stack, entry points, complexity metrics, test coverage,
    infrastructure detection, dependency analysis, and actionable insights.
    """
    db_path = get_project_db_path(project_path)
    if not db_path:
        return {"error": "Project not indexed. Run index_project first."}

    # Load the full index for comprehensive analysis
    stored = load_index(project_path)
    if not stored:
        return {"error": "Could not load project index. Try re-indexing."}

    file_index = stored["file_index"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    # Load imports for connectivity analysis
    imports_data = load_file_imports(project_path)

    # Build comprehensive report (all pure Python, no LLM)
    report = build_comprehensive_report(
        project_root, file_index, symbol_map, imports_data
    )

    # Include semantic intelligence if available (from LLM enrichment)
    semantic = load_semantic_summary(project_path)
    modules = load_module_summaries(project_path)

    # Load basic summary for backward compatibility
    summary = load_project_summary(project_path)
    parsed_summary = {}
    if summary:
        for key, value in summary.items():
            try:
                parsed_summary[key] = json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                parsed_summary[key] = value

    return {
        "status": "ok",
        # Legacy fields for backward compatibility
        "summary": parsed_summary,
        "semantic": semantic if semantic else None,
        "modules": modules if modules else None,
        # New comprehensive report
        "report": report,
    }


@summary_router.reasoner()
async def get_project_report(project_path: str) -> dict:
    """
    Return a developer-focused project report with actionable insights.

    This is the enhanced summary endpoint that provides:
    - Project description and README excerpt
    - Architecture pattern detection (MVC, layered, microservices, etc.)
    - Language breakdown with percentages
    - Entry points and core files
    - Code complexity metrics (file sizes, symbol density)
    - Test coverage analysis with framework detection
    - Infrastructure and CI/CD detection
    - External dependency analysis
    - Top connected files (most imported)
    - Developer insights and recommendations
    """
    stored = load_index(project_path)
    if not stored:
        return {"error": "No index found. Run index_project first."}

    file_index = stored["file_index"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]
    imports_data = load_file_imports(project_path)

    report = build_comprehensive_report(
        project_root, file_index, symbol_map, imports_data
    )

    # Add semantic summary if available
    semantic = load_semantic_summary(project_path)
    if semantic:
        report["semantic_summary"] = semantic

    modules = load_module_summaries(project_path)
    if modules:
        report["module_summaries"] = modules

    report["status"] = "ok"
    report["project_id"] = stored.get("project_id", "")
    report["project_root"] = project_root

    return report


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
    for row in rows:
        if isinstance(row, dict):
            cat = row["category"]
            grouped.setdefault(cat, [])
            grouped[cat].append({
                "file": row["rel_path"],
                "symbol": row["symbol_name"],
                "detail": row["detail"],
            })
        else:
            rel_path, symbol_name, cat, detail = row
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
    for row in rows:
        if isinstance(row, dict):
            source = row["source_path"]
            name = row["imported_name"]
            target = row.get("target_path")
        else:
            source, name, target = row

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
