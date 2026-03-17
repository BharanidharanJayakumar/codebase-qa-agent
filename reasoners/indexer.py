import asyncio
import copy
import time
from pathlib import Path
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.scanner import scan_directory, read_file
from skills.extractor import extract_symbols, extract_keywords, chunk_file
from skills.storage import save_index, load_index, delete_project as storage_delete_project
from skills.aggregator import build_project_summary, extract_imports, categorize_symbols
from skills.summarizer import generate_hierarchical_summary

# Embeddings are optional — built at index time if available
try:
    from skills.embeddings import build_and_save_embeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

indexer_router = AgentRouter(prefix="indexer", tags=["indexing"])


class IndexResult(BaseModel):
    files_indexed: int
    project_root: str
    indexed_at: float
    message: str
    status: str = "ready"  # "ready" = basic index done, "enriching" = LLM running in background


# Track enrichment status per project
_enrichment_status: dict[str, str] = {}  # project_path -> "enriching" | "complete" | "failed"


@indexer_router.reasoner()
async def index_project(project_path: str) -> dict:
    """
    Index a project directory — pure Python, no LLM calls.

    Scans all files, extracts keywords and symbols, chunks file content
    at function/class boundaries for precise retrieval at query time.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_index_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_path": "/home/you/myproject"}}'
    """
    # Validate input path
    project = Path(project_path).resolve()
    if not project.exists():
        return {"error": f"Path does not exist: {project_path}", "files_indexed": 0}
    if not project.is_dir():
        return {"error": f"Path is not a directory: {project_path}", "files_indexed": 0}

    files = scan_directory(project_path)

    if not files:
        return IndexResult(
            files_indexed=0,
            project_root=str(project),
            indexed_at=time.time(),
            message="No indexable source files found in this directory.",
        ).model_dump()

    file_index = {}    # rel_path → {chunks, keywords, symbols, ...}
    keyword_map = {}   # keyword → [rel_paths]
    symbol_map = {}    # symbol_name → [{file, line, type}, ...]
    all_imports = []   # (source_path, imported_name, target_path|None)
    all_categories = []  # (rel_path, symbol_name, category, detail)
    project_files = {f["relative_path"] for f in files}

    for file_meta in files:
        path = file_meta["path"]
        rel_path = file_meta["relative_path"]

        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        symbols = extract_symbols(content, path)
        keywords = extract_keywords(content)
        chunks = chunk_file(content, symbols)

        # Build keyword → files map
        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if rel_path not in keyword_map[kw]:
                keyword_map[kw].append(rel_path)

        # Build symbol → locations map (one-to-many: multiple files can define same name)
        for sym in symbols:
            entry = {"file": rel_path, "line": sym["line"], "type": sym["type"]}
            symbol_map.setdefault(sym["name"], [])
            symbol_map[sym["name"]].append(entry)

        file_index[rel_path] = {
            "chunks": chunks,
            "keywords": keywords,
            "symbols": [s["name"] for s in symbols],
            "extension": file_meta["extension"],
            "size_bytes": file_meta["size_bytes"],
            "last_modified": file_meta["last_modified"],
        }

        # Extract imports for project-level intelligence
        file_imports = extract_imports(content, rel_path, project_files)
        all_imports.extend(file_imports)

        # Regex categorization as fallback (will be replaced by LLM below)
        file_categories = categorize_symbols(rel_path, content, symbols)
        all_categories.extend(file_categories)

        indexer_router.app.note(
            f"Indexed: {rel_path} [{len(symbols)} symbols, {len(chunks)} chunks]",
            tags=["indexing", "progress"]
        )

    # ── Phase 1: Save basic index immediately (fast — user gets instant response) ──
    project_summary = build_project_summary(project_path, file_index, symbol_map)
    indexed_at = time.time()
    save_index(
        file_index, keyword_map, symbol_map, project_path, indexed_at,
        project_summary=project_summary,
        imports_data=all_imports,
        categories_data=all_categories,
    )

    indexer_router.app.note(
        f"Phase 1 complete: {len(file_index)} files indexed. Starting LLM enrichment in background...",
        tags=["indexing", "phase1"]
    )

    # ── Phase 2: LLM enrichment runs in background (doesn't block response) ──
    _enrichment_status[project_path] = "enriching"

    bg_file_index = copy.deepcopy(file_index)
    bg_keyword_map = copy.deepcopy(keyword_map)
    bg_symbol_map = copy.deepcopy(symbol_map)
    bg_imports = list(all_imports)
    bg_categories = list(all_categories)

    async def _enrich_index_background():
        """Background task: generate hierarchical folder summaries + project synthesis."""
        try:
            # Hierarchical Summaries (the only LLM work at index time)
            module_sums = []
            semantic_sum = {}
            try:
                module_sums, semantic_sum = await generate_hierarchical_summary(
                    bg_file_index, bg_symbol_map, project_path, bg_imports, indexer_router
                )
                if module_sums:
                    indexer_router.app.note(
                        f"Generated {len(module_sums)} module summaries + project synthesis",
                        tags=["indexing", "summarizer"]
                    )
            except Exception as e:
                indexer_router.app.note(
                    f"Summarization skipped: {e}",
                    tags=["indexing", "summarizer", "warning"]
                )

            # Save enriched index with summaries
            save_index(
                bg_file_index, bg_keyword_map, bg_symbol_map, project_path,
                time.time(),
                project_summary=project_summary,
                imports_data=bg_imports,
                categories_data=bg_categories,
                module_summaries_data=module_sums,
                semantic_summary_data=semantic_sum,
            )

            _enrichment_status[project_path] = "complete"
            indexer_router.app.note(
                "Phase 2 complete: summaries saved. Index fully upgraded.",
                tags=["indexing", "phase2", "complete"]
            )

        except Exception as e:
            _enrichment_status[project_path] = "failed"
            indexer_router.app.note(
                f"Background enrichment failed: {e}",
                tags=["indexing", "phase2", "error"]
            )

    # Fire and forget — the background task enriches the index while user can already query
    asyncio.create_task(_enrich_index_background())

    # Build and persist embeddings (optional)
    embeddings_count = 0
    if EMBEDDINGS_AVAILABLE:
        try:
            embeddings_count = build_and_save_embeddings(file_index, project_path)
        except Exception:
            pass

    return IndexResult(
        files_indexed=len(file_index),
        project_root=project_path,
        indexed_at=indexed_at,
        message=f"Indexed {len(file_index)} files. LLM enrichment running in background.",
        status="enriching",
    ).model_dump()


@indexer_router.reasoner()
async def get_enrichment_status(project_path: str = "") -> dict:
    """Check the LLM enrichment status for a project."""
    if not project_path:
        # Return all statuses
        return {"statuses": dict(_enrichment_status)}

    status = _enrichment_status.get(project_path, "unknown")

    # Also check if semantic_summary exists in DB (enrichment may have completed in a prior run)
    if status == "unknown":
        from skills.storage import load_semantic_summary
        sem = load_semantic_summary(project_path)
        if sem:
            status = "complete"

    return {"project_path": project_path, "enrichment_status": status}


@indexer_router.reasoner()
async def update_index(project_path: str) -> dict:
    """
    Incrementally update the index: re-index changed files, remove deleted ones.
    """
    stored = load_index(project_path)
    if not stored:
        return {"error": "No index found. Run index_project first.", "files_updated": 0}

    since = stored["indexed_at"]
    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    # Detect deleted files: files in index but no longer on disk
    current_files = scan_directory(project_path)
    current_rel_paths = {f["relative_path"] for f in current_files}
    deleted = set(file_index.keys()) - current_rel_paths

    for rel_path in deleted:
        _purge_file_from_maps(rel_path, file_index, keyword_map, symbol_map)

    # Detect changed files
    changed = [f for f in current_files if f["last_modified"] > since]

    for file_meta in changed:
        path = file_meta["path"]
        rel_path = str(Path(path).relative_to(project_root))

        # Clean stale entries before re-adding
        _purge_file_from_maps(rel_path, file_index, keyword_map, symbol_map)

        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        symbols = extract_symbols(content, path)
        keywords = extract_keywords(content)
        chunks = chunk_file(content, symbols)

        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if rel_path not in keyword_map[kw]:
                keyword_map[kw].append(rel_path)

        for sym in symbols:
            entry = {"file": rel_path, "line": sym["line"], "type": sym["type"]}
            symbol_map.setdefault(sym["name"], [])
            symbol_map[sym["name"]].append(entry)

        file_index[rel_path] = {
            "chunks": chunks,
            "keywords": keywords,
            "symbols": [s["name"] for s in symbols],
            "extension": file_meta["extension"],
            "size_bytes": file_meta["size_bytes"],
            "last_modified": file_meta["last_modified"],
        }

        indexer_router.app.note(f"Updated: {rel_path}", tags=["update"])

    # Rebuild project-level intelligence after update
    all_imports = []
    all_categories = []
    all_project_files = set(file_index.keys())

    for rel_path, meta in file_index.items():
        fpath = Path(project_root) / rel_path
        if not fpath.exists():
            continue
        try:
            content = fpath.read_text(errors="replace")
        except Exception:
            continue

        symbols = extract_symbols(content, str(fpath))
        file_imports = extract_imports(content, rel_path, all_project_files)
        all_imports.extend(file_imports)
        file_categories = categorize_symbols(rel_path, content, symbols)
        all_categories.extend(file_categories)

    # Save basic index immediately
    project_summary = build_project_summary(project_root, file_index, symbol_map)
    new_timestamp = time.time()
    save_index(
        file_index, keyword_map, symbol_map, project_root, new_timestamp,
        project_summary=project_summary,
        imports_data=all_imports,
        categories_data=all_categories,
    )

    # LLM enrichment in background (same pattern as index_project)
    bg_fi = copy.deepcopy(file_index)
    bg_kw = copy.deepcopy(keyword_map)
    bg_sm = copy.deepcopy(symbol_map)
    bg_imp = list(all_imports)
    bg_cats = list(all_categories)

    async def _enrich_update_background():
        try:
            mod_sums, sem_sum = [], {}
            try:
                mod_sums, sem_sum = await generate_hierarchical_summary(
                    bg_fi, bg_sm, project_root, bg_imp, indexer_router
                )
            except Exception:
                pass

            save_index(
                bg_fi, bg_kw, bg_sm, project_root, time.time(),
                project_summary=project_summary,
                imports_data=bg_imp,
                categories_data=bg_cats,
                module_summaries_data=mod_sums,
                semantic_summary_data=sem_sum,
            )
            indexer_router.app.note(
                "Update enrichment complete.", tags=["update", "phase2", "complete"]
            )
        except Exception:
            pass

    asyncio.create_task(_enrich_update_background())

    return {
        "files_updated": len(changed),
        "files_deleted": len(deleted),
        "updated_files": [f["relative_path"] for f in changed],
        "deleted_files": list(deleted),
        "message": f"Re-indexed {len(changed)} changed, removed {len(deleted)} deleted. LLM enrichment running in background.",
    }


@indexer_router.reasoner()
async def watch_project(project_path: str) -> dict:
    """
    Start watching a project for file changes. Automatically re-indexes when files change.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_watch_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_path": "/home/you/myproject"}}'
    """
    try:
        from skills.watcher import start_watching, list_watchers
    except ImportError:
        return {"error": "watchfiles not installed. Run: pip install watchfiles", "watching": False}

    project = Path(project_path).resolve()
    if not project.is_dir():
        return {"error": f"Not a directory: {project_path}", "watching": False}

    async def _on_change(path):
        await update_index(path)
        indexer_router.app.note(f"Auto-updated index for {path}", tags=["watcher"])

    watcher_id = await start_watching(str(project), _on_change)
    return {
        "watching": bool(watcher_id),
        "project_path": str(project),
        "active_watchers": list_watchers(),
    }


@indexer_router.reasoner()
async def unwatch_project(project_path: str) -> dict:
    """
    Stop watching a project directory.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_unwatch_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_path": "/home/you/myproject"}}'
    """
    try:
        from skills.watcher import stop_watching, list_watchers
    except ImportError:
        return {"stopped": False}

    stopped = stop_watching(project_path)
    return {
        "stopped": stopped,
        "active_watchers": list_watchers(),
    }


@indexer_router.reasoner()
async def clone_and_index(github_url: str) -> dict:
    """
    Clone a GitHub repository and index it. Accepts full URLs or owner/repo shorthand.
    If already cloned, pulls latest changes and re-indexes.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_clone_and_index \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"github_url": "https://github.com/owner/repo"}}'
    """
    from skills.git_ops import clone_repo

    clone_result = clone_repo(github_url)
    if "error" in clone_result:
        return {"error": clone_result["error"], "files_indexed": 0}

    project_path = clone_result["path"]
    index_result = await index_project(project_path)

    return {
        **index_result,
        "project_root": project_path,
        "github_url": github_url,
        "owner_repo": clone_result.get("owner_repo", ""),
        "clone_action": clone_result.get("action", ""),
    }


@indexer_router.reasoner()
async def delete_project(project_identifier: str, delete_repo: bool = False) -> dict:
    """
    Delete an indexed project by path, slug, or project_id.
    Set delete_repo=true to also remove the cloned repository from disk.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_delete_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_identifier": "codebase-qa-agent", "delete_repo": true}}'
    """
    # Find the project root before deleting the DB
    import shutil
    from skills.storage import resolve_project_db
    import sqlite3

    repo_path = None
    db_path = resolve_project_db(project_identifier)
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT value FROM meta WHERE key='project_root'").fetchone()
            if row:
                repo_path = row[0]
            conn.close()
        except Exception:
            pass

    deleted = storage_delete_project(project_identifier)

    repo_deleted = False
    if deleted and delete_repo and repo_path:
        repo = Path(repo_path)
        if repo.exists() and repo.is_dir():
            try:
                shutil.rmtree(str(repo))
                repo_deleted = True
            except Exception:
                pass

    if deleted:
        # Also stop any active watcher
        try:
            from skills.watcher import stop_watching
            stop_watching(project_identifier)
        except ImportError:
            pass

    return {
        "deleted": deleted,
        "repo_deleted": repo_deleted,
        "project_identifier": project_identifier,
        "message": (
            "Project and repo deleted successfully." if repo_deleted
            else "Project deleted successfully." if deleted
            else "Project not found."
        ),
    }


def _purge_file_from_maps(rel_path: str, file_index: dict, keyword_map: dict, symbol_map: dict) -> None:
    """Remove all traces of a file from file_index, keyword_map, and symbol_map."""
    file_index.pop(rel_path, None)

    # Remove from keyword map
    for kw in list(keyword_map.keys()):
        if rel_path in keyword_map[kw]:
            keyword_map[kw].remove(rel_path)
        if not keyword_map[kw]:
            del keyword_map[kw]

    # Remove from symbol map (one-to-many: filter out entries for this file)
    for name in list(symbol_map.keys()):
        symbol_map[name] = [e for e in symbol_map[name] if e["file"] != rel_path]
        if not symbol_map[name]:
            del symbol_map[name]
