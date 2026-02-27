import time
from pathlib import Path
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.scanner import scan_directory, read_file
from skills.extractor import extract_symbols, extract_keywords, chunk_file
from skills.storage import save_index, load_index, delete_project as storage_delete_project

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

        indexer_router.app.note(
            f"Indexed: {rel_path} [{len(symbols)} symbols, {len(chunks)} chunks]",
            tags=["indexing", "progress"]
        )

    indexed_at = time.time()
    save_index(file_index, keyword_map, symbol_map, project_path, indexed_at)

    # Build and persist embeddings (optional, runs if sentence-transformers is installed)
    embeddings_count = 0
    if EMBEDDINGS_AVAILABLE:
        try:
            embeddings_count = build_and_save_embeddings(file_index, project_path)
            indexer_router.app.note(
                f"Built {embeddings_count} embeddings for semantic search",
                tags=["indexing", "embeddings"]
            )
        except Exception:
            pass  # Graceful degradation

    return IndexResult(
        files_indexed=len(file_index),
        project_root=project_path,
        indexed_at=indexed_at,
        message=f"Indexed {len(file_index)} files ({embeddings_count} embeddings). Ready to answer questions.",
    ).model_dump()


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

    new_timestamp = time.time()
    save_index(file_index, keyword_map, symbol_map, project_root, new_timestamp)

    return {
        "files_updated": len(changed),
        "files_deleted": len(deleted),
        "updated_files": [f["relative_path"] for f in changed],
        "deleted_files": list(deleted),
        "message": f"Re-indexed {len(changed)} changed, removed {len(deleted)} deleted file(s).",
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
        "github_url": github_url,
        "owner_repo": clone_result.get("owner_repo", ""),
        "clone_action": clone_result.get("action", ""),
    }


@indexer_router.reasoner()
async def delete_project(project_identifier: str) -> dict:
    """
    Delete an indexed project by path, slug, or project_id.

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_delete_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_identifier": "codebase-qa-agent"}}'
    """
    deleted = storage_delete_project(project_identifier)
    if deleted:
        # Also stop any active watcher
        try:
            from skills.watcher import stop_watching
            stop_watching(project_identifier)
        except ImportError:
            pass

    return {
        "deleted": deleted,
        "project_identifier": project_identifier,
        "message": "Project deleted successfully." if deleted else "Project not found.",
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
