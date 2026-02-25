import time
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.scanner import scan_directory, read_file, get_changed_files
from skills.extractor import extract_symbols, extract_keywords
from skills.storage import save_index, load_index

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

    Scans all files, extracts keywords and symbols, stores raw content
    chunks in memory. Fast because we defer all LLM work to query time.

    This is the correct RAG pattern:
    - Index phase: instant (file reads + Python processing only)
    - Query phase: LLM sees only the 3-5 relevant files, not all 50

    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_index_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_path": "/home/you/myproject"}}'
    """
    files = scan_directory(project_path)

    file_index = {}    # rel_path → {keywords, symbols, content_chunk, last_modified}
    keyword_map = {}   # keyword → [rel_paths]
    symbol_map = {}    # symbol_name → {file, line, type}

    for file_meta in files:
        path = file_meta["path"]
        rel_path = file_meta["relative_path"]

        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        # Pure Python — no LLM, instant
        symbols = extract_symbols(content, path)
        keywords = extract_keywords(content)

        # Build keyword → files map
        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if rel_path not in keyword_map[kw]:
                keyword_map[kw].append(rel_path)

        # Build symbol → location map
        for sym in symbols:
            symbol_map[sym["name"]] = {
                "file": rel_path,
                "line": sym["line"],
                "type": sym["type"],
            }

        # Store a content chunk (first 4000 chars) — used at query time
        # We store raw content so the LLM gets real code, not a pre-baked summary
        file_index[rel_path] = {
            "content_chunk": content[:4000],
            "keywords": keywords,
            "symbols": [s["name"] for s in symbols],
            "extension": file_meta["extension"],
            "size_bytes": file_meta["size_bytes"],
            "last_modified": file_meta["last_modified"],
        }

        indexer_router.app.note(
            f"Indexed: {rel_path} [{len(symbols)} symbols, {len(keywords)} keywords]",
            tags=["indexing", "progress"]
        )

    indexed_at = time.time()

    save_index(file_index, keyword_map, symbol_map, project_path, indexed_at)

    return IndexResult(
        files_indexed=len(file_index),
        project_root=project_path,
        indexed_at=indexed_at,
        message=f"Indexed {len(file_index)} files in seconds. Ready to answer questions.",
    ).model_dump()


@indexer_router.reasoner()
async def update_index(project_path: str) -> dict:
    """
    Re-index only files changed since last run. Still no LLM calls.
    """
    stored = load_index()
    if not stored:
        return {"error": "No index found. Run index_project first.", "files_updated": 0}

    since = stored["indexed_at"]
    changed = get_changed_files(project_path, since_timestamp=since)

    if not changed:
        return {"message": "No files changed since last index.", "files_updated": 0}

    file_index = stored["file_index"]
    keyword_map = stored["keyword_map"]
    symbol_map = stored["symbol_map"]
    project_root = stored["project_root"]

    for file_meta in changed:
        path = file_meta["path"]
        rel_path = path.replace(project_root + "/", "")

        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        symbols = extract_symbols(content, path)
        keywords = extract_keywords(content)

        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if rel_path not in keyword_map[kw]:
                keyword_map[kw].append(rel_path)

        for sym in symbols:
            symbol_map[sym["name"]] = {"file": rel_path, "line": sym["line"], "type": sym["type"]}

        file_index[rel_path] = {
            "content_chunk": content[:4000],
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
        "updated_files": [f["relative_path"] for f in changed],
        "message": f"Re-indexed {len(changed)} changed file(s).",
    }
