import time
import json
from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.scanner import scan_directory, read_file, get_changed_files
from skills.extractor import extract_symbols, extract_keywords

# The prefix "indexer" means all endpoints from this router will be:
# POST /execute/codebase-qa-agent.indexer_index_project
# POST /execute/codebase-qa-agent.indexer_update_index
indexer_router = AgentRouter(prefix="indexer", tags=["indexing"])


class FileSummary(BaseModel):
    """Structured output the LLM must return for each file it summarizes."""
    summary: str = Field(description="1-2 sentence plain English description of what this file does")
    purpose: str = Field(description="One word category: auth, database, api, ui, config, util, test, model, etc.")
    dependencies: list[str] = Field(description="Key imports or modules this file depends on")


class IndexResult(BaseModel):
    """What we return to the caller after indexing completes."""
    files_indexed: int
    project_root: str
    indexed_at: float
    message: str


@indexer_router.reasoner()
async def index_project(project_path: str) -> dict:
    """
    Full index of a project directory.

    Walk all files → read each one → LLM summarizes → everything stored in
    Agentfield memory. After this runs once, answer_question never touches
    the filesystem again — it reads from memory only.

    Call this:
    curl -X POST http://localhost:8080/api/v1/execute/codebase-qa-agent.indexer_index_project \\
      -H "Content-Type: application/json" \\
      -d '{"input": {"project_path": "/home/you/myproject"}}'
    """
    files = scan_directory(project_path)

    file_summaries = {}
    keyword_map = {}    # keyword → [file_paths]
    symbol_map = {}     # symbol_name → {file, line, type}

    for file_meta in files:
        path = file_meta["path"]
        rel_path = file_meta["relative_path"]

        # Read the raw file content using our Skill
        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        # --- LLM call: summarize this file ---
        # We pass only the first 3000 chars to keep the prompt lean.
        # A good summary doesn't need the entire file — just enough context.
        snippet = content[:3000]
        summary_result = await indexer_router.ai(
            system=(
                "You are a senior software engineer analyzing a codebase. "
                "Summarize source files concisely and accurately."
            ),
            user=(
                f"File: {rel_path}\n\n"
                f"```\n{snippet}\n```\n\n"
                "Summarize what this file does, its purpose category, "
                "and its key dependencies."
            ),
            schema=FileSummary,
        )

        # Extract symbols and keywords using pure Python (no LLM needed here)
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

        # Store per-file summary
        file_summaries[rel_path] = {
            "summary": summary_result.summary,
            "purpose": summary_result.purpose,
            "dependencies": summary_result.dependencies,
            "keywords": keywords,
            "last_modified": file_meta["last_modified"],
        }

        # Log progress so you can watch it work in real time
        indexer_router.app.note(
            f"Indexed: {rel_path} [{summary_result.purpose}]",
            tags=["indexing", "progress"]
        )

    indexed_at = time.time()

    # --- Store everything in Agentfield persistent memory ---
    # These keys persist between restarts — no re-indexing needed next session
    await indexer_router.app.memory.set("project_root", project_path)
    await indexer_router.app.memory.set("file_summaries", json.dumps(file_summaries))
    await indexer_router.app.memory.set("keyword_map", json.dumps(keyword_map))
    await indexer_router.app.memory.set("symbol_map", json.dumps(symbol_map))
    await indexer_router.app.memory.set("indexed_at", str(indexed_at))
    await indexer_router.app.memory.set("total_files", str(len(file_summaries)))

    return IndexResult(
        files_indexed=len(file_summaries),
        project_root=project_path,
        indexed_at=indexed_at,
        message=f"Successfully indexed {len(file_summaries)} files. Ready to answer questions.",
    ).model_dump()


@indexer_router.reasoner()
async def update_index(project_path: str) -> dict:
    """
    Incremental re-index — only processes files changed since last index.

    This is what you call after editing code. Instead of re-summarizing
    all 200 files, we only touch the ones whose mtime changed.
    Fast enough to call on every save if needed.
    """
    # Retrieve when we last indexed from memory
    indexed_at_str = await indexer_router.app.memory.get("indexed_at")
    if not indexed_at_str:
        return {"error": "No index found. Run index_project first.", "files_updated": 0}

    since = float(indexed_at_str)
    changed = get_changed_files(project_path, since_timestamp=since)

    if not changed:
        return {"message": "No files changed since last index.", "files_updated": 0}

    # Load existing maps from memory so we patch them, not overwrite
    file_summaries = json.loads(await indexer_router.app.memory.get("file_summaries") or "{}")
    keyword_map = json.loads(await indexer_router.app.memory.get("keyword_map") or "{}")
    symbol_map = json.loads(await indexer_router.app.memory.get("symbol_map") or "{}")

    project_root = await indexer_router.app.memory.get("project_root") or project_path

    for file_meta in changed:
        path = file_meta["path"]
        try:
            rel_path = str(path).replace(project_root + "/", "")
        except Exception:
            rel_path = path

        file_data = read_file(path)
        content = file_data.get("content", "")
        if not content.strip():
            continue

        snippet = content[:3000]
        summary_result = await indexer_router.ai(
            system="You are a senior software engineer analyzing a codebase. Summarize source files concisely.",
            user=f"File: {rel_path}\n\n```\n{snippet}\n```\n\nSummarize what this file does, its purpose, and key dependencies.",
            schema=FileSummary,
        )

        symbols = extract_symbols(content, path)
        keywords = extract_keywords(content)

        for kw in keywords:
            keyword_map.setdefault(kw, [])
            if rel_path not in keyword_map[kw]:
                keyword_map[kw].append(rel_path)

        for sym in symbols:
            symbol_map[sym["name"]] = {"file": rel_path, "line": sym["line"], "type": sym["type"]}

        file_summaries[rel_path] = {
            "summary": summary_result.summary,
            "purpose": summary_result.purpose,
            "dependencies": summary_result.dependencies,
            "keywords": keywords,
            "last_modified": file_meta["last_modified"],
        }

        indexer_router.app.note(f"Updated: {rel_path}", tags=["update", "indexing"])

    # Write patched maps back to memory
    new_timestamp = time.time()
    await indexer_router.app.memory.set("file_summaries", json.dumps(file_summaries))
    await indexer_router.app.memory.set("keyword_map", json.dumps(keyword_map))
    await indexer_router.app.memory.set("symbol_map", json.dumps(symbol_map))
    await indexer_router.app.memory.set("indexed_at", str(new_timestamp))

    return {
        "files_updated": len(changed),
        "updated_files": [f["relative_path"] for f in changed],
        "message": f"Re-indexed {len(changed)} changed file(s).",
    }
