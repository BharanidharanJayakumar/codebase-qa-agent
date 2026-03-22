"""
Navigator agent — LLM-directed, question-driven code exploration.

Instead of pre-computing everything at index time, the navigator:
1. Builds a compact "project map" (summaries + file listing + dependency graph)
2. LLM reads the map and decides which files to examine for the question
3. Reads only those files from disk — targeted, efficient context building
4. Optionally drills deeper if the answer LLM requests more files

This is the "Agentic RAG" pattern: the LLM decides what to retrieve.
"""

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field
from agentfield import AgentRouter

from skills.extractor import extract_keywords
from skills.scanner import read_file
from skills.storage import (
    load_module_summaries,
    load_semantic_summary,
    load_project_summary,
    load_file_imports,
)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 24_000
MAX_FILES_IN_MAP = 150


# ── Schemas ──────────────────────────────────────────────────────────────────

class NavigationDecision(BaseModel):
    """LLM output: what to look at for this question."""
    needs_code: bool = Field(
        description="False if summaries/metadata are sufficient (overview, theme, architecture, aggregate questions). True if actual source code is needed."
    )
    target_files: list[str] = Field(
        description="Specific file paths to read (max 8). Use exact paths from the file map."
    )
    target_folders: list[str] = Field(
        description="Folders to explore if unsure which specific files (max 3). System will pick representative files."
    )
    reasoning: str = Field(
        description="Brief explanation of why these files/folders were chosen"
    )
    search_terms: list[str] = Field(
        description="Backup keywords for search-based retrieval if file navigation misses"
    )


class CodeCitation(BaseModel):
    """A reference to a specific location in the codebase."""
    file: str = Field(description="Relative file path")
    start_line: int = Field(description="Starting line number")
    end_line: int | None = Field(default=None, description="Ending line number (if range)")
    symbol: str = Field(default="", description="Function/class name if applicable")
    snippet: str = Field(default="", description="Brief code snippet or what this citation supports")


class AnswerWithDrilldown(BaseModel):
    """LLM output: answer with code citations and optional request for more files."""
    answer: str = Field(description="Clear, direct answer with inline references like [1], [2], etc.")
    citations: list[CodeCitation] = Field(
        default_factory=list,
        description="Numbered list of code references backing up claims in the answer. [1] maps to citations[0], etc."
    )
    relevant_files: list[str] = Field(description="Files most relevant to this answer")
    confidence: str = Field(description="high, medium, or low")
    follow_up: list[str] = Field(description="1-2 follow-up questions the user might want to ask")
    needs_more_context: bool = Field(
        default=False,
        description="Set True ONLY if you can see specific files in the project map that would significantly improve the answer but weren't provided"
    )
    additional_files: list[str] = Field(
        default_factory=list,
        description="If needs_more_context is True, list specific file paths from the project map to read next (max 5)"
    )


# ── Navigator System Prompt ──────────────────────────────────────────────────

NAVIGATOR_SYSTEM = """You are a code navigator for a codebase Q&A system. Given a user's question and a complete map of the project (file listing with symbols, module summaries, dependency graph), decide what information is needed to answer.

RULES:
- For overview/theme/purpose/architecture questions: set needs_code=False (the summaries and metadata are sufficient)
- For "how does X work?", "explain Y", "what does Z do?": set needs_code=True, pick the 3-8 most relevant files
- For aggregate questions (how many routes, list all DTOs, count controllers): set needs_code=False (categories data will be added automatically)
- For "what about X?" follow-up questions: set needs_code=True if the question asks about specific code

FILE SELECTION STRATEGY:
- Pick files based on: file names, symbol names, module summaries, AND the dependency graph
- FOLLOW DEPENDENCY CHAINS: if a controller depends on a service, include both
- Prefer implementation files over test files (unless the question is about tests)
- Prefer files with many symbols (they contain core logic)
- If unsure which specific files, specify target_folders instead
- Include search_terms as backup keywords
- Be precise — every unnecessary file wastes tokens"""


ANSWER_SYSTEM = """You are an expert software engineer helping a developer understand a codebase.

Answer questions using ONLY the provided context (source code, summaries, metadata).
Be specific and accurate — mention file names, function names, and how things connect.

CITATION RULES (IMPORTANT):
- Back up every technical claim with a numbered citation: [1], [2], etc.
- Each citation references a specific file and line range from the provided context
- The citation number maps to the citations array index (1-indexed: [1] = citations[0])
- Include the file path, start_line, end_line, and the symbol name if applicable
- Include a brief snippet or description of what the citation supports
- Every function, class, or important pattern you mention MUST have a citation
- Use the line numbers from the === FILE: path [lines X-Y] === headers in the context

Example citation format in your answer: "The authentication is handled by the `verify_token` function [1] which calls the JWT library [2]."

If the provided context is insufficient to fully answer the question:
- Set needs_more_context=True
- List specific file paths (from the project map shown to you earlier) in additional_files
- Only do this if you're confident specific files would help — don't guess
- Maximum 1 drill-down round, so choose wisely

If the context IS sufficient, set needs_more_context=False and additional_files=[]."""


# ── Build Project Map ────────────────────────────────────────────────────────

def build_project_map(
    project_root: str,
    file_index: dict,
    symbol_map: dict,
) -> str:
    """
    Build a compact "map" of the project for the navigator LLM.

    Includes:
    - Semantic summary (purpose, architecture, data flow)
    - Module summaries (per-folder purpose, patterns, key abstractions)
    - Import/dependency graph (compact: file → [imports])
    - File listing with symbols: `path: [func1, Class1, ...]`

    For 100 files, this is ~4000-5000 tokens.
    """
    parts = []

    # 1. Semantic summary
    sem = load_semantic_summary(project_root)
    if sem:
        parts.append("=== PROJECT UNDERSTANDING ===")
        for key in ["purpose", "architecture", "domain", "data_flow", "tech_decisions"]:
            if sem.get(key):
                parts.append(f"{key.replace('_', ' ').title()}: {sem[key]}")

    # 2. Project metadata
    meta = load_project_summary(project_root)
    if meta:
        parts.append("\n=== METADATA ===")
        if meta.get("languages"):
            langs = meta["languages"]
            if isinstance(langs, str):
                try:
                    langs = json.loads(langs)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(langs, dict):
                parts.append(f"Languages: {', '.join(f'{k}: {v} files' for k, v in langs.items())}")
        if meta.get("framework_hints"):
            hints = meta["framework_hints"]
            if isinstance(hints, str):
                try:
                    hints = json.loads(hints)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(hints, list) and hints:
                parts.append(f"Frameworks: {', '.join(hints)}")
        if meta.get("readme_content"):
            readme = str(meta["readme_content"])[:800]
            parts.append(f"\nREADME (excerpt):\n{readme}")

    # 3. Module summaries
    modules = load_module_summaries(project_root)
    if modules:
        parts.append("\n=== MODULE SUMMARIES ===")
        for mod in modules[:15]:
            parts.append(f"\n{mod['module_path']}/: {mod['summary']}")
            if mod.get("key_patterns"):
                patterns = mod["key_patterns"]
                if isinstance(patterns, str):
                    try:
                        patterns = json.loads(patterns)
                    except (json.JSONDecodeError, TypeError):
                        patterns = [patterns]
                parts.append(f"  Patterns: {', '.join(patterns[:5])}")
            if mod.get("key_abstractions"):
                abstractions = mod["key_abstractions"]
                if isinstance(abstractions, str):
                    try:
                        abstractions = json.loads(abstractions)
                    except (json.JSONDecodeError, TypeError):
                        abstractions = [abstractions]
                parts.append(f"  Key: {', '.join(abstractions[:8])}")

    # 4. Dependency graph (compact)
    imports = load_file_imports(project_root)
    if imports:
        parts.append("\n=== DEPENDENCY GRAPH ===")
        by_source: dict[str, list[str]] = {}
        for imp in imports:
            source = imp["source_path"] if isinstance(imp, dict) else imp[0]
            target = imp.get("target_path", "") if isinstance(imp, dict) else (imp[2] if len(imp) > 2 else "")
            if target:
                by_source.setdefault(source, []).append(target)
        for source, targets in sorted(by_source.items())[:50]:
            unique_targets = list(dict.fromkeys(targets))[:6]
            parts.append(f"  {source} → {', '.join(unique_targets)}")

    # 5. File listing with symbols
    parts.append("\n=== FILE MAP ===")

    # Sort files by symbol count (most logic-dense first), cap at MAX_FILES_IN_MAP
    file_entries = []
    for rel_path, meta_info in file_index.items():
        symbols = meta_info.get("symbols", [])
        file_entries.append((rel_path, symbols))
    file_entries.sort(key=lambda x: len(x[1]), reverse=True)

    for rel_path, symbols in file_entries[:MAX_FILES_IN_MAP]:
        if symbols:
            sym_str = ", ".join(symbols[:12])
            if len(symbols) > 12:
                sym_str += f" (+{len(symbols) - 12} more)"
            parts.append(f"  {rel_path}: [{sym_str}]")
        else:
            parts.append(f"  {rel_path}")

    if len(file_entries) > MAX_FILES_IN_MAP:
        parts.append(f"  ... and {len(file_entries) - MAX_FILES_IN_MAP} more files (see module summaries)")

    return "\n".join(parts)


# ── Navigate ─────────────────────────────────────────────────────────────────

async def navigate(
    question: str,
    project_root: str,
    file_index: dict,
    symbol_map: dict,
    router: AgentRouter,
) -> NavigationDecision:
    """
    LLM-driven navigation: reads the project map and decides what to look at.

    Returns a NavigationDecision with target files/folders to read.
    Falls back to regex-based navigation on LLM failure.
    """
    project_map = build_project_map(project_root, file_index, symbol_map)

    try:
        result = await router.ai(
            system=NAVIGATOR_SYSTEM,
            user=f"PROJECT MAP:\n{project_map}\n\n---\n\nQUESTION: {question}",
            schema=NavigationDecision,
            max_tokens=300,
        )

        # Validate target_files exist in file_index
        valid_files = [f for f in result.target_files if f in file_index]
        result.target_files = valid_files[:8]
        result.target_folders = result.target_folders[:3]

        logger.info(
            f"Navigator: needs_code={result.needs_code}, "
            f"files={len(result.target_files)}, folders={len(result.target_folders)}, "
            f"reasoning={result.reasoning[:80]}"
        )
        return result

    except Exception as e:
        logger.warning(f"Navigator LLM failed: {e}, using fallback")
        return navigate_fallback(question, file_index)


def navigate_fallback(question: str, file_index: dict | None = None) -> NavigationDecision:
    """Regex-based fallback when LLM navigation fails."""
    q_lower = question.lower()

    # Overview/theme questions
    overview_words = [
        "overview", "summary", "summarize", "what is this", "what does this",
        "theme", "purpose", "about this", "describe this", "this project",
        "this codebase", "this repo", "tech stack", "what language", "what framework",
    ]
    if any(w in q_lower for w in overview_words):
        return NavigationDecision(
            needs_code=False,
            target_files=[],
            target_folders=[],
            reasoning="Overview question — summaries sufficient",
            search_terms=[],
        )

    # Aggregate questions
    aggregate_words = ["how many", "count", "list all", "list every", "show all", "show every"]
    if any(w in q_lower for w in aggregate_words):
        return NavigationDecision(
            needs_code=False,
            target_files=[],
            target_folders=[],
            reasoning="Aggregate question — categories data sufficient",
            search_terms=[],
        )

    # Code-specific: use keyword extraction to find relevant files
    terms = extract_keywords(question, top_n=5)
    target_files = []
    if file_index:
        # Simple keyword matching against file paths and symbols
        for rel_path, meta in file_index.items():
            path_lower = rel_path.lower()
            symbols_lower = [s.lower() for s in meta.get("symbols", [])]
            for term in terms:
                t = term.lower()
                if t in path_lower or any(t in s for s in symbols_lower):
                    target_files.append(rel_path)
                    break
            if len(target_files) >= 6:
                break

    return NavigationDecision(
        needs_code=True,
        target_files=target_files,
        target_folders=[],
        reasoning=f"Code question — keyword matched {len(target_files)} files",
        search_terms=terms,
    )


# ── Read Targeted Files ──────────────────────────────────────────────────────

def read_targeted_files(
    decision: NavigationDecision,
    file_index: dict,
    project_root: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Read the files specified by the navigator from disk.

    For target_files: reads full content via scanner.read_file()
    For target_folders: picks top 3 files by symbol count from each folder
    Falls back to chunks from the index if disk read fails.
    """
    parts = []
    chars_used = 0

    # Resolve files from target_folders
    folder_files = []
    for folder in decision.target_folders:
        folder_lower = folder.rstrip("/").lower()
        candidates = []
        for rel_path, meta in file_index.items():
            if rel_path.lower().startswith(folder_lower + "/") or rel_path.lower().startswith(folder_lower):
                candidates.append((rel_path, len(meta.get("symbols", []))))
        # Pick top 3 by symbol count
        candidates.sort(key=lambda x: x[1], reverse=True)
        folder_files.extend([c[0] for c in candidates[:3]])

    # Combine and deduplicate
    all_files = list(dict.fromkeys(decision.target_files + folder_files))

    for rel_path in all_files:
        if chars_used >= max_chars:
            break

        # Try reading from disk first
        abs_path = os.path.join(project_root, rel_path)
        content = ""

        result = read_file(abs_path)
        if result.get("content"):
            content = result["content"]
        else:
            # Fallback: reconstruct from chunks in the index
            meta = file_index.get(rel_path, {})
            chunks = meta.get("chunks", [])
            if chunks:
                content = "\n".join(c.get("content", "") for c in chunks)

        if not content:
            continue

        # Cap per-file content to leave room for other files
        remaining = max_chars - chars_used
        per_file_cap = min(remaining, 8000)
        if len(content) > per_file_cap:
            content = content[:per_file_cap] + "\n... (truncated)"

        # Add line numbers to content for citation support
        numbered_lines = []
        for i, line in enumerate(content.split("\n"), 1):
            numbered_lines.append(f"{i:4d} | {line}")
        numbered_content = "\n".join(numbered_lines)

        total_lines = content.count("\n") + 1
        header = f"=== FILE: {rel_path} [lines 1-{total_lines}] ==="
        parts.append(f"{header}\n{numbered_content}")
        chars_used += len(header) + len(numbered_content) + 2

    if not parts:
        return ""

    return "\n\n".join(parts)


def read_files_by_paths(
    file_paths: list[str],
    file_index: dict,
    project_root: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Read specific files by path — used for drill-down rounds."""
    decision = NavigationDecision(
        needs_code=True,
        target_files=file_paths[:5],
        target_folders=[],
        reasoning="Drill-down request from answer LLM",
        search_terms=[],
    )
    return read_targeted_files(decision, file_index, project_root, max_chars)


# ── Build Summary Context ────────────────────────────────────────────────────

def build_summary_context(project_root: str) -> str:
    """
    Build context from pre-computed summaries (no file reading needed).
    Used when navigator says needs_code=False.
    """
    parts = []

    # Semantic summary
    sem = load_semantic_summary(project_root)
    if sem:
        section = ["=== PROJECT UNDERSTANDING ==="]
        for key in ["purpose", "architecture", "domain", "data_flow", "tech_decisions"]:
            if sem.get(key):
                section.append(f"{key.replace('_', ' ').title()}: {sem[key]}")
        if sem.get("key_patterns"):
            try:
                patterns = json.loads(sem["key_patterns"]) if isinstance(sem["key_patterns"], str) else sem["key_patterns"]
                section.append(f"Key Patterns: {', '.join(patterns)}")
            except (json.JSONDecodeError, TypeError):
                section.append(f"Key Patterns: {sem['key_patterns']}")
        parts.append("\n".join(section))

    # Module summaries
    modules = load_module_summaries(project_root)
    if modules:
        section = ["=== MODULE SUMMARIES ==="]
        for mod in modules[:10]:
            section.append(f"\n{mod['module_path']}/: {mod['summary']}")
            if mod.get("key_patterns"):
                kp = mod["key_patterns"]
                if isinstance(kp, str):
                    try:
                        kp = json.loads(kp)
                    except (json.JSONDecodeError, TypeError):
                        kp = [kp]
                section.append(f"  Patterns: {', '.join(kp[:5])}")
            if mod.get("domain_concepts"):
                dc = mod["domain_concepts"]
                if isinstance(dc, str):
                    try:
                        dc = json.loads(dc)
                    except (json.JSONDecodeError, TypeError):
                        dc = [dc]
                section.append(f"  Domain: {', '.join(dc[:5])}")
            if mod.get("key_abstractions"):
                ka = mod["key_abstractions"]
                if isinstance(ka, str):
                    try:
                        ka = json.loads(ka)
                    except (json.JSONDecodeError, TypeError):
                        ka = [ka]
                section.append(f"  Key: {', '.join(ka[:5])}")
        parts.append("\n".join(section))

    # Metadata
    meta = load_project_summary(project_root)
    if meta:
        section = ["=== PROJECT METADATA ==="]
        if meta.get("languages"):
            langs = meta["languages"]
            if isinstance(langs, str):
                try:
                    langs = json.loads(langs)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(langs, dict):
                section.append(f"Languages: {', '.join(f'{k}: {v} files' for k, v in langs.items())}")
        if meta.get("framework_hints"):
            hints = meta["framework_hints"]
            if isinstance(hints, str):
                try:
                    hints = json.loads(hints)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(hints, list) and hints:
                section.append(f"Frameworks: {', '.join(hints)}")
        if meta.get("total_lines"):
            section.append(f"Total lines: {meta['total_lines']}")
        if meta.get("readme_content"):
            section.append(f"\nREADME:\n{str(meta['readme_content'])[:1500]}")
        parts.append("\n".join(section))

    # Import graph
    imports = load_file_imports(project_root)
    if imports:
        section = ["=== DEPENDENCY GRAPH ==="]
        by_source: dict[str, list[str]] = {}
        for imp in imports[:100]:
            source = imp["source_path"] if isinstance(imp, dict) else imp[0]
            target = imp.get("imported_name", "") if isinstance(imp, dict) else imp[1]
            by_source.setdefault(source, []).append(target)
        for source, targets in list(by_source.items())[:20]:
            section.append(f"  {source} imports: {', '.join(targets[:8])}")
        parts.append("\n".join(section))

    return "\n\n".join(parts)
