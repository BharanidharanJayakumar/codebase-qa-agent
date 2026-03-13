"""
Hierarchical LLM-powered code summarization.

At index time, reads representative code files and generates:
1. Module-level summaries (per top-level directory)
2. A synthesized project-level semantic summary

Uses structured output (Pydantic) and rate-limited LLM calls via AgentField.
Gracefully degrades if LLM is unavailable.
"""
import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Entry-point filenames that are especially representative
ENTRY_POINTS = {
    "main.py", "app.py", "server.py", "index.py", "manage.py", "__main__.py",
    "index.ts", "index.js", "app.ts", "app.js", "server.ts", "server.js",
    "main.go", "main.rs", "Main.java", "Program.cs", "Startup.cs",
    "index.tsx", "index.jsx", "app.tsx", "app.jsx",
}

# Max chars per representative file sent to LLM
MAX_FILE_CHARS = 2000
# Max representative files per module
MAX_FILES_PER_MODULE = 3
# Delay between LLM calls (rate limiting for Groq free tier)
CALL_DELAY_SECONDS = 3
# Max retries on rate limit
MAX_RETRIES = 3


class ModuleSummarySchema(BaseModel):
    purpose: str = Field(description="What this module/directory does in 1-2 sentences")
    key_patterns: list[str] = Field(description="Design patterns used (e.g. 'repository pattern', 'middleware chain')")
    domain_concepts: list[str] = Field(description="Domain concepts this module handles (e.g. 'user authentication', 'payment processing')")
    key_abstractions: list[str] = Field(description="Most important classes/functions and what they do, as brief strings")


class ProjectSemanticSchema(BaseModel):
    purpose: str = Field(description="What this project does, in 2-3 sentences")
    architecture: str = Field(description="How the project is structured and why")
    key_patterns: list[str] = Field(description="Major design patterns across the project")
    domain: str = Field(description="The problem domain this project addresses")
    data_flow: str = Field(description="How data flows through the system")
    tech_decisions: str = Field(description="Notable technology choices and their rationale")


def _strip_comments_and_blanks(content: str) -> str:
    """Remove blank lines and single-line comments to save tokens."""
    lines = content.split("\n")
    stripped = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Skip common single-line comment patterns
        if s.startswith("//") or s.startswith("#") or s.startswith("*") or s.startswith("/*"):
            continue
        stripped.append(line)
    return "\n".join(stripped)


def _group_files_by_module(file_index: dict) -> dict[str, list[str]]:
    """Group files by their top-level directory (module)."""
    modules: dict[str, list[str]] = defaultdict(list)
    for rel_path in file_index:
        parts = Path(rel_path).parts
        if len(parts) == 1:
            modules["."].append(rel_path)  # root-level files
        else:
            modules[parts[0]].append(rel_path)
    return dict(modules)


def _score_file(rel_path: str, file_meta: dict, import_counts: dict[str, int]) -> float:
    """Score a file for representativeness."""
    score = 0.0
    filename = Path(rel_path).name

    # Symbol count — more symbols = more logic
    score += len(file_meta.get("symbols", [])) * 2

    # Entry point bonus
    if filename in ENTRY_POINTS:
        score += 20

    # Import hub bonus (many files import from this one)
    score += import_counts.get(rel_path, 0) * 3

    # README bonus
    if filename.lower().startswith("readme"):
        score += 15

    # Larger files tend to be more important (but cap it)
    size_kb = file_meta.get("size_bytes", 0) / 1024
    score += min(size_kb, 10)

    return score


def _select_representative_files(
    module_files: list[str],
    file_index: dict,
    import_counts: dict[str, int],
) -> list[str]:
    """Select up to MAX_FILES_PER_MODULE representative files from a module."""
    scored = []
    for rel_path in module_files:
        meta = file_index.get(rel_path, {})
        score = _score_file(rel_path, meta, import_counts)
        scored.append((rel_path, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [path for path, _ in scored[:MAX_FILES_PER_MODULE]]


def _build_import_counts(imports_data: list) -> dict[str, int]:
    """Count how many files import from each file (hub detection)."""
    counts: dict[str, int] = defaultdict(int)
    for imp in imports_data:
        target = imp[2] if isinstance(imp, (list, tuple)) else imp.get("target_path")
        if target:
            counts[target] += 1
    return dict(counts)


def _get_file_content(rel_path: str, project_root: str, file_index: dict) -> str:
    """Get file content from chunks (already in index) or read from disk."""
    meta = file_index.get(rel_path, {})
    chunks = meta.get("chunks", [])
    if chunks:
        content = "\n".join(c.get("content", "") for c in chunks)
    else:
        # Fallback: read from disk
        full_path = Path(project_root) / rel_path
        try:
            content = full_path.read_text(errors="replace")[:50000]
        except Exception:
            return ""

    stripped = _strip_comments_and_blanks(content)
    return stripped[:MAX_FILE_CHARS]


def _build_module_prompt(
    module_path: str,
    module_files: list[str],
    representative_files: list[str],
    file_index: dict,
    project_root: str,
    module_imports: list[str],
) -> str:
    """Build the user prompt for a module summary LLM call."""
    parts = []

    # File listing with symbols
    parts.append(f"Module: {module_path}/")
    parts.append(f"Files ({len(module_files)}):")
    for rel_path in module_files[:30]:  # cap listing
        meta = file_index.get(rel_path, {})
        symbols = meta.get("symbols", [])
        sym_str = ", ".join(symbols[:8])
        if len(symbols) > 8:
            sym_str += f" (+{len(symbols) - 8} more)"
        parts.append(f"  {rel_path}: [{sym_str}]")

    # Import relationships
    if module_imports:
        parts.append(f"\nImports from other modules: {', '.join(module_imports[:15])}")

    # Representative file contents
    parts.append("\n--- Representative source code ---")
    for rel_path in representative_files:
        content = _get_file_content(rel_path, project_root, file_index)
        if content:
            parts.append(f"\n=== {rel_path} ===")
            parts.append(content)

    return "\n".join(parts)


def _get_module_imports(module_path: str, imports_data: list, all_modules: set[str]) -> list[str]:
    """Find which other modules this module imports from."""
    external_imports = set()
    for imp in imports_data:
        source = imp[0] if isinstance(imp, (list, tuple)) else imp.get("source_path", "")
        target = imp[2] if isinstance(imp, (list, tuple)) else imp.get("target_path")

        # Check if source is in this module
        source_parts = Path(source).parts
        source_module = source_parts[0] if len(source_parts) > 1 else "."
        if source_module != module_path:
            continue

        # Check if target is in a different module
        if target:
            target_parts = Path(target).parts
            target_module = target_parts[0] if len(target_parts) > 1 else "."
            if target_module != module_path and target_module in all_modules:
                external_imports.add(target_module)

    return list(external_imports)


async def _llm_call_with_retry(router, system: str, user: str, schema, max_tokens: int = 300):
    """Make an LLM call with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            result = await router.ai(
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                wait = (2 ** attempt) * 5  # 5s, 10s, 20s
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            else:
                logger.error(f"LLM call failed: {e}")
                raise
    raise RuntimeError("Max retries exceeded for LLM call")


async def generate_hierarchical_summary(
    file_index: dict,
    symbol_map: dict,
    project_root: str,
    imports_data: list,
    router,
) -> tuple[list[dict], dict]:
    """
    Generate hierarchical LLM summaries for a project.

    Returns:
        (module_summaries, project_semantic) — both ready for save_index().
        module_summaries: list of dicts with module_path, summary, key_patterns, etc.
        project_semantic: dict with purpose, architecture, patterns, domain, data_flow, tech_decisions.
    """
    if not file_index:
        return [], {}

    # Step 1: Group files by module and build import counts
    modules = _group_files_by_module(file_index)
    import_counts = _build_import_counts(imports_data)
    all_module_names = set(modules.keys())

    logger.info(f"Summarizing {len(modules)} modules for {project_root}")

    # Step 2: Generate module-level summaries
    module_summaries = []

    for module_path, module_files in modules.items():
        # Select representative files
        rep_files = _select_representative_files(module_files, file_index, import_counts)
        if not rep_files:
            continue

        # Build prompt
        module_imports = _get_module_imports(module_path, imports_data, all_module_names)
        prompt = _build_module_prompt(
            module_path, module_files, rep_files, file_index, project_root, module_imports
        )

        try:
            result = await _llm_call_with_retry(
                router,
                system=(
                    "You are a senior software architect analyzing a code module. "
                    "Based on the file listing and representative source code, summarize "
                    "this module's purpose, design patterns, domain concepts, and key abstractions. "
                    "Be concise and specific."
                ),
                user=prompt,
                schema=ModuleSummarySchema,
                max_tokens=300,
            )

            module_summaries.append({
                "module_path": module_path,
                "summary": result.purpose,
                "key_patterns": result.key_patterns,
                "domain_concepts": result.domain_concepts,
                "key_abstractions": result.key_abstractions,
                "generated_at": time.time(),
            })

            logger.info(f"  Summarized module: {module_path}")

        except Exception as e:
            logger.warning(f"  Failed to summarize module {module_path}: {e}")
            # Continue with other modules
            continue

        # Rate limiting
        await asyncio.sleep(CALL_DELAY_SECONDS)

    if not module_summaries:
        logger.warning("No module summaries generated, skipping project synthesis")
        return [], {}

    # Step 3: Synthesize project-level semantic summary
    synthesis_parts = []

    # Module summaries
    synthesis_parts.append("Module summaries:")
    for mod in module_summaries:
        synthesis_parts.append(f"\n{mod['module_path']}/: {mod['summary']}")
        if mod["key_patterns"]:
            synthesis_parts.append(f"  Patterns: {', '.join(mod['key_patterns'])}")
        if mod["domain_concepts"]:
            synthesis_parts.append(f"  Domain: {', '.join(mod['domain_concepts'])}")
        if mod["key_abstractions"]:
            synthesis_parts.append(f"  Key: {', '.join(mod['key_abstractions'][:5])}")

    # README (from disk)
    readme_content = ""
    for name in ["README.md", "README", "README.rst", "readme.md"]:
        readme_path = Path(project_root) / name
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(errors="replace")[:1500]
                break
            except Exception:
                pass

    if readme_content:
        synthesis_parts.append(f"\nREADME:\n{readme_content}")

    # Metadata
    extensions = defaultdict(int)
    total_symbols = 0
    for meta in file_index.values():
        extensions[meta.get("extension", "")] += 1
        total_symbols += len(meta.get("symbols", []))

    lang_str = ", ".join(f"{ext}: {count}" for ext, count in
                         sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:8])
    synthesis_parts.append(f"\nLanguages: {lang_str}")
    synthesis_parts.append(f"Total files: {len(file_index)}, Total symbols: {total_symbols}")

    # Cross-module import graph
    module_deps = defaultdict(set)
    for mod_sum in module_summaries:
        deps = _get_module_imports(mod_sum["module_path"], imports_data, all_module_names)
        if deps:
            module_deps[mod_sum["module_path"]] = set(deps)

    if module_deps:
        dep_str = "; ".join(f"{k} -> {', '.join(v)}" for k, v in module_deps.items())
        synthesis_parts.append(f"Module dependencies: {dep_str}")

    try:
        project_result = await _llm_call_with_retry(
            router,
            system=(
                "You are a senior software architect providing a comprehensive project overview. "
                "Based on the module summaries, README, and metadata, synthesize a complete "
                "understanding of this project. Cover its purpose, architecture, design patterns, "
                "problem domain, data flow, and technology decisions. Be specific and insightful."
            ),
            user="\n".join(synthesis_parts),
            schema=ProjectSemanticSchema,
            max_tokens=500,
        )

        project_semantic = {
            "purpose": project_result.purpose,
            "architecture": project_result.architecture,
            "key_patterns": json.dumps(project_result.key_patterns),
            "domain": project_result.domain,
            "data_flow": project_result.data_flow,
            "tech_decisions": project_result.tech_decisions,
        }

        logger.info(f"  Project synthesis complete: {project_result.purpose[:80]}...")

    except Exception as e:
        logger.warning(f"Failed to synthesize project summary: {e}")
        project_semantic = {}

    return module_summaries, project_semantic
