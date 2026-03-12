"""
Project-level intelligence: summaries, import graphs, and symbol categorization.
All pure Python — no LLM calls. Runs at index time.
"""

import json
import re
from collections import Counter
from pathlib import Path


# ── Framework detection patterns ──────────────────────────────────────────────

DEPENDENCY_FILES = [
    "package.json", "requirements.txt", "Pipfile", "pyproject.toml",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Gemfile",
    "composer.json", "mix.exs", "pubspec.yaml",
]

FRAMEWORK_PATTERNS = {
    "package.json": {
        "next": "Next.js", "react": "React", "vue": "Vue.js", "angular": "Angular",
        "express": "Express", "fastify": "Fastify", "nestjs": "NestJS", "nuxt": "Nuxt",
        "svelte": "Svelte", "remix": "Remix", "gatsby": "Gatsby",
        "tailwindcss": "Tailwind CSS", "prisma": "Prisma", "drizzle": "Drizzle",
    },
    "requirements.txt": {
        "fastapi": "FastAPI", "django": "Django", "flask": "Flask",
        "sqlalchemy": "SQLAlchemy", "celery": "Celery", "pydantic": "Pydantic",
        "pytest": "Pytest", "pandas": "Pandas", "numpy": "NumPy",
        "tensorflow": "TensorFlow", "torch": "PyTorch",
    },
    "Cargo.toml": {
        "actix": "Actix", "rocket": "Rocket", "tokio": "Tokio", "serde": "Serde",
    },
    "go.mod": {
        "gin-gonic": "Gin", "echo": "Echo", "fiber": "Fiber", "chi": "Chi",
    },
}

# ── Import extraction patterns ────────────────────────────────────────────────

IMPORT_PATTERNS = {
    ".py": [
        re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),
        re.compile(r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE),
    ],
    ".js": [
        re.compile(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
    ],
    ".ts": [
        re.compile(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
    ],
    ".go": [
        re.compile(r"""["']([^"']+)["']"""),
    ],
    ".java": [
        re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE),
    ],
    ".cs": [
        re.compile(r"^\s*using\s+([\w.]+);", re.MULTILINE),
    ],
    ".rb": [
        re.compile(r"""^\s*require\s+['"]([^'"]+)['"]""", re.MULTILINE),
    ],
    ".php": [
        re.compile(r"^\s*use\s+([\w\\]+);", re.MULTILINE),
    ],
}
# Share patterns across extensions
for ext in (".jsx", ".tsx", ".mjs", ".cjs"):
    IMPORT_PATTERNS[ext] = IMPORT_PATTERNS[".js"]

# ── Symbol categorization patterns ────────────────────────────────────────────

DTO_NAME_SUFFIXES = ("dto", "request", "response", "entity", "schema", "model", "input", "output", "payload")
DTO_PATH_SEGMENTS = ("dto", "dtos", "models", "entities", "schemas")

ROUTE_PATH_SEGMENTS = ("routes", "routers", "controllers", "handlers", "endpoints", "api")
ROUTE_DECORATORS = re.compile(
    r"@(?:app|router|blueprint)\.(get|post|put|patch|delete|route)\s*\(",
    re.IGNORECASE,
)
ROUTE_ANNOTATIONS = re.compile(
    r"@(?:Get|Post|Put|Patch|Delete|Request)Mapping\s*\(",
)
ROUTE_EXPRESS = re.compile(
    r"(?:app|router)\.(get|post|put|patch|delete|use|all)\s*\(",
)

SERVICE_PATH_SEGMENTS = ("services", "service")
TEST_PATH_SEGMENTS = ("tests", "test", "__tests__", "spec", "specs")
CONFIG_PATH_SEGMENTS = ("config", "configs", "configuration")
MIDDLEWARE_PATH_SEGMENTS = ("middleware", "middlewares")


def build_project_summary(
    project_root: str, file_index: dict, symbol_map: dict
) -> dict:
    """Build project-level summary data. Returns dict of key→value for project_summary table."""
    root = Path(project_root)
    summary = {}

    # ── Languages (file count by extension) ──
    ext_counter = Counter()
    total_lines = 0
    for rel_path, meta in file_index.items():
        ext = meta.get("extension", "").lstrip(".")
        if ext:
            ext_counter[ext] += 1
        for chunk in meta.get("chunks", []):
            end = chunk.get("end_line", 0)
            if end > total_lines:
                total_lines = end

    summary["languages"] = dict(ext_counter.most_common(20))
    summary["total_lines"] = total_lines

    # ── Directory tree (top 2 levels with file counts) ──
    dir_tree = {}
    for rel_path in file_index:
        parts = Path(rel_path).parts
        if len(parts) >= 1:
            top = parts[0]
            dir_tree.setdefault(top, {"files": 0, "subdirs": {}})
            dir_tree[top]["files"] += 1
            if len(parts) >= 2:
                sub = parts[1]
                dir_tree[top]["subdirs"].setdefault(sub, 0)
                dir_tree[top]["subdirs"][sub] += 1
    summary["directory_tree"] = dir_tree

    # ── Total symbols by type ──
    type_counter = Counter()
    for locations in symbol_map.values():
        for loc in locations:
            sym_type = loc.get("type", "unknown")
            type_counter[sym_type] += 1
    summary["total_symbols"] = dict(type_counter.most_common(20))

    # ── README content ──
    readme_content = ""
    for name in ("README.md", "README", "readme.md", "Readme.md", "README.rst", "README.txt"):
        readme_path = root / name
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(errors="replace")[:4000]
            except Exception:
                pass
            break
    summary["readme_content"] = readme_content

    # ── Project description (first paragraph from README) ──
    description = ""
    if readme_content:
        lines = readme_content.split("\n")
        para = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started and stripped and not stripped.startswith("#") and not stripped.startswith("!"):
                started = True
            if started:
                if not stripped and para:
                    break
                if stripped:
                    para.append(stripped)
        description = " ".join(para)[:300]
    summary["project_description"] = description

    # ── Dependency files ──
    found_deps = []
    for dep_file in DEPENDENCY_FILES:
        if (root / dep_file).exists():
            found_deps.append(dep_file)
    summary["dependency_files"] = found_deps

    # ── Framework hints ──
    frameworks = set()
    for dep_file in found_deps:
        patterns = FRAMEWORK_PATTERNS.get(dep_file, {})
        if not patterns:
            continue
        try:
            content = (root / dep_file).read_text(errors="replace").lower()
            for keyword, name in patterns.items():
                if keyword in content:
                    frameworks.add(name)
        except Exception:
            pass
    summary["framework_hints"] = sorted(frameworks)

    return summary


def extract_imports(
    content: str, file_path: str, project_files: set[str]
) -> list[tuple[str, str, str | None]]:
    """Extract imports from a file. Returns list of (source_path, imported_name, target_path).
    target_path is resolved against project_files if possible, None if external."""
    ext = Path(file_path).suffix
    patterns = IMPORT_PATTERNS.get(ext, [])
    if not patterns:
        return []

    rel_path = file_path  # already relative
    results = []
    seen = set()

    for pattern in patterns:
        for match in pattern.finditer(content):
            imported = match.group(1)
            if imported in seen:
                continue
            seen.add(imported)

            # Try to resolve to a project file
            target = _resolve_import(imported, rel_path, ext, project_files)
            results.append((rel_path, imported, target))

    return results


def _resolve_import(imported: str, source_file: str, ext: str, project_files: set[str]) -> str | None:
    """Try to resolve an import to a project-relative file path."""
    if not imported.startswith(".") and ext not in (".py",):
        # Only resolve relative imports for JS/TS, all for Python
        if ext in (".py",):
            pass
        else:
            return None

    # For Python: convert dots to path
    if ext == ".py":
        candidate = imported.replace(".", "/")
        for suffix in (".py", "/__init__.py", ""):
            check = candidate + suffix
            if check in project_files:
                return check
        return None

    # For JS/TS: resolve relative paths
    if imported.startswith("."):
        source_dir = str(Path(source_file).parent)
        resolved = str(Path(source_dir) / imported)
        # Normalize
        resolved = str(Path(resolved))
        for suffix in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
            check = resolved + suffix
            if check in project_files:
                return check
        return None

    return None


def categorize_symbols(
    rel_path: str, content: str, symbols: list[dict]
) -> list[tuple[str, str, str, str]]:
    """Categorize symbols by pattern. Returns list of (rel_path, symbol_name, category, detail)."""
    results = []
    path_lower = rel_path.lower()
    path_parts = set(Path(rel_path).parts)
    path_parts_lower = {p.lower() for p in path_parts}

    for sym in symbols:
        name = sym.get("name", "")
        sym_type = sym.get("type", "")
        name_lower = name.lower()
        categories = []

        # ── DTOs / Models ──
        if any(seg in path_parts_lower for seg in DTO_PATH_SEGMENTS):
            categories.append(("dto", f"type={sym_type}"))
        elif name_lower.endswith(DTO_NAME_SUFFIXES):
            categories.append(("dto", f"type={sym_type}"))

        # ── Tests ──
        if any(seg in path_parts_lower for seg in TEST_PATH_SEGMENTS):
            categories.append(("test", f"type={sym_type}"))
        elif path_lower.startswith("test_") or ".test." in path_lower or ".spec." in path_lower:
            categories.append(("test", f"type={sym_type}"))

        # ── Services ──
        if any(seg in path_parts_lower for seg in SERVICE_PATH_SEGMENTS):
            categories.append(("service", f"type={sym_type}"))
        elif name_lower.endswith("service"):
            categories.append(("service", f"type={sym_type}"))

        # ── Config ──
        if any(seg in path_parts_lower for seg in CONFIG_PATH_SEGMENTS):
            categories.append(("config", f"type={sym_type}"))

        # ── Middleware ──
        if any(seg in path_parts_lower for seg in MIDDLEWARE_PATH_SEGMENTS):
            categories.append(("middleware", f"type={sym_type}"))

        for cat, detail in categories:
            results.append((rel_path, name, cat, detail))

    # ── Routes (content-based, not per-symbol) ──
    is_route_path = any(seg in path_parts_lower for seg in ROUTE_PATH_SEGMENTS)
    if is_route_path or ROUTE_DECORATORS.search(content) or ROUTE_ANNOTATIONS.search(content) or ROUTE_EXPRESS.search(content):
        # Find route-defining symbols
        route_syms = [s for s in symbols if s.get("type") in ("function", "method")]
        if route_syms:
            for sym in route_syms:
                results.append((rel_path, sym["name"], "route", f"line={sym.get('line', 0)}"))
        elif symbols:
            results.append((rel_path, symbols[0]["name"], "route", "file-level"))

    return results
