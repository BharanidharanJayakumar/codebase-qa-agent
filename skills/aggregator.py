"""
Project-level intelligence: summaries, import graphs, and symbol categorization.
All pure Python — no LLM calls. Runs at index time.
"""

import json
import os
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


# ── Extension to language name mapping ───────────────────────────────────────

EXT_TO_LANGUAGE = {
    "py": "Python", "js": "JavaScript", "ts": "TypeScript", "jsx": "React JSX",
    "tsx": "React TSX", "go": "Go", "java": "Java", "cs": "C#", "rs": "Rust",
    "rb": "Ruby", "php": "PHP", "c": "C", "cpp": "C++", "h": "C/C++ Header",
    "hpp": "C++ Header", "swift": "Swift", "kt": "Kotlin", "scala": "Scala",
    "ex": "Elixir", "exs": "Elixir Script", "dart": "Dart", "lua": "Lua",
    "sh": "Shell", "bash": "Bash", "zsh": "Zsh", "ps1": "PowerShell",
    "sql": "SQL", "graphql": "GraphQL", "gql": "GraphQL", "proto": "Protocol Buffers",
    "yaml": "YAML", "yml": "YAML", "toml": "TOML", "json": "JSON",
    "xml": "XML", "html": "HTML", "css": "CSS", "scss": "SCSS", "less": "LESS",
    "md": "Markdown", "rst": "reStructuredText", "vue": "Vue SFC", "svelte": "Svelte",
}


# ── Architecture pattern detection ───────────────────────────────────────────

ARCHITECTURE_INDICATORS = {
    "mvc": {
        "dirs": {"controllers", "models", "views"},
        "name": "MVC (Model-View-Controller)",
        "description": "Separates data (models), UI (views), and logic (controllers)",
    },
    "layered": {
        "dirs": {"services", "repositories", "controllers"},
        "name": "Layered Architecture",
        "description": "Organized in layers: presentation → business logic → data access",
    },
    "clean": {
        "dirs": {"domain", "usecases", "adapters", "entities"},
        "name": "Clean Architecture",
        "description": "Domain-centric with dependency inversion between layers",
    },
    "hexagonal": {
        "dirs": {"ports", "adapters", "domain"},
        "name": "Hexagonal (Ports & Adapters)",
        "description": "Core domain isolated from external integrations via ports",
    },
    "modular": {
        "dirs": {"modules", "features", "packages"},
        "name": "Modular / Feature-based",
        "description": "Code organized by feature/domain module rather than technical layer",
    },
    "microservices": {
        "files": {"docker-compose.yml", "docker-compose.yaml", "Dockerfile"},
        "dirs": {"gateway", "services", "proto"},
        "name": "Microservices / Multi-service",
        "description": "Multiple independently deployable services",
    },
    "monorepo": {
        "dirs": {"packages", "apps", "libs"},
        "files": {"lerna.json", "pnpm-workspace.yaml", "turbo.json", "nx.json"},
        "name": "Monorepo",
        "description": "Multiple projects/packages managed in a single repository",
    },
    "serverless": {
        "files": {"serverless.yml", "serverless.yaml", "template.yaml", "sam.yaml"},
        "dirs": {"functions", "lambdas"},
        "name": "Serverless",
        "description": "Cloud functions / event-driven serverless architecture",
    },
}


def detect_architecture_pattern(file_index: dict, project_root: str) -> dict:
    """Detect architecture patterns from directory structure and config files."""
    root = Path(project_root)
    all_dirs = set()
    for rel_path in file_index:
        parts = Path(rel_path).parts
        for p in parts[:-1]:
            all_dirs.add(p.lower())

    detected = []
    for pattern_key, indicators in ARCHITECTURE_INDICATORS.items():
        score = 0
        required_dirs = indicators.get("dirs", set())
        required_files = indicators.get("files", set())

        if required_dirs:
            matches = required_dirs & all_dirs
            if len(matches) >= 2 or (len(required_dirs) <= 2 and matches):
                score += len(matches)

        if required_files:
            for f in required_files:
                if (root / f).exists():
                    score += 2

        if score > 0:
            detected.append({
                "pattern": indicators["name"],
                "description": indicators["description"],
                "confidence": "high" if score >= 3 else "medium" if score >= 2 else "low",
                "score": score,
            })

    detected.sort(key=lambda x: x["score"], reverse=True)
    return {
        "primary": detected[0] if detected else {
            "pattern": "Standard",
            "description": "Conventional project structure",
            "confidence": "low",
        },
        "all_detected": detected,
    }


# ── Entry point discovery ────────────────────────────────────────────────────

ENTRY_POINT_PATTERNS = [
    ("main.py", "Python main entry point"),
    ("app.py", "Python application entry"),
    ("manage.py", "Django management script"),
    ("wsgi.py", "WSGI application entry"),
    ("asgi.py", "ASGI application entry"),
    ("__main__.py", "Python package entry point"),
    ("cli.py", "CLI entry point"),
    ("server.py", "Server entry point"),
    ("index.js", "JavaScript entry point"),
    ("index.ts", "TypeScript entry point"),
    ("server.js", "Node.js server entry"),
    ("server.ts", "Node.js server entry"),
    ("app.js", "Express/Node app entry"),
    ("app.ts", "Express/Node app entry"),
    ("main.go", "Go main entry point"),
    ("Application.java", "Spring Boot entry"),
    ("main.rs", "Rust main entry point"),
    ("lib.rs", "Rust library entry point"),
]


def find_entry_points(file_index: dict, project_root: str) -> list[dict]:
    """Discover application entry points."""
    entries = []
    file_paths = set(file_index.keys())

    for pattern, description in ENTRY_POINT_PATTERNS:
        for rel_path in file_paths:
            filename = Path(rel_path).name
            if filename == pattern or rel_path == pattern or rel_path.endswith("/" + pattern):
                meta = file_index.get(rel_path, {})
                entries.append({
                    "file": rel_path,
                    "description": description,
                    "symbols": meta.get("symbols", [])[:5],
                    "size_bytes": meta.get("size_bytes", 0),
                })

    # Check for __name__ == "__main__" pattern
    for rel_path, meta in file_index.items():
        if not rel_path.endswith(".py"):
            continue
        chunks = meta.get("chunks", [])
        content = "\n".join(c.get("content", "") for c in chunks)
        if '__name__' in content and '__main__' in content:
            if not any(e["file"] == rel_path for e in entries):
                entries.append({
                    "file": rel_path,
                    "description": "Python script with __main__ guard",
                    "symbols": meta.get("symbols", [])[:5],
                    "size_bytes": meta.get("size_bytes", 0),
                })

    return entries[:15]


# ── Code complexity metrics ──────────────────────────────────────────────────

def compute_complexity_metrics(file_index: dict) -> dict:
    """Compute code complexity and size metrics."""
    file_sizes = []
    file_lines = []
    symbols_per_file = []
    ext_lines = Counter()

    for rel_path, meta in file_index.items():
        size = meta.get("size_bytes", 0)
        file_sizes.append((rel_path, size))

        chunks = meta.get("chunks", [])
        max_line = 0
        for chunk in chunks:
            end = chunk.get("end_line", 0)
            if end > max_line:
                max_line = end
        file_lines.append((rel_path, max_line))

        sym_count = len(meta.get("symbols", []))
        symbols_per_file.append((rel_path, sym_count))

        ext = meta.get("extension", "").lstrip(".")
        ext_lines[ext] += max_line

    total_files = len(file_index)
    total_lines = sum(lines for _, lines in file_lines)
    total_size = sum(size for _, size in file_sizes)
    total_symbols = sum(count for _, count in symbols_per_file)

    file_sizes.sort(key=lambda x: x[1], reverse=True)
    file_lines.sort(key=lambda x: x[1], reverse=True)
    symbols_per_file.sort(key=lambda x: x[1], reverse=True)

    language_breakdown = []
    for ext, lines in ext_lines.most_common(10):
        lang_name = EXT_TO_LANGUAGE.get(ext, ext.upper())
        pct = round(lines / total_lines * 100, 1) if total_lines > 0 else 0
        language_breakdown.append({
            "language": lang_name,
            "extension": ext,
            "lines": lines,
            "percentage": pct,
        })

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "total_size_bytes": total_size,
        "total_size_human": _human_size(total_size),
        "total_symbols": total_symbols,
        "avg_file_size_lines": round(total_lines / total_files) if total_files > 0 else 0,
        "avg_symbols_per_file": round(total_symbols / total_files, 1) if total_files > 0 else 0,
        "largest_files": [
            {"file": f, "size_bytes": s, "size_human": _human_size(s)}
            for f, s in file_sizes[:5]
        ],
        "longest_files": [
            {"file": f, "lines": l} for f, l in file_lines[:5]
        ],
        "most_complex_files": [
            {"file": f, "symbol_count": c} for f, c in symbols_per_file[:5]
        ],
        "language_breakdown": language_breakdown,
    }


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ── Infrastructure detection ─────────────────────────────────────────────────

INFRA_FILES = {
    "Dockerfile": {"category": "containerization", "name": "Docker"},
    "docker-compose.yml": {"category": "containerization", "name": "Docker Compose"},
    "docker-compose.yaml": {"category": "containerization", "name": "Docker Compose"},
    ".dockerignore": {"category": "containerization", "name": "Docker"},
    ".github/workflows": {"category": "ci_cd", "name": "GitHub Actions"},
    ".gitlab-ci.yml": {"category": "ci_cd", "name": "GitLab CI"},
    "Jenkinsfile": {"category": "ci_cd", "name": "Jenkins"},
    ".circleci/config.yml": {"category": "ci_cd", "name": "CircleCI"},
    ".travis.yml": {"category": "ci_cd", "name": "Travis CI"},
    "bitbucket-pipelines.yml": {"category": "ci_cd", "name": "Bitbucket Pipelines"},
    "azure-pipelines.yml": {"category": "ci_cd", "name": "Azure Pipelines"},
    "k8s/": {"category": "orchestration", "name": "Kubernetes"},
    "kubernetes/": {"category": "orchestration", "name": "Kubernetes"},
    "helm/": {"category": "orchestration", "name": "Helm Charts"},
    "Chart.yaml": {"category": "orchestration", "name": "Helm Charts"},
    "terraform/": {"category": "iac", "name": "Terraform"},
    "main.tf": {"category": "iac", "name": "Terraform"},
    ".eslintrc": {"category": "quality", "name": "ESLint"},
    ".eslintrc.js": {"category": "quality", "name": "ESLint"},
    ".eslintrc.json": {"category": "quality", "name": "ESLint"},
    "eslint.config.js": {"category": "quality", "name": "ESLint"},
    ".prettierrc": {"category": "quality", "name": "Prettier"},
    ".flake8": {"category": "quality", "name": "Flake8"},
    "pyproject.toml": {"category": "quality", "name": "Python Project Config"},
    "ruff.toml": {"category": "quality", "name": "Ruff"},
    "vercel.json": {"category": "deployment", "name": "Vercel"},
    "netlify.toml": {"category": "deployment", "name": "Netlify"},
    "fly.toml": {"category": "deployment", "name": "Fly.io"},
    "Procfile": {"category": "deployment", "name": "Heroku"},
    ".env.example": {"category": "config", "name": "Environment Config"},
    ".env.sample": {"category": "config", "name": "Environment Config"},
    "Makefile": {"category": "build", "name": "Make"},
}


def detect_infrastructure(project_root: str) -> dict:
    """Detect infrastructure, CI/CD, deployment, and tooling configuration."""
    root = Path(project_root)
    found = {}

    for file_or_dir, info in INFRA_FILES.items():
        target = root / file_or_dir
        if target.exists():
            cat = info["category"]
            found.setdefault(cat, [])
            if info["name"] not in found[cat]:
                found[cat].append(info["name"])

    gh_workflows = root / ".github" / "workflows"
    if gh_workflows.is_dir():
        found.setdefault("ci_cd", [])
        if "GitHub Actions" not in found["ci_cd"]:
            found["ci_cd"].append("GitHub Actions")
        workflow_count = len(list(gh_workflows.glob("*.yml"))) + len(list(gh_workflows.glob("*.yaml")))
        if workflow_count > 0:
            found["ci_cd_detail"] = [f"{workflow_count} workflow(s)"]

    return found


# ── Test coverage estimation ─────────────────────────────────────────────────

TEST_FRAMEWORKS = {
    ".py": {"pytest", "unittest", "nose", "hypothesis"},
    ".js": {"jest", "mocha", "vitest", "ava", "tape", "jasmine"},
    ".ts": {"jest", "mocha", "vitest", "jasmine"},
    ".go": {"testing"},
    ".java": {"junit", "testng", "mockito"},
    ".rs": {"test"},
    ".rb": {"rspec", "minitest"},
    ".cs": {"xunit", "nunit", "mstest"},
}


def analyze_test_coverage(file_index: dict) -> dict:
    """Estimate test coverage and identify test infrastructure."""
    test_files = []
    source_files = []
    test_frameworks_found = set()

    for rel_path, meta in file_index.items():
        path_lower = rel_path.lower()
        path_parts = {p.lower() for p in Path(rel_path).parts}
        is_test = (
            any(seg in path_parts for seg in TEST_PATH_SEGMENTS)
            or "test_" in Path(rel_path).name.lower()
            or ".test." in path_lower
            or ".spec." in path_lower
            or "_test." in path_lower
        )

        if is_test:
            test_files.append(rel_path)
            chunks = meta.get("chunks", [])
            content = "\n".join(c.get("content", "") for c in chunks[:3])
            ext = meta.get("extension", "")
            frameworks = TEST_FRAMEWORKS.get(ext, set())
            for fw in frameworks:
                if fw.lower() in content.lower():
                    test_frameworks_found.add(fw)
        else:
            source_files.append(rel_path)

    total = len(file_index)
    test_count = len(test_files)
    source_count = len(source_files)
    ratio = round(test_count / source_count * 100, 1) if source_count > 0 else 0

    return {
        "test_file_count": test_count,
        "source_file_count": source_count,
        "test_to_source_ratio": ratio,
        "test_ratio_label": (
            "Strong" if ratio >= 80 else
            "Good" if ratio >= 50 else
            "Moderate" if ratio >= 25 else
            "Low" if ratio > 0 else
            "None detected"
        ),
        "test_frameworks": sorted(test_frameworks_found),
        "test_files": test_files[:20],
    }


# ── Top files by connectivity ────────────────────────────────────────────────

def find_top_connected_files(file_index: dict, imports_data: list) -> list[dict]:
    """Find files most imported by others (highest fan-in)."""
    imported_by = Counter()

    for entry in imports_data:
        if isinstance(entry, (list, tuple)):
            target = entry[2] if len(entry) > 2 else None
        elif isinstance(entry, dict):
            target = entry.get("target_path")
        else:
            continue
        if target:
            imported_by[target] += 1

    top = imported_by.most_common(10)
    results = []
    for path, count in top:
        meta = file_index.get(path, {})
        results.append({
            "file": path,
            "imported_by_count": count,
            "symbols": meta.get("symbols", [])[:8],
            "role": "Core dependency — many files depend on this",
        })

    return results


# ── External dependency analysis ─────────────────────────────────────────────

def detect_external_dependencies(project_root: str) -> dict:
    """Parse dependency files to extract external libraries."""
    root = Path(project_root)
    deps = {}

    # package.json
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(errors="replace"))
            prod = data.get("dependencies", {})
            dev = data.get("devDependencies", {})
            deps["npm"] = {
                "production": list(prod.keys()),
                "development": list(dev.keys()),
                "total": len(prod) + len(dev),
            }
        except (json.JSONDecodeError, Exception):
            pass

    # requirements.txt
    req_txt = root / "requirements.txt"
    if req_txt.exists():
        try:
            lines = req_txt.read_text(errors="replace").strip().split("\n")
            packages = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    name = re.split(r"[>=<!\[]", line)[0].strip()
                    if name:
                        packages.append(name)
            deps["pip"] = {"packages": packages, "total": len(packages)}
        except Exception:
            pass

    # pyproject.toml (basic parsing)
    pyproject = root / "pyproject.toml"
    if pyproject.exists() and "pip" not in deps:
        try:
            content = pyproject.read_text(errors="replace")
            dep_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if dep_match:
                dep_str = dep_match.group(1)
                packages = re.findall(r'"([^"]+)"', dep_str)
                names = [re.split(r"[>=<!\[]", p)[0].strip() for p in packages]
                deps["pip"] = {"packages": names, "total": len(names)}
        except Exception:
            pass

    # go.mod
    go_mod = root / "go.mod"
    if go_mod.exists():
        try:
            content = go_mod.read_text(errors="replace")
            require_match = re.findall(r'^\s+([\w./\-]+)\s+v', content, re.MULTILINE)
            deps["go"] = {"modules": require_match, "total": len(require_match)}
        except Exception:
            pass

    # Cargo.toml
    cargo = root / "Cargo.toml"
    if cargo.exists():
        try:
            content = cargo.read_text(errors="replace")
            crate_matches = re.findall(r'^(\w[\w-]*)\s*=', content, re.MULTILINE)
            skip = {"name", "version", "edition", "authors", "description", "license", "repository"}
            crates = [c for c in crate_matches if c not in skip]
            deps["cargo"] = {"crates": crates, "total": len(crates)}
        except Exception:
            pass

    # Gemfile
    gemfile = root / "Gemfile"
    if gemfile.exists():
        try:
            content = gemfile.read_text(errors="replace")
            gems = re.findall(r"""gem\s+['"]([^'"]+)['"]""", content)
            deps["bundler"] = {"gems": gems, "total": len(gems)}
        except Exception:
            pass

    return deps


# ── Comprehensive project report ─────────────────────────────────────────────

def build_comprehensive_report(
    project_root: str, file_index: dict, symbol_map: dict, imports_data: list = None
) -> dict:
    """Build a comprehensive project report with all analysis."""
    basic_summary = build_project_summary(project_root, file_index, symbol_map)
    architecture = detect_architecture_pattern(file_index, project_root)
    entry_points = find_entry_points(file_index, project_root)
    complexity = compute_complexity_metrics(file_index)
    infrastructure = detect_infrastructure(project_root)
    tests = analyze_test_coverage(file_index)
    external_deps = detect_external_dependencies(project_root)
    top_connected = find_top_connected_files(file_index, imports_data or [])

    insights = _generate_insights(
        basic_summary, architecture, complexity, tests, infrastructure, entry_points
    )

    return {
        "project_description": basic_summary.get("project_description", ""),
        "readme_excerpt": basic_summary.get("readme_content", "")[:500],
        "architecture": architecture,
        "entry_points": entry_points,
        "complexity": complexity,
        "infrastructure": infrastructure,
        "test_analysis": tests,
        "external_dependencies": external_deps,
        "top_connected_files": top_connected,
        "framework_hints": basic_summary.get("framework_hints", []),
        "dependency_files": basic_summary.get("dependency_files", []),
        "directory_tree": basic_summary.get("directory_tree", {}),
        "insights": insights,
    }


def _generate_insights(
    summary: dict, architecture: dict, complexity: dict,
    tests: dict, infrastructure: dict, entry_points: list
) -> list[str]:
    """Generate developer-focused insights about the project."""
    insights = []

    total_files = complexity.get("total_files", 0)
    total_lines = complexity.get("total_lines", 0)
    if total_files < 10:
        insights.append(f"Small project with {total_files} files and ~{total_lines} lines of code")
    elif total_files < 50:
        insights.append(f"Medium-sized project: {total_files} files, ~{total_lines:,} lines of code")
    elif total_files < 200:
        insights.append(f"Substantial project: {total_files} files, ~{total_lines:,} lines of code")
    else:
        insights.append(f"Large codebase: {total_files} files, ~{total_lines:,} lines of code")

    primary = architecture.get("primary", {})
    if primary.get("pattern") and primary["pattern"] != "Standard":
        insights.append(f"Architecture: {primary['pattern']} — {primary.get('description', '')}")

    frameworks = summary.get("framework_hints", [])
    if frameworks:
        insights.append(f"Tech stack: {', '.join(frameworks)}")

    test_ratio = tests.get("test_to_source_ratio", 0)
    test_label = tests.get("test_ratio_label", "None")
    if test_ratio > 0:
        insights.append(f"Test coverage: {test_label} ({test_ratio}% test-to-source ratio, {tests['test_file_count']} test files)")
    else:
        insights.append("No test files detected — consider adding tests")

    if infrastructure.get("ci_cd"):
        insights.append(f"CI/CD: {', '.join(infrastructure['ci_cd'])}")
    if infrastructure.get("containerization"):
        insights.append(f"Containerized with {', '.join(infrastructure['containerization'])}")
    if infrastructure.get("deployment"):
        insights.append(f"Deployment: {', '.join(infrastructure['deployment'])}")

    if entry_points:
        main_entries = [e["file"] for e in entry_points[:3]]
        insights.append(f"Entry points: {', '.join(main_entries)}")

    avg_lines = complexity.get("avg_file_size_lines", 0)
    if avg_lines > 300:
        insights.append(f"Average file length is {avg_lines} lines — some files may benefit from splitting")
    elif avg_lines > 0:
        insights.append(f"Clean file sizes averaging {avg_lines} lines per file")

    most_complex = complexity.get("most_complex_files", [])
    if most_complex and most_complex[0]["symbol_count"] > 20:
        top = most_complex[0]
        insights.append(f"Most complex file: {top['file']} with {top['symbol_count']} symbols")

    return insights

