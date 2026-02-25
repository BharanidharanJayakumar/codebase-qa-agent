import os
from pathlib import Path
from typing import Iterator

# File types we care about — skipping images, binaries, lock files etc.
# These are the files that actually contain logic we want to understand
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c",
    ".rb", ".php", ".cs", ".swift",
    ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".env.example",
    ".md", ".txt", ".sh",
}

# Directories that never contain useful project code
# Indexing node_modules or .git would be both slow and useless
IGNORED_DIRS = {
    # Universal
    ".git",
    # Python
    "node_modules", "__pycache__", ".venv", "venv", "env", ".env",
    ".pytest_cache", ".mypy_cache",
    # JS/TS
    "dist", "build", ".next", ".nuxt", "coverage", ".turbo",
    # Java / Kotlin / Gradle
    "target", ".gradle", "out", "classes",
    # .NET / C#
    "bin", "obj", ".vs", "packages",
    # Rust
    # "target" already covered above
    # Go
    "vendor",
    # Ruby
    ".bundle",
    # General
    "tmp", "temp", "logs", ".cache",
}


def scan_directory(root_path: str) -> list[dict]:
    """
    Walk a directory tree and return metadata for every indexable file.

    Returns a list of dicts — each with path, extension, size, last_modified.
    We capture last_modified here so the indexer can later decide
    which files actually changed and need re-indexing.
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise ValueError(f"Path does not exist: {root_path}")

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames in-place — this tells os.walk to skip ignored dirs
        # without descending into them at all (important for node_modules etc.)
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue

            stat = file_path.stat()
            files.append({
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(root)),
                "extension": file_path.suffix,
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
            })

    # Sort by path so the index is deterministic and easy to read in logs
    return sorted(files, key=lambda f: f["relative_path"])


def read_file(file_path: str, max_bytes: int = 50_000) -> dict:
    """
    Read a file's content safely.

    max_bytes cap prevents feeding a 10MB generated file into the LLM.
    50KB is enough for any real source file — if a file is bigger,
    something unusual is going on (generated code, data files, etc.)
    """
    path = Path(file_path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_bytes)
            truncated = path.stat().st_size > max_bytes

        return {
            "path": str(path),
            "content": content,
            "truncated": truncated,
            "encoding": "utf-8",
        }
    except PermissionError:
        return {"path": str(path), "content": "", "error": "permission_denied"}
    except Exception as e:
        return {"path": str(path), "content": "", "error": str(e)}


def get_changed_files(root_path: str, since_timestamp: float) -> list[dict]:
    """
    Return only files modified after `since_timestamp` (a Unix timestamp).

    This is what makes update_index efficient — instead of re-reading
    all 200 files when you edit one, we only process what actually changed.
    """
    all_files = scan_directory(root_path)
    return [f for f in all_files if f["last_modified"] > since_timestamp]
