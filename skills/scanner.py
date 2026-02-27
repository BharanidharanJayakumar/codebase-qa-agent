import os
import stat
from pathlib import Path

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
    # Go
    "vendor",
    # Ruby
    ".bundle",
    # General
    "tmp", "temp", "logs", ".cache",
}

# Maximum file size to index (1MB). Anything larger is likely generated code.
MAX_INDEXABLE_SIZE = 1_000_000


def _is_binary(file_path: Path, sample_size: int = 8192) -> bool:
    """Detect binary files by checking for null bytes in the first 8KB."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except (OSError, PermissionError):
        return True  # Can't read → treat as binary → skip


def _is_safe_path(file_path: Path, root: Path) -> bool:
    """Ensure a resolved path stays within the project root (prevents traversal)."""
    try:
        file_path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def scan_directory(root_path: str) -> list[dict]:
    """
    Walk a directory tree and return metadata for every indexable file.

    Safety guards:
    - Skips symlinks that point outside the project root
    - Skips dangling symlinks
    - Skips binary files
    - Skips files larger than MAX_INDEXABLE_SIZE
    - Validates all paths stay within root (prevents traversal via symlinks)
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise ValueError(f"Path does not exist: {root_path}")
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {root_path}")

    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Skip ignored directories and symlinked directories
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS
            and not (Path(dirpath) / d).is_symlink()
        ]

        for filename in filenames:
            file_path = Path(dirpath) / filename

            # Skip symlinks — they can point outside the project or be dangling
            if file_path.is_symlink():
                continue

            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue

            # stat() can fail on race conditions (file deleted between walk and stat)
            try:
                file_stat = file_path.stat()
            except (OSError, PermissionError):
                continue

            # Skip non-regular files (devices, sockets, etc.)
            if not stat.S_ISREG(file_stat.st_mode):
                continue

            # Skip oversized files (likely generated)
            if file_stat.st_size > MAX_INDEXABLE_SIZE:
                continue

            # Skip empty files
            if file_stat.st_size == 0:
                continue

            # Ensure path hasn't escaped root via symlinks in parent dirs
            if not _is_safe_path(file_path, root):
                continue

            files.append({
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(root)),
                "extension": file_path.suffix,
                "size_bytes": file_stat.st_size,
                "last_modified": file_stat.st_mtime,
            })

    return sorted(files, key=lambda f: f["relative_path"])


def read_file(file_path: str, max_bytes: int = 50_000) -> dict:
    """
    Read a file's content safely.

    Guards:
    - Binary detection (null bytes in first 8KB)
    - max_bytes cap (default 50KB)
    - Permission and OS error handling
    """
    path = Path(file_path)

    # Binary check before reading full content
    if _is_binary(path):
        return {"path": str(path), "content": "", "error": "binary_file"}

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_bytes)

        # Double-check: if content looks binary after read (e.g. lots of replacement chars)
        replacement_ratio = content.count("\ufffd") / max(len(content), 1)
        if replacement_ratio > 0.1:
            return {"path": str(path), "content": "", "error": "binary_file"}

        try:
            truncated = path.stat().st_size > max_bytes
        except OSError:
            truncated = False

        return {
            "path": str(path),
            "content": content,
            "truncated": truncated,
            "encoding": "utf-8",
        }
    except PermissionError:
        return {"path": str(path), "content": "", "error": "permission_denied"}
    except OSError as e:
        return {"path": str(path), "content": "", "error": str(e)}
