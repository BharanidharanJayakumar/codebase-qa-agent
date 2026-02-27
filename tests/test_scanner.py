"""Tests for skills/scanner.py — directory scanning and file reading."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.scanner import scan_directory, read_file, _is_binary, _is_safe_path


# --- scan_directory ---

def test_scan_finds_python_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("print('hello')")
        (Path(tmpdir) / "utils.py").write_text("def helper(): pass")
        files = scan_directory(tmpdir)
        paths = [f["relative_path"] for f in files]
        assert "main.py" in paths
        assert "utils.py" in paths


def test_scan_ignores_node_modules():
    with tempfile.TemporaryDirectory() as tmpdir:
        nm = Path(tmpdir) / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("module.exports = {}")
        (Path(tmpdir) / "app.js").write_text("const x = 1;")
        files = scan_directory(tmpdir)
        paths = [f["relative_path"] for f in files]
        assert "app.js" in paths
        assert not any("node_modules" in p for p in paths)


def test_scan_ignores_git_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        git = Path(tmpdir) / ".git"
        git.mkdir()
        (git / "config").write_text("bare = false")
        (Path(tmpdir) / "main.py").write_text("x = 1")
        files = scan_directory(tmpdir)
        assert not any(".git" in f["relative_path"] for f in files)


def test_scan_skips_empty_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "empty.py").write_text("")
        (Path(tmpdir) / "real.py").write_text("x = 1")
        files = scan_directory(tmpdir)
        paths = [f["relative_path"] for f in files]
        assert "real.py" in paths
        assert "empty.py" not in paths


def test_scan_skips_unsupported_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (Path(tmpdir) / "app.py").write_text("x = 1")
        files = scan_directory(tmpdir)
        paths = [f["relative_path"] for f in files]
        assert "app.py" in paths
        assert "image.png" not in paths


def test_scan_nonexistent_raises():
    try:
        scan_directory("/nonexistent/path/abc123")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_scan_returns_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("hello = 1")
        files = scan_directory(tmpdir)
        assert len(files) == 1
        f = files[0]
        assert "path" in f
        assert "relative_path" in f
        assert "extension" in f
        assert "size_bytes" in f
        assert "last_modified" in f
        assert f["extension"] == ".py"


# --- read_file ---

def test_read_file_normal():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("print('hello world')")
        f.flush()
        result = read_file(f.name)
        assert result["content"] == "print('hello world')"
        assert result.get("error") is None
    os.unlink(f.name)


def test_read_file_binary_detection():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"\x00\x01\x02\x03binary content")
        f.flush()
        result = read_file(f.name)
        assert result["error"] == "binary_file"
        assert result["content"] == ""
    os.unlink(f.name)


# --- _is_binary ---

def test_is_binary_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("just text")
        f.flush()
        assert _is_binary(Path(f.name)) is False
    os.unlink(f.name)


def test_is_binary_binary_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"\x00" * 100)
        f.flush()
        assert _is_binary(Path(f.name)) is True
    os.unlink(f.name)


# --- _is_safe_path ---

def test_safe_path_within_root():
    root = Path("/home/user/project")
    child = Path("/home/user/project/src/main.py")
    assert _is_safe_path(child, root) is True


def test_safe_path_outside_root():
    root = Path("/home/user/project")
    outsider = Path("/etc/passwd")
    assert _is_safe_path(outsider, root) is False
