"""Tests for skills/storage.py — index persistence, project management, sessions."""
import sys
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import skills.storage as storage


def _make_test_data():
    """Create minimal test index data."""
    file_index = {
        "src/main.py": {
            "chunks": [
                {"start_line": 1, "end_line": 10, "content": "def main():\n    pass", "symbol": "main"},
            ],
            "keywords": ["main", "entry"],
            "symbols": ["main"],
            "extension": ".py",
            "size_bytes": 100,
            "last_modified": time.time(),
        }
    }
    keyword_map = {"main": ["src/main.py"], "entry": ["src/main.py"]}
    symbol_map = {"main": [{"file": "src/main.py", "line": 1, "type": "function"}]}
    return file_index, keyword_map, symbol_map


class TestSaveLoadIndex:
    def setup_method(self):
        self._orig_index_dir = storage.INDEX_DIR
        self._tmpdir = tempfile.mkdtemp()
        storage.INDEX_DIR = Path(self._tmpdir)

    def teardown_method(self):
        storage.INDEX_DIR = self._orig_index_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        project_root = "/tmp/test-project"

        storage.save_index(file_index, keyword_map, symbol_map, project_root, time.time())
        loaded = storage.load_index(project_root)

        assert loaded is not None
        assert loaded["project_root"] == project_root
        assert "src/main.py" in loaded["file_index"]
        assert loaded["file_index"]["src/main.py"]["keywords"] == ["main", "entry"]
        assert "main" in loaded["keyword_map"]
        assert "main" in loaded["symbol_map"]

    def test_load_returns_none_for_nonexistent(self):
        loaded = storage.load_index("/nonexistent/project")
        assert loaded is None

    def test_slug_and_project_id(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        project_root = "/home/user/my-cool-project"

        storage.save_index(file_index, keyword_map, symbol_map, project_root, time.time())
        loaded = storage.load_index(project_root)

        assert loaded["slug"] == "my-cool-project"
        assert loaded["project_id"].startswith("my-cool-project_")

    def test_load_by_slug(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        project_root = "/home/user/my-project"

        storage.save_index(file_index, keyword_map, symbol_map, project_root, time.time())
        loaded = storage.load_index("my-project")

        assert loaded is not None
        assert loaded["project_root"] == project_root

    def test_load_by_project_id(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        project_root = "/home/user/test-app"

        storage.save_index(file_index, keyword_map, symbol_map, project_root, time.time())
        project_id = storage._make_project_id(project_root)
        loaded = storage.load_index(project_id)

        assert loaded is not None
        assert loaded["project_root"] == project_root

    def test_list_indexed_projects(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        storage.save_index(file_index, keyword_map, symbol_map, "/tmp/proj-a", time.time())
        storage.save_index(file_index, keyword_map, symbol_map, "/tmp/proj-b", time.time())

        projects = storage.list_indexed_projects()
        slugs = [p["slug"] for p in projects]
        assert "proj-a" in slugs
        assert "proj-b" in slugs
        assert len(projects) == 2

    def test_delete_project_by_slug(self):
        file_index, keyword_map, symbol_map = _make_test_data()
        storage.save_index(file_index, keyword_map, symbol_map, "/tmp/deleteme", time.time())

        assert storage.delete_project("deleteme") is True
        assert storage.load_index("/tmp/deleteme") is None

    def test_delete_nonexistent(self):
        assert storage.delete_project("nope") is False


class TestSessions:
    def setup_method(self):
        self._orig_index_dir = storage.INDEX_DIR
        self._tmpdir = tempfile.mkdtemp()
        storage.INDEX_DIR = Path(self._tmpdir)

    def teardown_method(self):
        storage.INDEX_DIR = self._orig_index_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_session(self):
        storage.save_session_turn("s1", "What is main?", "It's the entry point.", ["main.py"])
        storage.save_session_turn("s1", "What about tests?", "No tests found.", [])

        turns = storage.load_session("s1")
        assert len(turns) == 2
        assert turns[0]["question"] == "What is main?"
        assert turns[1]["question"] == "What about tests?"

    def test_load_empty_session(self):
        turns = storage.load_session("nonexistent")
        assert turns == []

    def test_session_max_turns(self):
        for i in range(10):
            storage.save_session_turn("s2", f"Q{i}", f"A{i}", [])

        turns = storage.load_session("s2", max_turns=3)
        assert len(turns) == 3
