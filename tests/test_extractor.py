"""Tests for skills/extractor.py — symbol extraction, chunking, keywords."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.extractor import extract_symbols, _extract_symbols_regex, chunk_file, extract_keywords


# --- extract_symbols (regex path) ---

def test_extract_python_symbols():
    code = """
import os

class MyClass:
    pass

def my_function(x):
    return x

async def async_handler(req):
    pass
"""
    symbols = _extract_symbols_regex(code, "test.py")
    names = [s["name"] for s in symbols]
    assert "MyClass" in names
    assert "my_function" in names
    assert "async_handler" in names
    assert len(symbols) == 3


def test_extract_js_symbols():
    code = """
function fetchData(url) {
    return fetch(url);
}

const processItems = async (items) => {
    return items;
}

class UserService {
}
"""
    symbols = _extract_symbols_regex(code, "app.js")
    names = [s["name"] for s in symbols]
    assert "fetchData" in names
    assert "processItems" in names
    assert "UserService" in names


def test_extract_go_symbols():
    code = """
func main() {
}

func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {
}

type Config struct {
    Port int
}
"""
    symbols = _extract_symbols_regex(code, "main.go")
    names = [s["name"] for s in symbols]
    assert "main" in names
    assert "handleRequest" in names
    assert "Config" in names


def test_extract_ts_interface():
    code = """
interface UserProps {
    name: string;
    age: number;
}

type Status = "active" | "inactive";
"""
    symbols = _extract_symbols_regex(code, "types.ts")
    names = [s["name"] for s in symbols]
    assert "UserProps" in names
    assert "Status" in names


def test_extract_unsupported_extension():
    symbols = _extract_symbols_regex("some content", "data.csv")
    assert symbols == []


def test_extract_empty_content():
    symbols = _extract_symbols_regex("", "test.py")
    assert symbols == []


# --- chunk_file ---

def test_chunk_file_with_symbols():
    code = """import os

def foo():
    return 1

def bar():
    return 2
"""
    symbols = [
        {"name": "foo", "type": "function", "line": 3},
        {"name": "bar", "type": "function", "line": 6},
    ]
    chunks = chunk_file(code, symbols)
    assert len(chunks) == 3  # header + foo + bar
    assert chunks[0]["symbol"] is None  # header (import)
    assert chunks[1]["symbol"] == "foo"
    assert chunks[2]["symbol"] == "bar"


def test_chunk_file_no_symbols():
    code = "line1\nline2\nline3"
    chunks = chunk_file(code, [])
    assert len(chunks) == 1
    assert chunks[0]["symbol"] is None
    assert "line1" in chunks[0]["content"]


def test_chunk_file_empty():
    chunks = chunk_file("", [])
    assert len(chunks) == 1


def test_chunk_respects_max_lines():
    lines = [f"line {i}" for i in range(200)]
    code = "\n".join(lines)
    symbols = [{"name": "big_func", "type": "function", "line": 1}]
    chunks = chunk_file(code, symbols, max_chunk_lines=60)
    assert chunks[0]["end_line"] <= 60


# --- extract_keywords ---

def test_extract_keywords_basic():
    code = "def authenticate_user(username, password): pass"
    keywords = extract_keywords(code)
    assert "authenticate" in keywords
    assert "user" in keywords
    assert "username" in keywords
    assert "password" in keywords


def test_extract_keywords_camelcase():
    code = "function getUserById(userId) { return userId; }"
    keywords = extract_keywords(code)
    assert "getuserbyid" in keywords
    assert "userid" in keywords


def test_extract_keywords_filters_stopwords():
    code = "import this from that return the result"
    keywords = extract_keywords(code)
    assert "the" not in keywords
    assert "import" not in keywords
    assert "result" in keywords


def test_extract_keywords_empty():
    keywords = extract_keywords("")
    assert keywords == []
