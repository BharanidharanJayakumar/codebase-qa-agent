import json
import os
from pathlib import Path

# Store the index in a well-known location that survives across process executions
INDEX_DIR = Path.home() / ".codebase-qa-agent"
INDEX_FILE = INDEX_DIR / "index.json"


def save_index(file_index: dict, keyword_map: dict, symbol_map: dict, project_root: str, indexed_at: float) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    data = {
        "project_root": project_root,
        "indexed_at": indexed_at,
        "file_index": file_index,
        "keyword_map": keyword_map,
        "symbol_map": symbol_map,
    }
    INDEX_FILE.write_text(json.dumps(data))


def load_index() -> dict | None:
    if not INDEX_FILE.exists():
        return None
    return json.loads(INDEX_FILE.read_text())
