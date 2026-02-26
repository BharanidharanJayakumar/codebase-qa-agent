import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

SCHEMA_VERSION = 4
INDEX_DIR = Path.home() / ".codebase-qa-agent"
DB_FILE = INDEX_DIR / "index.db"  # legacy single-project path
# Keep the old JSON path for migration
LEGACY_JSON = INDEX_DIR / "index.json"


def _project_db_path(project_root: str) -> Path:
    """Return a per-project DB path based on a hash of the project root."""
    slug = hashlib.sha256(project_root.encode()).hexdigest()[:12]
    name = Path(project_root).name  # human-readable prefix
    return INDEX_DIR / "projects" / f"{name}_{slug}.db"


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the SQLite database, creating tables if needed."""
    if db_path is None:
        db_path = DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # write-ahead logging for crash safety
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            rel_path TEXT PRIMARY KEY,
            extension TEXT,
            size_bytes INTEGER,
            last_modified REAL,
            keywords TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rel_path TEXT NOT NULL REFERENCES files(rel_path) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            content TEXT,
            symbol_name TEXT
        );

        CREATE TABLE IF NOT EXISTS symbols (
            name TEXT NOT NULL,
            rel_path TEXT NOT NULL REFERENCES files(rel_path) ON DELETE CASCADE,
            line INTEGER,
            type TEXT,
            PRIMARY KEY (name, rel_path, line)
        );

        CREATE TABLE IF NOT EXISTS keyword_files (
            keyword TEXT NOT NULL,
            rel_path TEXT NOT NULL REFERENCES files(rel_path) ON DELETE CASCADE,
            PRIMARY KEY (keyword, rel_path)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_keyword_files_keyword ON keyword_files(keyword);
    """)


def save_index(file_index: dict, keyword_map: dict, symbol_map: dict,
               project_root: str, indexed_at: float) -> None:
    """Save the full index to a per-project SQLite database."""
    db_path = _project_db_path(project_root)
    conn = _get_db(db_path)
    try:
        conn.execute("BEGIN")
        # Clear existing data
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM symbols")
        conn.execute("DELETE FROM keyword_files")
        conn.execute("DELETE FROM files")

        # Metadata
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('project_root', ?)", (project_root,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('indexed_at', ?)", (str(indexed_at),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('total_files', ?)", (str(len(file_index)),))

        # Files
        for rel_path, meta in file_index.items():
            conn.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                (rel_path, meta["extension"], meta["size_bytes"],
                 meta["last_modified"], json.dumps(meta["keywords"]))
            )
            # Chunks
            for i, chunk in enumerate(meta.get("chunks", [])):
                conn.execute(
                    "INSERT INTO chunks (rel_path, chunk_index, start_line, end_line, content, symbol_name) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (rel_path, i, chunk["start_line"], chunk["end_line"],
                     chunk["content"], chunk.get("symbol"))
                )

        # Symbols (one-to-many)
        for name, locations in symbol_map.items():
            for loc in locations:
                conn.execute(
                    "INSERT OR REPLACE INTO symbols VALUES (?, ?, ?, ?)",
                    (name, loc["file"], loc["line"], loc["type"])
                )

        # Keywords
        for keyword, rel_paths in keyword_map.items():
            for rel_path in rel_paths:
                conn.execute(
                    "INSERT OR REPLACE INTO keyword_files VALUES (?, ?)",
                    (keyword, rel_path)
                )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Clean up legacy JSON if it exists
    if LEGACY_JSON.exists():
        LEGACY_JSON.unlink()


def load_index(project_path: str = "") -> dict | None:
    """Load the index for a specific project. Falls back to legacy DB if no project_path.
    If the DB is corrupted, deletes it and returns None (triggers re-index)."""
    # Try legacy JSON migration first
    if not DB_FILE.exists() and LEGACY_JSON.exists():
        return _migrate_from_json()

    # Determine which DB to open
    if project_path:
        db_path = _project_db_path(project_path)
    else:
        # Check per-project DBs first (return most recently indexed)
        db_path = _find_latest_project_db()
        if db_path is None:
            # Fall back to legacy single-project DB
            db_path = DB_FILE

    if not db_path.exists():
        return None

    try:
        conn = _get_db(db_path)
    except Exception:
        db_path.unlink(missing_ok=True)
        return None

    try:
        # Check schema version
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not row or int(row["value"]) != SCHEMA_VERSION:
            return None

        project_root = conn.execute("SELECT value FROM meta WHERE key='project_root'").fetchone()
        indexed_at = conn.execute("SELECT value FROM meta WHERE key='indexed_at'").fetchone()
        if not project_root or not indexed_at:
            return None

        # Build file_index
        file_index = {}
        for file_row in conn.execute("SELECT * FROM files").fetchall():
            rel_path = file_row["rel_path"]
            chunks = []
            for chunk_row in conn.execute(
                "SELECT * FROM chunks WHERE rel_path=? ORDER BY chunk_index", (rel_path,)
            ).fetchall():
                chunks.append({
                    "start_line": chunk_row["start_line"],
                    "end_line": chunk_row["end_line"],
                    "content": chunk_row["content"],
                    "symbol": chunk_row["symbol_name"],
                })

            sym_rows = conn.execute("SELECT name FROM symbols WHERE rel_path=?", (rel_path,)).fetchall()

            file_index[rel_path] = {
                "chunks": chunks,
                "keywords": json.loads(file_row["keywords"]),
                "symbols": [r["name"] for r in sym_rows],
                "extension": file_row["extension"],
                "size_bytes": file_row["size_bytes"],
                "last_modified": file_row["last_modified"],
            }

        # Build keyword_map
        keyword_map = {}
        for row in conn.execute("SELECT keyword, rel_path FROM keyword_files").fetchall():
            keyword_map.setdefault(row["keyword"], []).append(row["rel_path"])

        # Build symbol_map (one-to-many)
        symbol_map = {}
        for row in conn.execute("SELECT name, rel_path, line, type FROM symbols").fetchall():
            symbol_map.setdefault(row["name"], []).append({
                "file": row["rel_path"],
                "line": row["line"],
                "type": row["type"],
            })

        return {
            "schema_version": SCHEMA_VERSION,
            "project_root": project_root["value"],
            "indexed_at": float(indexed_at["value"]),
            "file_index": file_index,
            "keyword_map": keyword_map,
            "symbol_map": symbol_map,
        }
    except (sqlite3.DatabaseError, json.JSONDecodeError, KeyError, ValueError):
        conn.close()
        db_path.unlink(missing_ok=True)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _find_latest_project_db() -> Path | None:
    """Find the most recently modified per-project DB."""
    projects_dir = INDEX_DIR / "projects"
    if not projects_dir.exists():
        return None
    dbs = list(projects_dir.glob("*.db"))
    if not dbs:
        return None
    return max(dbs, key=lambda p: p.stat().st_mtime)


def list_indexed_projects() -> list[dict]:
    """List all indexed projects with metadata."""
    projects_dir = INDEX_DIR / "projects"
    if not projects_dir.exists():
        return []
    results = []
    for db_path in projects_dir.glob("*.db"):
        try:
            conn = _get_db(db_path)
            root = conn.execute("SELECT value FROM meta WHERE key='project_root'").fetchone()
            at = conn.execute("SELECT value FROM meta WHERE key='indexed_at'").fetchone()
            total = conn.execute("SELECT value FROM meta WHERE key='total_files'").fetchone()
            conn.close()
            if root:
                results.append({
                    "project_root": root["value"],
                    "indexed_at": float(at["value"]) if at else 0,
                    "total_files": int(total["value"]) if total else 0,
                    "db_file": str(db_path),
                })
        except Exception:
            continue
    return results


def _get_sessions_db() -> sqlite3.Connection:
    """Open the shared sessions database (not per-project)."""
    INDEX_DIR.mkdir(exist_ok=True)
    db_path = INDEX_DIR / "sessions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            relevant_files TEXT,
            created_at REAL NOT NULL,
            PRIMARY KEY (id, turn_index)
        )
    """)
    return conn


def save_session_turn(session_id: str, question: str, answer: str,
                      relevant_files: list[str]) -> int:
    """Append a Q&A turn to a session. Returns the turn index."""
    import time as _time
    conn = _get_sessions_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) FROM sessions WHERE id=?",
            (session_id,)
        ).fetchone()
        turn_index = row[0] + 1
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, turn_index, question, answer,
             json.dumps(relevant_files), _time.time())
        )
        conn.commit()
        return turn_index
    finally:
        conn.close()


def load_session(session_id: str, max_turns: int = 5) -> list[dict]:
    """Load the last N turns of a session for context."""
    try:
        conn = _get_sessions_db()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT question, answer, relevant_files FROM sessions "
            "WHERE id=? ORDER BY turn_index DESC LIMIT ?",
            (session_id, max_turns)
        ).fetchall()
        turns = [
            {
                "question": r["question"],
                "answer": r["answer"],
                "relevant_files": json.loads(r["relevant_files"]),
            }
            for r in reversed(rows)  # chronological order
        ]
        return turns
    except Exception:
        return []
    finally:
        conn.close()


def _migrate_from_json() -> dict | None:
    """One-time migration: read legacy index.json, write to SQLite, delete JSON."""
    try:
        data = json.loads(LEGACY_JSON.read_text())
    except (json.JSONDecodeError, KeyError):
        return None

    # Attempt migration if the old JSON has the needed fields
    if "file_index" not in data:
        return None

    save_index(
        data["file_index"],
        data.get("keyword_map", {}),
        data.get("symbol_map", {}),
        data.get("project_root", ""),
        data.get("indexed_at", 0),
    )
    return load_index()
