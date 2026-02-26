import json
import os
import sqlite3
import tempfile
from pathlib import Path

SCHEMA_VERSION = 3
INDEX_DIR = Path.home() / ".codebase-qa-agent"
DB_FILE = INDEX_DIR / "index.db"
# Keep the old JSON path for migration
LEGACY_JSON = INDEX_DIR / "index.json"


def _get_db() -> sqlite3.Connection:
    """Open the SQLite database, creating tables if needed."""
    INDEX_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
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
    """Save the full index to SQLite. Replaces all existing data."""
    conn = _get_db()
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


def load_index() -> dict | None:
    """Load the full index from SQLite. Returns None if no index exists.
    If the DB is corrupted, deletes it and returns None (triggers re-index)."""
    # Try legacy JSON migration first
    if not DB_FILE.exists() and LEGACY_JSON.exists():
        return _migrate_from_json()

    if not DB_FILE.exists():
        return None

    try:
        conn = _get_db()
    except Exception:
        # DB file exists but is corrupted — delete and trigger re-index
        DB_FILE.unlink(missing_ok=True)
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
        # DB is corrupted or has invalid data — delete and trigger re-index
        conn.close()
        DB_FILE.unlink(missing_ok=True)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
