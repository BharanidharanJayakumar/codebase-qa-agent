import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

SCHEMA_VERSION = 5
INDEX_DIR = Path.home() / ".codebase-qa-agent"
DB_FILE = INDEX_DIR / "index.db"  # legacy single-project path
# Keep the old JSON path for migration
LEGACY_JSON = INDEX_DIR / "index.json"


def _make_slug(project_root: str) -> str:
    """Generate a human-readable slug from a project path. e.g. 'codebase-qa-agent'."""
    return Path(project_root).name.lower().replace(" ", "-")


def _make_project_id(project_root: str) -> str:
    """Generate a unique project ID: slug + short hash. e.g. 'codebase-qa-agent_a1b2c3d4e5f6'."""
    slug = _make_slug(project_root)
    short_hash = hashlib.sha256(project_root.encode()).hexdigest()[:12]
    return f"{slug}_{short_hash}"


def _project_db_path(project_root: str) -> Path:
    """Return a per-project DB path based on project ID."""
    project_id = _make_project_id(project_root)
    return INDEX_DIR / "projects" / f"{project_id}.db"


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

        CREATE TABLE IF NOT EXISTS embeddings (
            rel_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY (rel_path, chunk_index),
            FOREIGN KEY (rel_path) REFERENCES files(rel_path) ON DELETE CASCADE
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
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM symbols")
        conn.execute("DELETE FROM keyword_files")
        conn.execute("DELETE FROM files")

        # Metadata
        slug = _make_slug(project_root)
        project_id = _make_project_id(project_root)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('project_root', ?)", (project_root,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('indexed_at', ?)", (str(indexed_at),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('total_files', ?)", (str(len(file_index)),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('slug', ?)", (slug,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('project_id', ?)", (project_id,))

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
    """Load the index for a specific project. Accepts path, slug, or project_id.
    Falls back to legacy DB if no project_path.
    If the DB is corrupted, deletes it and returns None (triggers re-index)."""
    # Try legacy JSON migration first
    if not DB_FILE.exists() and LEGACY_JSON.exists():
        return _migrate_from_json()

    # Determine which DB to open — supports path, slug, or project_id
    db_path = resolve_project_db(project_path) if project_path else None
    if db_path is None:
        db_path = _find_latest_project_db()
    if db_path is None:
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

        slug_row = conn.execute("SELECT value FROM meta WHERE key='slug'").fetchone()
        pid_row = conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
        root_val = project_root["value"]

        return {
            "schema_version": SCHEMA_VERSION,
            "project_root": root_val,
            "project_id": pid_row["value"] if pid_row else _make_project_id(root_val),
            "slug": slug_row["value"] if slug_row else _make_slug(root_val),
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
    """List all indexed projects with metadata including slug and project_id."""
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
            slug_row = conn.execute("SELECT value FROM meta WHERE key='slug'").fetchone()
            pid_row = conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
            conn.close()
            if root:
                project_root = root["value"]
                results.append({
                    "project_id": pid_row["value"] if pid_row else _make_project_id(project_root),
                    "slug": slug_row["value"] if slug_row else _make_slug(project_root),
                    "project_root": project_root,
                    "indexed_at": float(at["value"]) if at else 0,
                    "total_files": int(total["value"]) if total else 0,
                })
        except Exception:
            continue
    return results


def delete_project(project_identifier: str) -> bool:
    """Delete an indexed project by path, slug, or project_id. Returns True if deleted."""
    projects_dir = INDEX_DIR / "projects"
    if not projects_dir.exists():
        return False

    for db_path in projects_dir.glob("*.db"):
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            root = conn.execute("SELECT value FROM meta WHERE key='project_root'").fetchone()
            slug_row = conn.execute("SELECT value FROM meta WHERE key='slug'").fetchone()
            pid_row = conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
            conn.close()

            match = False
            if root and root["value"] == project_identifier:
                match = True
            elif slug_row and slug_row["value"] == project_identifier:
                match = True
            elif pid_row and pid_row["value"] == project_identifier:
                match = True

            if match:
                db_path.unlink()
                return True
        except Exception:
            continue
    return False


def resolve_project_db(identifier: str) -> Path | None:
    """Resolve a project identifier (path, slug, or project_id) to its DB path."""
    if not identifier:
        return _find_latest_project_db()

    # Direct path match first (fastest)
    direct = _project_db_path(identifier)
    if direct.exists():
        return direct

    # Search by slug or project_id
    projects_dir = INDEX_DIR / "projects"
    if not projects_dir.exists():
        return None

    for db_path in projects_dir.glob("*.db"):
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            slug_row = conn.execute("SELECT value FROM meta WHERE key='slug'").fetchone()
            pid_row = conn.execute("SELECT value FROM meta WHERE key='project_id'").fetchone()
            conn.close()

            if slug_row and slug_row["value"] == identifier:
                return db_path
            if pid_row and pid_row["value"] == identifier:
                return db_path
        except Exception:
            continue
    return None


def save_embeddings(project_root: str, embeddings_data: list[tuple[str, int, bytes]]) -> None:
    """Save pre-computed embeddings to the project DB.
    embeddings_data: list of (rel_path, chunk_index, vector_bytes)"""
    db_path = _project_db_path(project_root)
    if not db_path.exists():
        return
    conn = _get_db(db_path)
    try:
        conn.execute("DELETE FROM embeddings")
        conn.executemany(
            "INSERT INTO embeddings (rel_path, chunk_index, vector) VALUES (?, ?, ?)",
            embeddings_data,
        )
        conn.commit()
    finally:
        conn.close()


def load_embeddings(project_root: str = "", identifier: str = "") -> list[tuple[str, int, bytes]]:
    """Load pre-computed embeddings from the project DB.
    Returns list of (rel_path, chunk_index, vector_bytes)."""
    if identifier:
        db_path = resolve_project_db(identifier)
    elif project_root:
        db_path = _project_db_path(project_root)
    else:
        db_path = _find_latest_project_db()

    if not db_path or not db_path.exists():
        return []

    try:
        conn = _get_db(db_path)
        rows = conn.execute("SELECT rel_path, chunk_index, vector FROM embeddings").fetchall()
        conn.close()
        return [(r["rel_path"], r["chunk_index"], r["vector"]) for r in rows]
    except Exception:
        return []


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
