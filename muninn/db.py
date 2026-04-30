"""Database management for Muninn v0.2 — Disco Elysium architecture."""

import os
import sqlite3
import sqlite_vec
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema_v2.sql"


def get_db_path() -> str:
    """Get database path from environment or default."""
    return os.getenv("DB_PATH", "./muninn.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with sqlite-vec loaded."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension via Python package
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as e:
        print(f"Warning: Could not load sqlite-vec: {e}")
        print("Vector search will not work. Install: pip install sqlite-vec")

    return conn


def get_embedding_dims(conn: sqlite3.Connection) -> int:
    """Get configured embedding dimensions from DB, or default."""
    try:
        row = conn.execute(
            "SELECT value FROM embedding_config WHERE key = 'dimensions'"
        ).fetchone()
        if row:
            return int(row["value"])
    except Exception:
        pass
    return 1024  # default for Qwen3-Embedding


def init_db(db_path: str | None = None, dimensions: int = None) -> sqlite3.Connection:
    """Initialize database with v0.2 schema (Disco Elysium).

    Args:
        db_path: Path to the database file.
        dimensions: Embedding vector dimensions. If None, uses EMBEDDING_DIMS env var
                   or defaults to 1024 (Qwen3-Embedding).
    """
    if dimensions is None:
        dimensions = int(os.getenv("EMBEDDING_DIMS", "1024"))

    conn = get_connection(db_path)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # Remove sqlite CLI comments (.load directives)
    clean_schema = "\n".join(
        line for line in schema.splitlines()
        if not line.strip().startswith("-- .load")
    )

    # Replace dimension placeholders in schema
    # Supports FLOAT[1024] → FLOAT[384] etc.
    import re
    clean_schema = re.sub(r'FLOAT\[\d+\]', f'FLOAT[{dimensions}]', clean_schema)

    try:
        conn.executescript(clean_schema)
    except sqlite3.OperationalError as e:
        # Virtual tables (vec0) might fail if sqlite-vec not loaded
        if "vec0" in str(e).lower() or "no such module" in str(e).lower():
            # Re-run skipping vec0 virtual tables
            statements = []
            for statement in clean_schema.split(";"):
                s = statement.strip()
                if s and "vec0" not in s.lower():
                    try:
                        conn.execute(s)
                    except sqlite3.OperationalError:
                        pass
            print("Warning: Created tables without vector search (sqlite-vec not available)")
        else:
            raise

    # Update dimensions in config if different from schema default
    conn.execute("UPDATE embedding_config SET value = ? WHERE key = 'dimensions'", [str(dimensions)])
    conn.commit()
    return conn


if __name__ == "__main__":
    print("Initializing Muninn v0.2 database...")
    conn = init_db()
    print(f"Database created at: {get_db_path()}")
    dims = get_embedding_dims(conn)
    print(f"Embedding dimensions: {dims}")
    conn.close()
