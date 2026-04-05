"""Database management for Muninn."""

import os
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


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

    # Load sqlite-vec extension
    try:
        conn.enable_load_extension(True)
        # Try common locations
        vec_paths = [
            "vec0",  # system PATH
            os.path.join(os.path.dirname(__file__), "..", "vec0"),
            os.path.join(os.path.dirname(__file__), "vec0"),
        ]
        loaded = False
        for vec_path in vec_paths:
            try:
                conn.load_extension(vec_path)
                loaded = True
                break
            except sqlite3.OperationalError:
                continue

        if not loaded:
            print(f"Warning: sqlite-vec extension not found. Vector search will not work.")
            print("Install from: https://github.com/asg017/sqlite-vec")
    except Exception as e:
        print(f"Warning: Could not load extensions: {e}")

    return conn


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize database with schema."""
    conn = get_connection(db_path)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # Skip comment lines about loading extension
    statements = []
    for line in schema.split(";"):
        line = line.strip()
        if line and not line.startswith("--") and "vec0" not in line.lower():
            try:
                conn.execute(line)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    # Virtual tables might fail if sqlite-vec not loaded
                    if "vec0" in line.lower():
                        print(f"Skipping virtual table (sqlite-vec not loaded): {line[:60]}...")
                    else:
                        raise

    conn.commit()
    return conn


if __name__ == "__main__":
    print("Initializing Muninn database...")
    conn = init_db()
    print(f"Database created at: {get_db_path()}")
    conn.close()
