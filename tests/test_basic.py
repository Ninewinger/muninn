"""Tests for Muninn."""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path


@pytest.fixture
def db_path():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def conn(db_path):
    """Create a database connection with schema."""
    from muninn.db import init_db
    connection = init_db(db_path)
    yield connection
    connection.close()


class TestSchema:
    """Test that the database schema initializes correctly."""

    def test_creates_tables(self, conn):
        """All expected tables should exist."""
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]

        assert "peers" in table_names
        assert "memories" in table_names
        assert "connections" in table_names
        assert "sessions" in table_names
        assert "events" in table_names
        assert "activations" in table_names
        assert "consolidation_log" in table_names
        assert "memory_peers" in table_names

    def test_creates_fts(self, conn):
        """FTS5 virtual table should exist."""
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memory_fts%'"
        ).fetchall()
        assert len(tables) > 0


class TestPeers:
    """Test peer CRUD operations."""

    def test_create_peer(self, conn):
        """Should insert a peer correctly."""
        conn.execute("""
            INSERT INTO peers (id, name, type, description)
            VALUES ('test_peer', 'Test Peer', 'tema', 'A test peer')
        """)
        conn.commit()

        peer = conn.execute(
            "SELECT * FROM peers WHERE id = 'test_peer'"
        ).fetchone()

        assert peer is not None
        assert peer["name"] == "Test Peer"
        assert peer["type"] == "tema"
        assert peer["is_active"] == 1
        assert peer["confidence"] == 0.1

    def test_soft_delete_peer(self, conn):
        """Should soft delete a peer."""
        conn.execute("""
            INSERT INTO peers (id, name, type) VALUES ('del_peer', 'Delete Me', 'tema')
        """)
        conn.execute("UPDATE peers SET is_active = 0 WHERE id = 'del_peer'")
        conn.commit()

        peer = conn.execute(
            "SELECT * FROM peers WHERE id = 'del_peer'"
        ).fetchone()
        assert peer["is_active"] == 0


class TestMemories:
    """Test memory CRUD and lifecycle."""

    def test_create_memory(self, conn):
        """Should insert a memory correctly."""
        conn.execute("""
            INSERT INTO memories (content, type, source)
            VALUES ('Test memory content', 'hecho', 'manual')
        """)
        conn.commit()

        memory = conn.execute(
            "SELECT * FROM memories WHERE content = 'Test memory content'"
        ).fetchone()

        assert memory is not None
        assert memory["type"] == "hecho"
        assert memory["confidence"] == 0.5

    def test_link_memory_to_peer(self, conn):
        """Should link a memory to a peer."""
        conn.execute("""
            INSERT INTO peers (id, name, type) VALUES ('link_peer', 'Link Test', 'tema')
        """)
        conn.execute("""
            INSERT INTO memories (content, type) VALUES ('Linked memory', 'hecho')
        """)
        conn.execute("""
            INSERT INTO memory_peers (memory_id, peer_id, relevance)
            VALUES (1, 'link_peer', 0.8)
        """)
        conn.commit()

        link = conn.execute(
            "SELECT * FROM memory_peers WHERE peer_id = 'link_peer'"
        ).fetchone()

        assert link is not None
        assert link["relevance"] == 0.8


class TestConnections:
    """Test peer connections."""

    def test_create_connection(self, conn):
        """Should create a connection between peers."""
        conn.execute("""
            INSERT INTO peers (id, name, type) VALUES ('peer_a', 'Peer A', 'tema')
        """)
        conn.execute("""
            INSERT INTO peers (id, name, type) VALUES ('peer_b', 'Peer B', 'tema')
        """)
        conn.execute("""
            INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES ('peer_a', 'peer_b', 'conecta', 0.8, 'Test connection')
        """)
        conn.commit()

        conn_row = conn.execute(
            "SELECT * FROM connections WHERE from_peer_id = 'peer_a'"
        ).fetchone()

        assert conn_row is not None
        assert conn_row["strength"] == 0.8
        assert conn_row["relation_type"] == "conecta"
