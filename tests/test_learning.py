"""Tests for the learning pipeline."""

import pytest
import sqlite3
from unittest.mock import patch
import numpy as np
import struct

from muninn.db import init_db
from muninn.learning import learn_fact, learn_batch, forget_facet, merge_facets


@pytest.fixture
def db_path():
    """Create a temporary database for testing."""
    import tempfile
    import os

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


@pytest.fixture
def mock_embed():
    """Mock embed to return fixed random vectors."""

    def mock_func(text, is_query=False, instruction=None):
        np.random.seed(hash(text) % 2**32)  # Deterministic seed from text
        return np.random.rand(1024).tolist()

    return mock_func


class TestLearning:
    """Test the learning pipeline."""

    @patch("muninn.learning.embed")
    @patch("muninn.learning.get_connection")
    def test_learn_fact_creates_peer(self, mock_conn, mock_embed, conn, db_path):
        """Learning a new fact should create a new peer."""
        mock_embed.return_value = [0.1] * 1024
        mock_conn.return_value = conn

        result = learn_fact("Diego started learning Rust", db_path=db_path)
        assert result["action"] == "created_peer"
        assert "peer_id" in result["details"]

        # Check peer was created
        peer = conn.execute(
            "SELECT * FROM peers WHERE id = ?", [result["details"]["peer_id"]]
        ).fetchone()
        assert peer is not None
        assert "Learned: Diego started learning Rust" in peer["name"]

    @patch("muninn.learning.embed")
    @patch("muninn.learning._find_similar_facets")
    @patch("muninn.learning.get_connection")
    def test_learn_fact_adds_facet(self, mock_conn, mock_find, mock_embed, conn):
        """Medium similarity should add facet to existing peer."""
        mock_embed.return_value = [0.1] * 1024
        mock_find.return_value = [
            {"facet_id": 1, "peer_id": "peer1", "similarity": 0.5}
        ]
        mock_conn.return_value = conn

        # First create the peer
        conn.execute(
            "INSERT INTO peers (id, name, type) VALUES ('peer1', 'Peer1', 'tema')"
        )

        result = learn_fact("New fact for existing peer", db_path=None)
        assert result["action"] == "added_facet"
        assert result["details"]["peer_id"] == "peer1"

    @patch("muninn.learning.get_connection")
    def test_learn_batch(self, mock_conn, conn):
        """Test batch learning."""
        mock_conn.return_value = conn
        with patch("muninn.learning.embed") as mock_embed:
            mock_embed.return_value = [0.1] * 1024
            facts = ["Fact 1", "Fact 2"]
            results = learn_batch(facts, db_path=None)
            assert len(results) == 2
            assert all(r["action"] == "created_peer" for r in results)

    @patch("muninn.learning.get_connection")
    def test_forget_facet(self, mock_conn, conn):
        """Test forgetting a facet."""
        mock_conn.return_value = conn
        # First create a facet
        conn.execute(
            "INSERT INTO peers (id, name, type) VALUES ('test_peer', 'Test', 'tema')"
        )
        conn.execute(
            "INSERT INTO peer_facets (peer_id, facet_type, text) VALUES ('test_peer', 'tecnico', 'test')"
        )
        facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
            [facet_id, b"\x00" * 4096],
        )  # Fake embedding

        success = forget_facet(facet_id, db_path=None)
        assert success

        # Check embedding removed
        emb = conn.execute(
            "SELECT * FROM facet_embeddings WHERE facet_id = ?", [facet_id]
        ).fetchone()
        assert emb is None

    @patch("muninn.learning.get_connection")
    def test_merge_facets(self, mock_conn, conn):
        """Test merging facets."""
        mock_conn.return_value = conn
        # Create two facets
        conn.execute(
            "INSERT INTO peers (id, name, type) VALUES ('test_peer', 'Test', 'tema')"
        )
        conn.execute(
            "INSERT INTO peer_facets (peer_id, facet_type, text) VALUES ('test_peer', 'tecnico', 'text1')"
        )
        fid1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO peer_facets (peer_id, facet_type, text) VALUES ('test_peer', 'tecnico', 'text2')"
        )
        fid2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        with patch("muninn.learning.embed") as mock_embed:
            mock_embed.return_value = [0.1] * 1024
            # Add embeddings
            vec_bytes = struct.pack(f"{len([0.1] * 1024)}f", *([0.1] * 1024))
            conn.execute(
                "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
                [fid1, vec_bytes],
            )
            conn.execute(
                "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
                [fid2, vec_bytes],
            )
            success = merge_facets(fid1, fid2, db_path=None)
            assert success

        # Check source embedding removed
        emb1 = conn.execute(
            "SELECT * FROM facet_embeddings WHERE facet_id = ?", [fid1]
        ).fetchone()
        assert emb1 is None

        # Check target updated
        facet = conn.execute(
            "SELECT text FROM peer_facets WHERE id = ?", [fid2]
        ).fetchone()
        assert "text1 | text2" in facet["text"]
