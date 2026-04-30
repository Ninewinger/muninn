"""Learning pipeline for Muninn — enable the system to learn new information from conversations.

This module provides functions to learn new facts, creating or updating peers and facets
based on similarity thresholds and embedding comparisons.
"""

import json
import struct
import uuid
from typing import List, Dict, Optional

from .db import get_connection
from .embeddings_v2 import embed, cosine_similarity
from .router_v2 import route


def learn_fact(fact: str, db_path: str | None = None) -> dict:
    """
    Learn a new fact by comparing it against existing knowledge.

    Decision logic:
    1. Embed the fact
    2. Compare similarity against all facet embeddings
    3. If sim > 0.85: update/merge existing facet
    4. If sim > 0.15 with peer's best facet: add new facet to that peer
    5. Else: create new peer with single facet

    Args:
        fact: The fact to learn (string)
        db_path: Path to database

    Returns:
        dict: Learning result with action taken
    """
    conn = get_connection(db_path)

    # Embed the fact
    fact_embedding = embed(fact)

    # Find similar facets
    similar_facets = _find_similar_facets(fact_embedding, db_path)

    # Group by peer, find best similarity per peer
    peer_best_sims = {}
    facet_details = {}
    for facet in similar_facets:
        pid = facet["peer_id"]
        sim = facet["similarity"]
        if pid not in peer_best_sims or sim > peer_best_sims[pid]:
            peer_best_sims[pid] = sim
            facet_details[pid] = facet

    # Decision logic
    if similar_facets and similar_facets[0]["similarity"] > 0.85:
        # Update/merge the most similar facet
        facet = similar_facets[0]
        success = _update_facet(facet["facet_id"], fact, db_path)
        action = "updated_facet"
        details = {
            "facet_id": facet["facet_id"],
            "peer_id": facet["peer_id"],
            "similarity": facet["similarity"],
        }
    elif peer_best_sims and max(peer_best_sims.values()) > 0.15:
        # Add facet to the peer with highest best similarity
        best_peer = max(peer_best_sims, key=peer_best_sims.get)
        facet_id = _add_facet_to_peer(best_peer, fact, db_path)
        action = "added_facet"
        details = {
            "peer_id": best_peer,
            "facet_id": facet_id,
            "similarity": peer_best_sims[best_peer],
        }
    else:
        # Create new peer
        peer_id = _create_peer(fact, db_path)
        action = "created_peer"
        details = {"peer_id": peer_id}

    conn.commit()
    conn.close()

    return {"action": action, "fact": fact, "details": details}


def learn_batch(facts: List[str], db_path: str | None = None) -> List[dict]:
    """Learn multiple facts."""
    return [learn_fact(fact, db_path) for fact in facts]


def forget_facet(facet_id: int, db_path: str | None = None) -> bool:
    """Remove a facet (soft delete by deleting embedding, facet remains)."""
    conn = get_connection(db_path)

    # Delete embedding
    conn.execute("DELETE FROM facet_embeddings WHERE facet_id = ?", [facet_id])

    # Optionally mark facet as inactive, but schema doesn't have is_active for facets
    # For now, just remove embedding so it doesn't activate

    conn.commit()
    conn.close()
    return True


def merge_facets(
    source_facet_id: int, target_facet_id: int, db_path: str | None = None
) -> bool:
    """Merge source facet into target facet."""
    conn = get_connection(db_path)

    # Get both facets
    source = conn.execute(
        "SELECT text FROM peer_facets WHERE id = ?", [source_facet_id]
    ).fetchone()
    target = conn.execute(
        "SELECT text FROM peer_facets WHERE id = ?", [target_facet_id]
    ).fetchone()

    if not source or not target:
        conn.close()
        return False

    # Merge texts
    merged_text = f"{target['text']} | {source['text']}"

    # Update target
    _update_facet(target_facet_id, merged_text, db_path)

    # Remove source
    forget_facet(source_facet_id, db_path)

    conn.close()
    return True


# Internal helper functions


def _find_similar_facets(
    fact_embedding: List[float], db_path: str | None = None
) -> List[Dict]:
    """Find facets similar to the fact embedding."""
    conn = get_connection(db_path)

    # Get all facets with their embeddings
    facets = conn.execute("""
        SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, fe.embedding
        FROM peer_facets pf
        JOIN facet_embeddings fe ON pf.id = fe.facet_id
    """).fetchall()

    similar = []
    for facet in facets:
        # Unpack embedding
        try:
            vec_bytes = facet["embedding"]
            dims = len(fact_embedding)  # Assume same dims
            stored_vec = list(struct.unpack(f"{dims}f", vec_bytes))
            sim = cosine_similarity(fact_embedding, stored_vec)
            similar.append(
                {
                    "facet_id": facet["id"],
                    "peer_id": facet["peer_id"],
                    "facet_type": facet["facet_type"],
                    "facet_text": facet["text"],
                    "similarity": sim,
                }
            )
        except struct.error:
            continue

    conn.close()

    # Sort by similarity desc
    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar


def _create_peer(fact: str, db_path: str | None = None) -> str:
    """Create a new peer with a single facet."""
    conn = get_connection(db_path)

    peer_id = f"peer_{uuid.uuid4().hex[:8]}"
    name = f"Learned: {fact[:50]}{'...' if len(fact) > 50 else ''}"

    # Insert peer
    conn.execute(
        """
        INSERT INTO peers (id, name, type, domain, description)
        VALUES (?, ?, 'tema', 'learning', ?)
    """,
        [peer_id, name, fact],
    )

    # Add facet
    facet_id = _add_facet_to_peer(peer_id, fact, db_path)

    conn.close()
    return peer_id


def _add_facet_to_peer(peer_id: str, fact: str, db_path: str | None = None) -> int:
    """Add a new facet to an existing peer."""
    conn = get_connection(db_path)

    # Insert new facet
    conn.execute(
        """
        INSERT INTO peer_facets (peer_id, facet_type, text)
        VALUES (?, 'tecnico', ?)
    """,
        [peer_id, fact],
    )

    facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Embed
    vector = embed(fact)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute(
        "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
        [facet_id, vec_bytes],
    )

    conn.close()
    return facet_id


def _update_facet(facet_id: int, new_text: str, db_path: str | None = None) -> bool:
    """Update facet text and re-embed."""
    conn = get_connection(db_path)

    # Get current facet
    facet = conn.execute(
        "SELECT text FROM peer_facets WHERE id = ?", [facet_id]
    ).fetchone()
    if not facet:
        conn.close()
        return False

    # Merge texts: append new fact to existing
    updated_text = f"{facet['text']} | {new_text}"

    # Update facet text
    conn.execute(
        "UPDATE peer_facets SET text = ?, updated_at = datetime('now') WHERE id = ?",
        [updated_text, facet_id],
    )

    # Re-embed
    vector = embed(updated_text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM facet_embeddings WHERE facet_id = ?", [facet_id])
    conn.execute(
        "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
        [facet_id, vec_bytes],
    )

    conn.close()
    return True
