"""Semantic router — activates peers based on event embeddings."""

import sqlite3
from typing import List, Dict, Optional

from .embeddings import embed, cosine_similarity
from .db import get_connection


def route(
    text: str,
    db_path: str | None = None,
    top_k: int = 5,
    min_threshold: float = 0.0,
) -> List[Dict]:
    """
    Evaluate text against all peers and return activated ones.

    Returns list of dicts with: peer_id, name, type, similarity, representation, threshold
    """
    text_embedding = embed(text)

    conn = get_connection(db_path)

    # Get all active peers with their embeddings
    peers = conn.execute("""
        SELECT p.id, p.name, p.type, p.representation,
               p.activation_threshold, p.confidence, p.tags
        FROM peers p
        WHERE p.is_active = 1
    """).fetchall()

    # Get peer embeddings from sqlite-vec
    activated = []

    for peer in peers:
        peer_id = peer["id"]
        try:
            row = conn.execute(
                "SELECT embedding FROM peer_embeddings WHERE peer_id = ?",
                (peer_id,)
            ).fetchone()

            if row is None:
                continue

            peer_embedding = row["embedding"]
            # sqlite-vec returns raw bytes, need to convert
            import struct
            dims = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
            peer_vec = list(struct.unpack(f"{dims}f", peer_embedding))

            similarity = cosine_similarity(text_embedding, peer_vec)
            threshold = peer["activation_threshold"]

            if similarity >= threshold and similarity >= min_threshold:
                activated.append({
                    "peer_id": peer_id,
                    "name": peer["name"],
                    "type": peer["type"],
                    "similarity": round(similarity, 4),
                    "threshold": threshold,
                    "representation": peer["representation"],
                    "confidence": peer["confidence"],
                })
        except Exception:
            continue

    conn.close()

    # Sort by similarity (highest first), limit to top_k
    activated.sort(key=lambda x: x["similarity"], reverse=True)
    return activated[:top_k]
