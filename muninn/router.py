"""Semantic router — activates peers based on event embeddings."""

import os
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

    # Get all active peers
    peers = conn.execute("""
        SELECT p.id, p.name, p.type, p.representation,
               p.activation_threshold, p.confidence, p.tags
        FROM peers p
        WHERE p.is_active = 1
    """).fetchall()

    if not peers:
        conn.close()
        return []

    peer_ids = [p["id"] for p in peers]
    thresholds = {p["id"]: p["activation_threshold"] for p in peers}
    peer_data = {p["id"]: dict(p) for p in peers}

    # Use sqlite-vec knn search
    import struct
    dims = 384
    query_bytes = struct.pack(f"{dims}f", *text_embedding)

    results = conn.execute("""
        SELECT peer_id, distance
        FROM peer_embeddings
        WHERE embedding MATCH ? AND k = ?
    """, [query_bytes, max(top_k * 2, len(peer_ids))]).fetchall()

    activated = []
    for row in results:
        peer_id = row["peer_id"]
        # sqlite-vec distance = cosine distance (1 - similarity for normalized vectors)
        similarity = 1.0 - row["distance"]
        threshold = thresholds.get(peer_id, 0.65)

        if similarity >= threshold and similarity >= min_threshold:
            p = peer_data[peer_id]
            activated.append({
                "peer_id": peer_id,
                "name": p["name"],
                "type": p["type"],
                "similarity": round(similarity, 4),
                "threshold": threshold,
                "representation": p["representation"],
                "confidence": p["confidence"],
            })

    conn.close()

    # Sort by similarity (highest first), limit to top_k
    activated.sort(key=lambda x: x["similarity"], reverse=True)
    return activated[:top_k]
