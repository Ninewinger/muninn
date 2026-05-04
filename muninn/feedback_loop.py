"""Feedback Loop for Muninn — correction learning when routing is wrong.

When Hermes detects a routing error, it sends a correction.
Muninn learns from it by adjusting thresholds, boosting correct peers,
and penalizing incorrect ones.

Architecture:
  - POST /api/v1/feedback/correction: register a routing correction
  - GET /api/v1/feedback/stats: view accumulated corrections
"""
import json, struct, os, time
from typing import List, Dict, Optional
from datetime import datetime

from .db import get_connection
from .embeddings_v2 import embed, cosine_similarity, get_backend

# Correction weights (stored in memory, persisted via DB)
# peer_id -> cumulative correction score (-1.0 to +1.0)
_correction_cache: Dict[str, float] = {}

# Per-facet corrections: facet_id -> (correct_count, wrong_count, last_query)
_facet_corrections: Dict[int, Dict] = {}


def register_correction(
    peer_id: str,
    query: str,
    should_match: bool,
    confidence: float = 0.5,
    db_path: str = None,
) -> dict:
    """
    Register a routing correction.
    
    Args:
        peer_id: The peer that should/shouldn't have been activated
        query: The query text that caused the misrouting
        should_match: True if peer SHOULD have matched, False if shouldn't
        confidence: How sure we are (0.0-1.0)
    
    Returns:
        dict with correction result and adjusted threshold
    """
    conn = get_connection(db_path)
    
    # Update peer-level correction cache
    current = _correction_cache.get(peer_id, 0.0)
    delta = (1.0 if should_match else -1.0) * confidence * 0.1
    new_val = max(-1.0, min(1.0, current + delta))
    _correction_cache[peer_id] = new_val
    
    # Record in DB as an event
    conn.execute(
        """INSERT INTO events (content, type, channel, metadata)
           VALUES (?, 'feedback_correction', 'hermes_agent', ?)""",
        [f"Correction: {query} {'should' if should_match else 'should NOT'} match {peer_id}",
         json.dumps({"query": query, "should_match": should_match, "confidence": confidence, "peer_id": peer_id})]
    )
    
    # If correcting positive (this peer SHOULD match this query):
    # Find the facet(s) that should have activated and boost them
    if should_match:
        result = _boost_peer_for_query(conn, peer_id, query, db_path)
    else:
        # Peer shouldn't have matched — increase its threshold
        result = _penalize_peer_for_query(conn, peer_id, query)
    
    conn.commit()
    conn.close()
    
    # Also record in memory for Muninn's dreaming
    _record_as_memory(peer_id, query, should_match, db_path)
    
    result["peer_id"] = peer_id
    result["correction_score"] = round(new_val, 4)
    result["should_match"] = should_match
    return result


def _boost_peer_for_query(conn, peer_id: str, query: str, db_path: str) -> dict:
    """Boost a peer so it matches this query type better next time."""
    # 1. Get peer info
    peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
    if not peer:
        return {"action": "skipped", "reason": "peer not found"}
    
    # 2. Lower the activation threshold slightly
    current_threshold = peer["activation_threshold"]
    new_threshold = max(0.05, current_threshold - 0.02)
    conn.execute("UPDATE peers SET activation_threshold = ? WHERE id = ?",
                 [new_threshold, peer_id])
    
    # 3. Find if there's an existing facet that partially matches the query
    facets = conn.execute(
        "SELECT id, text FROM peer_facets WHERE peer_id = ?", [peer_id]
    ).fetchall()
    
    # 4. If no facet mentions keywords from query, add a new facet
    query_lower = query.lower()
    needs_new_facet = True
    for f in facets:
        text_lower = f["text"].lower() if f["text"] else ""
        q_words = [w for w in query_lower.split() if len(w) > 3]
        if q_words and any(w in text_lower for w in q_words):
            needs_new_facet = False
            # Track correction on this facet
            _facet_corrections[f["id"]] = _facet_corrections.get(f["id"], {"correct": 0, "wrong": 0})
            _facet_corrections[f["id"]]["correct"] += 1
            break
    
    if needs_new_facet:
        # Add the query as a new facet to this peer
        conn.execute(
            "INSERT INTO peer_facets (peer_id, facet_type, text) VALUES (?, 'keyword', ?)",
            [peer_id, query]
        )
        facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Embed the new facet
        vector = embed(query)
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        conn.execute(
            "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
            [facet_id, vec_bytes]
        )
        
        return {
            "action": "added_facet",
            "facet_id": facet_id,
            "new_threshold": new_threshold,
            "message": f"Added facet '{query[:50]}' to {peer_id}, threshold lowered to {new_threshold:.2f}"
        }
    else:
        return {
            "action": "lowered_threshold",
            "new_threshold": new_threshold,
            "message": f"Threshold for {peer_id} lowered to {new_threshold:.2f} (existing facets cover query)"
        }


def _penalize_peer_for_query(conn, peer_id: str, query: str) -> dict:
    """Penalize a peer that was wrongly activated."""
    peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
    if not peer:
        return {"action": "skipped", "reason": "peer not found"}
    
    # Raise threshold
    current_threshold = peer["activation_threshold"]
    new_threshold = min(0.95, current_threshold + 0.03)
    conn.execute("UPDATE peers SET activation_threshold = ? WHERE id = ?",
                 [new_threshold, peer_id])
    
    # Track which facets were involved
    facets = conn.execute(
        "SELECT id, text FROM peer_facets WHERE peer_id = ?", [peer_id]
    ).fetchall()
    query_lower = query.lower()
    for f in facets:
        text_lower = f["text"].lower() if f["text"] else ""
        q_words = [w for w in query_lower.split() if len(w) > 3]
        if q_words and any(w in text_lower for w in q_words):
            _facet_corrections[f["id"]] = _facet_corrections.get(f["id"], {"correct": 0, "wrong": 0})
            _facet_corrections[f["id"]]["wrong"] += 1
    
    return {
        "action": "raised_threshold",
        "new_threshold": new_threshold,
        "message": f"Threshold for {peer_id} raised to {new_threshold:.2f}"
    }


def _record_as_memory(peer_id: str, query: str, should_match: bool, db_path: str):
    """Also record the correction as a Muninn memory for dreaming."""
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO memories (content, type, source, confidence, metadata)
           VALUES (?, 'correction', 'feedback_loop', 0.8, ?)""",
        [f"Feedback: {query} -> {'match' if should_match else 'NO match'} {peer_id}",
         json.dumps({"peer_id": peer_id, "query": query, "should_match": should_match})]
    )
    memory_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    # Also link memory to the peer
    conn.execute(
        "INSERT INTO memory_peers (memory_id, peer_id, relevance) VALUES (?, ?, 1.0)",
        [memory_id, peer_id]
    )
    conn.commit()
    conn.close()


def get_feedback_stats(db_path: str = None) -> dict:
    """Get accumulated feedback statistics."""
    conn = get_connection(db_path)
    
    corrections = conn.execute(
        "SELECT COUNT(*) as c FROM events WHERE type = 'feedback_correction'"
    ).fetchone()[0]
    
    # Get threshold adjustments from peers table
    peers = conn.execute(
        "SELECT id, activation_threshold, confidence FROM peers WHERE is_active = 1"
    ).fetchall()
    
    threshold_summary = [
        {"peer_id": p["id"], "threshold": p["activation_threshold"],
         "correction_score": _correction_cache.get(p["id"], 0.0)}
        for p in peers
    ]
    
    conn.close()
    
    return {
        "total_corrections": corrections,
        "correction_cache_size": len(_correction_cache),
        "facet_corrections": {str(k): v for k, v in _facet_corrections.items()},
        "thresholds": threshold_summary,
    }


def add_feedback_endpoints(app):
    """Register feedback endpoints on a FastAPI app."""
    
    @app.post("/api/v1/feedback/correction", status_code=201)
    async def feedback_correction(body: dict):
        result = register_correction(
            peer_id=body["peer_id"],
            query=body["query"],
            should_match=body.get("should_match", True),
            confidence=body.get("confidence", 0.5),
        )
        return result
    
    @app.get("/api/v1/feedback/stats")
    async def feedback_stats():
        return get_feedback_stats()


# Test function
def test():
    """Quick test of the feedback loop."""
    import subprocess
    from .api import app
    add_feedback_endpoints(app)
    
    print("Feedback endpoints added:")
    print("  POST /api/v1/feedback/correction")
    print("  GET  /api/v1/feedback/stats")


if __name__ == "__main__":
    test()