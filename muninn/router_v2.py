"""Semantic router v0.2 — Disco Elysium multi-activation with facets.

Architecture:
  - Each peer has N facets (emocional, fisico, social, contextual, tecnico)
  - Each facet has its own embedding
  - Input text is compared against ALL facets
  - Best facet per peer determines peer activation
  - Multiple peers can activate simultaneously
  - Top-K activations are returned with context for injection
  - Optional: bge-reranker-v2-m3 cross-encoder refines ranking
"""

import os
import sqlite3
import struct
from typing import List, Dict, Optional
from datetime import datetime

from .embeddings_v2 import embed, embed_batch, cosine_similarity, get_backend
from .db import get_connection

# Reranker singleton
_reranker = None


def get_reranker():
    """Load bge-reranker-v2-m3 cross-encoder (lazy, singleton)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print("  [Reranker] Cargando BAAI/bge-reranker-v2-m3...")
            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
            print("  [Reranker] Listo")
        except Exception as e:
            print(f"  [Reranker] No disponible: {e}")
            return None
    return _reranker


def route(
    text: str,
    db_path: str | None = None,
    top_k: int = 3,
    context_hour: int | None = None,
    instruction_override: str | None = None,
    use_reranker: bool = True,
) -> List[Dict]:
    """
    Multi-activation routing: evaluate text against all peer facets.

    Flow:
      1. Embed query → cosine similarity against all facet embeddings
      2. Filter by activation threshold
      3. If reranker available: rerank top-N candidates via cross-encoder
      4. Return top_k activations sorted by final score

    Returns list of dicts with activation details, sorted by total_score desc.
    """
    backend = get_backend()

    # Get instruction from DB config or use override
    instruction = instruction_override
    if not instruction:
        try:
            conn_temp = get_connection(db_path)
            row = conn_temp.execute(
                "SELECT value FROM embedding_config WHERE key = 'instruction'"
            ).fetchone()
            if row:
                instruction = row["value"]
            conn_temp.close()
        except Exception:
            pass

    # Embed the input text as a query (with instruction)
    text_embedding = embed(text, is_query=True, instruction=instruction)

    conn = get_connection(db_path)

    # Get all active peers
    peers = conn.execute("""
        SELECT p.id, p.name, p.type, p.domain, p.representation,
               p.activation_threshold, p.confidence, p.level, p.tags
        FROM peers p
        WHERE p.is_active = 1
    """).fetchall()

    if not peers:
        conn.close()
        return []

    peer_data = {p["id"]: dict(p) for p in peers}
    thresholds = {p["id"]: p["activation_threshold"] for p in peers}
    levels = {p["id"]: p["level"] for p in peers}

    # Get all facets for active peers
    facets = conn.execute("""
        SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, pf.weight
        FROM peer_facets pf
        WHERE pf.peer_id IN (%s)
    """ % ",".join(f"'{pid}'" for pid in peer_data.keys())).fetchall()

    if not facets:
        conn.close()
        return []

    # Get stored facet embeddings and compute similarity
    peer_best_scores = {}  # peer_id -> best activation

    for facet in facets:
        fid = facet["id"]
        pid = facet["peer_id"]

        # Get stored embedding
        row = conn.execute(
            "SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [fid]
        ).fetchone()

        if not row:
            continue

        # Unpack embedding bytes
        dims = backend.dimensions
        try:
            stored_vec = list(struct.unpack(f"{dims}f", row["embedding"]))
        except struct.error:
            continue

        similarity = cosine_similarity(text_embedding, stored_vec)

        # Bonus de nivel (peers más activos pesan más)
        level_bonus = (levels.get(pid, 1.0) - 1.0) * 0.05

        # Bonus de contexto (hora)
        context_bonus = 0.0
        if context_hour is not None:
            # Angel del atardecer: bonus si es tarde/noche
            if pid == "sombra_angel_atardecer" and 17 <= context_hour <= 23:
                context_bonus = 0.05
            # Casual: bonus si es fuera de horario laboral
            if pid == "casual_social" and (context_hour < 9 or context_hour > 21):
                context_bonus = 0.03

        total_score = similarity + level_bonus + context_bonus

        # Track best facet per peer
        if pid not in peer_best_scores or total_score > peer_best_scores[pid]["total_score"]:
            peer_best_scores[pid] = {
                "peer_id": pid,
                "facet_id": fid,
                "facet_type": facet["facet_type"],
                "facet_text": facet["text"],
                "similarity": similarity,
                "bonus_level": level_bonus,
                "bonus_context": context_bonus,
                "total_score": total_score,
            }

    conn.close()

    # Filter by threshold
    activated = [
        act for pid, act in peer_best_scores.items()
        if act["total_score"] >= thresholds.get(pid, 0.25)
    ]

    # Sort by embedding similarity desc
    activated.sort(key=lambda x: x["total_score"], reverse=True)

    # Rerank with cross-encoder if available and enough candidates
    if use_reranker and len(activated) > 1:
        reranker = get_reranker()
        if reranker is not None:
            # Rerank top candidates (max 15 to keep it fast)
            top_n = min(15, len(activated))
            pairs = [(text, act["facet_text"]) for act in activated[:top_n]]
            rerank_scores = reranker.predict(pairs)

            # Blend: use reranker score as the new total_score
            for i, score in enumerate(rerank_scores):
                activated[i]["rerank_score"] = float(score)
                activated[i]["total_score"] = float(score)

            # Re-sort by reranker score
            activated[:top_n] = sorted(
                activated[:top_n], key=lambda x: x["total_score"], reverse=True
            )

    # Limit to top_k
    activated = activated[:top_k]

    # Enrich with peer data
    for act in activated:
        p = peer_data.get(act["peer_id"], {})
        act["peer_name"] = p.get("name", "")
        act["peer_type"] = p.get("type", "")
        act["peer_domain"] = p.get("domain", "")
        act["representation"] = p.get("representation")
        act["confidence"] = p.get("confidence", 0.0)

    return activated


def route_with_context_injection(
    text: str,
    db_path: str | None = None,
    top_k: int = 3,
    context_hour: int | None = None,
) -> str:
    """
    Route and generate a context injection string from activated peers.

    This is the main entry point for the agent — it returns a formatted
    string to inject into the conversation context.
    """
    activations = route(text, db_path, top_k, context_hour)

    if not activations:
        return ""

    parts = []
    for act in activations:
        peer_line = f"[{act['peer_name']} ({act['peer_domain']})]"
        parts.append(peer_line)
        if act.get("representation"):
            parts.append(f"  Contexto: {act['representation']}")
        parts.append(f"  Activado por faceta {act['facet_type']} (score: {act['total_score']:.3f})")

    return "\n".join(parts)
