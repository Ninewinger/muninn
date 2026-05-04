"""Semantic router v0.4 - Gemini embedding + OpenRouter reranker.

Architecture:
  - Strategy A/B (faceted): compare query against individual facet embeddings
  - Strategy C (composite): build rich text per peer dynamically, embed once, compare
  - Strategy hybrid: blend faceted + composite scores for best of both worlds
  - Reranker: OpenRouter Cohere Rerank v3.5 (32K context, multilingual)
"""
import os
import sqlite3
import struct
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .embeddings_v2 import embed, embed_batch, cosine_similarity, get_backend
from .db import get_connection
from .context_bonus import compute_context_bonus, record_activation
from .reranker_openrouter import rerank as openrouter_rerank

# Composite embedding cache: {(db_path_hash, peer_id): (embedding, content_hash)}
_composite_cache: Dict[Tuple[str, str], Tuple[List[float], str]] = {}


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


def _build_composite_text(peer: dict, facets: List[dict]) -> str:
    """Build a single composite text for a peer from all its data."""
    parts = []

    # Name and domain
    name = peer.get("name", peer.get("id", ""))
    domain = peer.get("domain", "")
    if domain:
        parts.append(f"{name} ({domain})")
    else:
        parts.append(name)

    # Description
    desc = peer.get("description")
    if desc:
        parts.append(desc)

    # Representation (what gets injected when activated)
    rep = peer.get("representation")
    if rep:
        parts.append(f"Contexto activo: {rep}")

    # Tags
    tags = peer.get("tags", "[]")
    if tags and tags != "[]":
        parts.append(f"Etiquetas: {tags}")

    # All facet texts
    if facets:
        facet_parts = []
        for f in facets:
            ftype = f.get("facet_type", "")
            ftext = f.get("text", "")
            if ftext:
                facet_parts.append(f"{ftype}: {ftext}")
        if facet_parts:
            parts.append("Facetas: " + ". ".join(facet_parts))

    return ". ".join(parts)


def _get_composite_embeddings(
    conn: sqlite3.Connection,
    peer_data: Dict[str, dict],
    db_path: str,
    instruction: str | None = None,
) -> Dict[str, List[float]]:
    """Get or compute composite embeddings for all peers. Uses cache."""
    backend = get_backend()
    result = {}
    to_embed = {}

    for pid, peer in peer_data.items():
        # Get facets for this peer
        facets = conn.execute(
            "SELECT facet_type, text, weight FROM peer_facets WHERE peer_id = ?",
            [pid]
        ).fetchall()
        facets = [dict(f) for f in facets]

        composite_text = _build_composite_text(peer, facets)
        content_hash = hashlib.md5(composite_text.encode()).hexdigest()

        # Check cache
        cache_key = (db_path or ":memory:", pid)
        cached = _composite_cache.get(cache_key)
        if cached and cached[1] == content_hash:
            result[pid] = cached[0]
        else:
            to_embed[pid] = (composite_text, content_hash)

    # Embed uncached composites
    if to_embed:
        texts = [v[0] for v in to_embed.values()]
        pids = list(to_embed.keys())
        hashes = [v[1] for v in to_embed.values()]

        embeddings = embed_batch(texts, is_query=False, instruction=instruction)

        for pid, emb, chash in zip(pids, embeddings, hashes):
            cache_key = (db_path or ":memory:", pid)
            _composite_cache[cache_key] = (emb, chash)
            result[pid] = emb

    return result


def _route_faceted(
    text_embedding: List[float],
    conn: sqlite3.Connection,
    peer_data: Dict[str, dict],
    thresholds: Dict[str, float],
    levels: Dict[str, float],
    context_hour: int | None,
    backend,
) -> List[Dict]:
    """Strategy A/B: compare against individual facets (original logic)."""
    facets = conn.execute("""
        SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, pf.weight
        FROM peer_facets pf
        WHERE pf.peer_id IN (%s)
    """ % ",".join(f"'{pid}'" for pid in peer_data.keys())).fetchall()

    if not facets:
        return []

    peer_best_scores = {}

    for facet in facets:
        fid = facet["id"]
        pid = facet["peer_id"]

        row = conn.execute(
            "SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [fid]
        ).fetchone()
        if not row:
            continue

        dims = backend.dimensions
        try:
            stored_vec = list(struct.unpack(f"{dims}f", row["embedding"]))
        except struct.error:
            continue

        similarity = cosine_similarity(text_embedding, stored_vec)

        level_bonus = (levels.get(pid, 1.0) - 1.0) * 0.05
        context_bonus = _compute_context_bonus(pid, context_hour)

        total_score = similarity + level_bonus + context_bonus

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

    # Filter by threshold
    return [
        act for pid, act in peer_best_scores.items()
        if act["total_score"] >= thresholds.get(pid, 0.25)
    ]


def _route_composite(
    text_embedding: List[float],
    conn: sqlite3.Connection,
    peer_data: Dict[str, dict],
    thresholds: Dict[str, float],
    levels: Dict[str, float],
    context_hour: int | None,
    db_path: str,
    instruction: str | None = None,
) -> List[Dict]:
    """Strategy C: compare against composite peer embeddings."""
    composite_embs = _get_composite_embeddings(conn, peer_data, db_path, instruction)

    activated = []
    for pid, comp_emb in composite_embs.items():
        similarity = cosine_similarity(text_embedding, comp_emb)

        level_bonus = (levels.get(pid, 1.0) - 1.0) * 0.05
        context_bonus = _compute_context_bonus(pid, context_hour)
        total_score = similarity + level_bonus + context_bonus

        if total_score >= thresholds.get(pid, 0.25):
            activated.append({
                "peer_id": pid,
                "facet_type": "composite",
                "facet_text": _build_composite_text(
                    peer_data[pid],
                    [dict(f) for f in conn.execute(
                        "SELECT facet_type, text FROM peer_facets WHERE peer_id = ?", [pid]
                    ).fetchall()]
                )[:200],  # truncate for display
                "similarity": similarity,
                "bonus_level": level_bonus,
                "bonus_context": context_bonus,
                "total_score": total_score,
            })

    return activated


def _compute_context_bonus(peer_id: str, context_hour: int | None) -> float:
    """Compute context-based bonus — delegates to enhanced context_bonus module."""
    return compute_context_bonus(peer_id, context_hour)


def route(
    text: str,
    db_path: str | None = None,
    top_k: int = 3,
    context_hour: int | None = None,
    instruction_override: str | None = None,
    use_reranker: bool = True,
    strategy: str = "composite",
    alpha: float = 0.4,
) -> List[Dict]:
    """
    Multi-activation routing with configurable strategy.

    Strategies:
      - 'faceted': original facet-by-facet comparison (Strategy A/B)
      - 'composite': dynamic composite text per peer (Strategy C)
      - 'hybrid': blend of faceted + composite (alpha controls blend)

    Args:
        alpha: blend factor for hybrid (0.0 = pure composite, 1.0 = pure faceted)
    """
    # Read DB config early (model_name + instruction) before any backend calls
    instruction = instruction_override
    if not instruction or not os.getenv("EMBEDDING_MODEL"):
        try:
            conn_temp = get_connection(db_path)
            if not instruction:
                row = conn_temp.execute(
                    "SELECT value FROM embedding_config WHERE key = 'instruction'"
                ).fetchone()
                if row:
                    instruction = row["value"]
            if not os.getenv("EMBEDDING_MODEL"):
                model_row = conn_temp.execute(
                    "SELECT value FROM embedding_config WHERE key = 'model_name'"
                ).fetchone()
                if model_row:
                    os.environ["EMBEDDING_MODEL"] = model_row["value"]
            conn_temp.close()
        except Exception:
            pass

    backend = get_backend(db_path=db_path)

    # Embed the input text as a query (with instruction)
    text_embedding = embed(text, is_query=True, instruction=instruction)

    conn = get_connection(db_path)

    # Get all active peers
    peers = conn.execute("""
        SELECT p.id, p.name, p.type, p.domain, p.representation,
               p.activation_threshold, p.confidence, p.level, p.tags,
               p.description
        FROM peers p
        WHERE p.is_active = 1
    """).fetchall()

    if not peers:
        conn.close()
        return []

    peer_data = {p["id"]: dict(p) for p in peers}
    thresholds = {p["id"]: p["activation_threshold"] for p in peers}
    levels = {p["id"]: p["level"] for p in peers}

    if strategy == "faceted":
        activated = _route_faceted(
            text_embedding, conn, peer_data, thresholds, levels, context_hour, backend
        )

    elif strategy == "composite":
        activated = _route_composite(
            text_embedding, conn, peer_data, thresholds, levels, context_hour,
            db_path or "", instruction
        )

    elif strategy == "hybrid":
        faceted = _route_faceted(
            text_embedding, conn, peer_data, thresholds, levels, context_hour, backend
        )
        composite = _route_composite(
            text_embedding, conn, peer_data, thresholds, levels, context_hour,
            db_path or "", instruction
        )

        # Build score maps: peer_id -> total_score
        faceted_map = {a["peer_id"]: a for a in faceted}
        composite_map = {a["peer_id"]: a for a in composite}

        # All activated peer IDs from either strategy
        all_pids = set(faceted_map.keys()) | set(composite_map.keys())

        activated = []
        for pid in all_pids:
            f_score = faceted_map[pid]["total_score"] if pid in faceted_map else 0.0
            c_score = composite_map[pid]["total_score"] if pid in composite_map else 0.0
            blended = alpha * f_score + (1 - alpha) * c_score

            # Use the activation with higher individual score as base
            base = faceted_map.get(pid) or composite_map.get(pid)
            entry = dict(base)
            entry["total_score"] = blended
            entry["strategy"] = "hybrid"
            entry["faceted_score"] = f_score
            entry["composite_score"] = c_score
            activated.append(entry)

        # Re-filter by threshold (blended might dip below)
        activated = [a for a in activated if a["total_score"] >= thresholds.get(a["peer_id"], 0.25)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'faceted', 'composite', or 'hybrid'")

    # Sort by total_score desc
    activated.sort(key=lambda x: x["total_score"], reverse=True)

    # Rerank with OpenRouter Cohere reranker (32K context, multilingual)
    if use_reranker and len(activated) > 1:
        try:
            top_n = min(15, len(activated))
            rerank_docs = []
            for act in activated[:top_n]:
                peer = peer_data.get(act["peer_id"], {})
                repr_text = peer.get("representation", "")
                facet_text = act.get("facet_text", "")
                doc_text = (facet_text + " " + repr_text).strip()
                if not doc_text:
                    doc_text = act.get("peer_name", "") + " (" + act.get("peer_domain", "") + ")"
                rerank_docs.append({"peer_id": act["peer_id"], "text": doc_text})
            
            reranked = openrouter_rerank(text, rerank_docs, top_k=top_n)
            
            if reranked:
                reranked_map = {r["peer_id"]: r for r in reranked}
                for act in activated[:top_n]:
                    r = reranked_map.get(act["peer_id"])
                    if r and "rerank_score" in r:
                        act["rerank_score"] = r["rerank_score"]
                        act["similarity_before_rerank"] = act.get("similarity", 0.0)
                        act["total_score"] = r["rerank_score"]
                        act["rerank_model"] = r.get("rerank_model", "cohere/rerank-v3.5")
                
                activated[:top_n] = sorted(activated[:top_n], key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as e:
            print("  [Router] Reranker error: " + str(e))

        # Re-apply individual thresholds AFTER reranker re-scoring
        activated = [
            act for act in activated
            if act.get("total_score", 0) >= thresholds.get(act["peer_id"], 0.25)
        ]

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

    conn.close()
    return activated


def invalidate_composite_cache(db_path: str = None):
    """Clear composite embedding cache (call after DB changes)."""
    global _composite_cache
    if db_path:
        keys_to_remove = [k for k in _composite_cache if k[0] == db_path]
        for k in keys_to_remove:
            del _composite_cache[k]
    else:
        _composite_cache.clear()


def route_with_context_injection(
    text: str,
    db_path: str | None = None,
    top_k: int = 3,
    context_hour: int | None = None,
    strategy: str = "composite",
) -> str:
    """
    Route and generate a context injection string from activated peers.

    This is the main entry point for the agent — it returns a formatted
    string to inject into the conversation context.
    """
    activations = route(text, db_path, top_k, context_hour, strategy=strategy)

    if not activations:
        return ""

    parts = []
    for act in activations:
        peer_line = f"[{act['peer_name']} ({act['peer_domain']})]"
        parts.append(peer_line)
        if act.get("representation"):
            parts.append(f"  Contexto: {act['representation']}")
        strategy_tag = act.get("strategy", strategy)
        parts.append(f"  Activado por {act['facet_type']} ({strategy_tag}, score: {act['total_score']:.3f})")

    return "\n".join(parts)
