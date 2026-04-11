"""Muninn API v0.2 — Disco Elysium architecture.

Endpoints:
  - Peers (CRUD with facets)
  - Facets (CRUD)
  - Route (semantic activation)
  - Memories (CRUD + search)
  - Events (create + auto-route)
  - Sessions
  - Connections
"""

import json
import os
import struct
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .db import get_connection, init_db, get_embedding_dims
from .embeddings_v2 import embed, embed_batch, get_backend
from .models_v2 import (
    PeerCreate, PeerUpdate, PeerResponse,
    FacetCreate, FacetResponse,
    MemoryCreate, MemoryUpdate, MemoryResponse,
    EventCreate, EventResponse,
    ConnectionCreate, ConnectionResponse,
    SearchRequest, SearchResult,
    SessionCreate, SessionResponse,
    ActivationResponse,
    RouteRequest, RouteResponse,
    MessageResponse,
)
from .router_v2 import route, route_with_context_injection

app = FastAPI(
    title="Muninn",
    description="Memory system for AI agents — Disco Elysium architecture",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────

def row_to_dict(row) -> dict:
    """Convert a DB row to a plain dict, parsing JSON fields."""
    d = dict(row)
    if "tags" in d and isinstance(d["tags"], str):
        d["tags"] = json.loads(d["tags"])
    if "metadata" in d and isinstance(d["metadata"], str):
        d["metadata"] = json.loads(d["metadata"])
    return d


def store_facet_embedding(conn, facet_id: int, text: str):
    """Generate embedding for facet text and store in sqlite-vec."""
    dims = get_embedding_dims(conn)
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM facet_embeddings WHERE facet_id = ?", [facet_id])
    conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)", [facet_id, vec_bytes])


def store_memory_embedding(conn, memory_id: int, text: str):
    """Generate embedding for memory text and store."""
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", [memory_id])
    conn.execute("INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)", [memory_id, vec_bytes])


def get_peer_facets(conn, peer_id: str) -> list:
    """Get all facets for a peer with embedding status."""
    facets = conn.execute("""
        SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, pf.weight
        FROM peer_facets pf
        WHERE pf.peer_id = ?
    """, [peer_id]).fetchall()

    result = []
    for f in facets:
        has_emb = conn.execute(
            "SELECT 1 FROM facet_embeddings WHERE facet_id = ?", [f["id"]]
        ).fetchone() is not None
        result.append(FacetResponse(
            id=f["id"],
            peer_id=f["peer_id"],
            facet_type=f["facet_type"],
            text=f["text"],
            weight=f["weight"],
            has_embedding=has_emb,
        ))
    return result


# ════════════════════════════════════════════════════════════
# HEALTH & STATS
# ════════════════════════════════════════════════════════════

@app.get("/", response_model=dict)
async def root():
    return {
        "status": "ok",
        "name": "Muninn",
        "version": "0.2.0",
        "architecture": "Disco Elysium",
    }


@app.get("/stats", response_model=dict)
async def stats():
    conn = get_connection()
    try:
        total_peers = conn.execute("SELECT COUNT(*) FROM peers WHERE is_active=1").fetchone()[0]
        total_facets = conn.execute("SELECT COUNT(*) FROM peer_facets").fetchone()[0]
        total_memories = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active=1").fetchone()[0]
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        total_activations = conn.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
        total_connections = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]

        # Embedding config
        configs = conn.execute("SELECT key, value FROM embedding_config").fetchall()
        config = {c["key"]: c["value"] for c in configs}

        return {
            "total_peers": total_peers,
            "total_facets": total_facets,
            "total_memories": total_memories,
            "total_events": total_events,
            "total_activations": total_activations,
            "total_connections": total_connections,
            "embedding_config": config,
        }
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# ROUTE — Semantic activation
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/route", response_model=RouteResponse)
async def route_text(body: RouteRequest):
    """Route text through semantic router, return activated peers with facets."""
    db_path = os.environ.get("DB_PATH", "./muninn.db")
    activations = route(
        body.text,
        db_path=db_path,
        top_k=body.top_k,
        context_hour=body.context_hour,
        instruction_override=body.instruction_override,
        use_reranker=True,
    )

    backend = get_backend()

    activation_responses = [
        ActivationResponse(
            peer_id=a["peer_id"],
            peer_name=a.get("peer_name", ""),
            peer_type=a.get("peer_type", ""),
            peer_domain=a.get("peer_domain"),
            facet_id=a["facet_id"],
            facet_type=a["facet_type"],
            similarity=round(a["similarity"], 4),
            bonus_level=round(a.get("bonus_level", 0), 4),
            bonus_context=round(a.get("bonus_context", 0), 4),
            total_score=round(a["total_score"], 4),
            rerank_score=round(a["rerank_score"], 4) if a.get("rerank_score") else None,
            representation=a.get("representation"),
            confidence=a.get("confidence", 0.0),
        )
        for a in activations
    ]

    return RouteResponse(
        query=body.text,
        activations=activation_responses,
        model_name=backend.model_name,
        dimensions=backend.dimensions,
    )


@app.post("/api/v1/route/inject", response_model=dict)
async def route_inject(body: RouteRequest):
    """Route text and return formatted context injection string."""
    db_path = os.environ.get("DB_PATH", "./muninn.db")
    injection = route_with_context_injection(
        body.text,
        db_path=db_path,
        top_k=body.top_k,
        context_hour=body.context_hour,
    )
    return {"query": body.text, "injection": injection}


# ════════════════════════════════════════════════════════════
# PEERS
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/peers", response_model=PeerResponse, status_code=201)
async def create_peer(body: PeerCreate):
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM peers WHERE id = ?", [body.id]).fetchone()
        if existing:
            raise HTTPException(409, f"Peer '{body.id}' already exists")

        conn.execute("""
            INSERT INTO peers (id, name, type, domain, description, representation,
                             confidence, activation_threshold, level, max_activations, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            body.id, body.name, body.type, body.domain, body.description,
            body.representation, body.confidence, body.activation_threshold,
            body.level, body.max_activations, json.dumps(body.tags),
        ])

        # Create initial facets if provided
        created_facets = []
        for facet_data in body.facets:
            conn.execute("""
                INSERT INTO peer_facets (peer_id, facet_type, text, weight)
                VALUES (?, ?, ?, ?)
            """, [body.id, facet_data.facet_type, facet_data.text, facet_data.weight])
            facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            store_facet_embedding(conn, facet_id, facet_data.text)
            created_facets.append(facet_id)

        conn.commit()

        # Fetch created peer
        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [body.id]).fetchone()
        d = row_to_dict(peer)
        d["facets"] = get_peer_facets(conn, body.id)
        return d
    finally:
        conn.close()


@app.get("/api/v1/peers", response_model=list[PeerResponse])
async def list_peers(
    type: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    is_active: Optional[int] = Query(None),
):
    conn = get_connection()
    try:
        query = "SELECT * FROM peers WHERE 1=1"
        params = []
        if type:
            query += " AND type = ?"
            params.append(type)
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(is_active)
        query += " ORDER BY updated_at DESC"

        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = row_to_dict(r)
            d["facets"] = get_peer_facets(conn, r["id"])
            result.append(d)
        return result
    finally:
        conn.close()


@app.get("/api/v1/peers/{peer_id}", response_model=PeerResponse)
async def get_peer(peer_id: str):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")
        d = row_to_dict(peer)
        d["facets"] = get_peer_facets(conn, peer_id)
        return d
    finally:
        conn.close()


@app.put("/api/v1/peers/{peer_id}", response_model=PeerResponse)
async def update_peer(peer_id: str, body: PeerUpdate):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")

        updates = []
        params = []
        for field in ["name", "domain", "description", "representation", "confidence",
                       "activation_threshold", "level", "max_activations"]:
            val = getattr(body, field, None)
            if val is not None:
                updates.append(f"{field} = ?")
                params.append(val)
        if body.tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(body.tags))

        if updates:
            updates.append("updated_at = datetime('now')")
            params.append(peer_id)
            conn.execute(f"UPDATE peers SET {', '.join(updates)} WHERE id = ?", params)

        conn.commit()

        updated = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        d = row_to_dict(updated)
        d["facets"] = get_peer_facets(conn, peer_id)
        return d
    finally:
        conn.close()


@app.delete("/api/v1/peers/{peer_id}", response_model=MessageResponse)
async def delete_peer(peer_id: str):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT id FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")

        conn.execute("UPDATE peers SET is_active = 0, updated_at = datetime('now') WHERE id = ?", [peer_id])
        conn.commit()
        return MessageResponse(message=f"Peer '{peer_id}' soft-deleted")
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# FACETS — CRUD for peer facets
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/peers/{peer_id}/facets", response_model=FacetResponse, status_code=201)
async def create_facet(peer_id: str, body: FacetCreate):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT id FROM peers WHERE id = ? AND is_active = 1", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")

        conn.execute("""
            INSERT INTO peer_facets (peer_id, facet_type, text, weight)
            VALUES (?, ?, ?, ?)
        """, [peer_id, body.facet_type, body.text, body.weight])
        facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Auto-embed
        store_facet_embedding(conn, facet_id, body.text)

        conn.commit()
        return FacetResponse(
            id=facet_id,
            peer_id=peer_id,
            facet_type=body.facet_type,
            text=body.text,
            weight=body.weight,
            has_embedding=True,
        )
    finally:
        conn.close()


@app.get("/api/v1/peers/{peer_id}/facets", response_model=list[FacetResponse])
async def list_facets(peer_id: str):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT id FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")
        return get_peer_facets(conn, peer_id)
    finally:
        conn.close()


@app.put("/api/v1/facets/{facet_id}", response_model=FacetResponse)
async def update_facet(facet_id: int, body: FacetCreate):
    conn = get_connection()
    try:
        facet = conn.execute("SELECT * FROM peer_facets WHERE id = ?", [facet_id]).fetchone()
        if not facet:
            raise HTTPException(404, f"Facet {facet_id} not found")

        conn.execute("""
            UPDATE peer_facets
            SET facet_type = ?, text = ?, weight = ?, updated_at = datetime('now')
            WHERE id = ?
        """, [body.facet_type, body.text, body.weight, facet_id])

        # Re-embed with new text
        store_facet_embedding(conn, facet_id, body.text)

        conn.commit()
        return FacetResponse(
            id=facet_id,
            peer_id=facet["peer_id"],
            facet_type=body.facet_type,
            text=body.text,
            weight=body.weight,
            has_embedding=True,
        )
    finally:
        conn.close()


@app.delete("/api/v1/facets/{facet_id}", response_model=MessageResponse)
async def delete_facet(facet_id: int):
    conn = get_connection()
    try:
        facet = conn.execute("SELECT id FROM peer_facets WHERE id = ?", [facet_id]).fetchone()
        if not facet:
            raise HTTPException(404, f"Facet {facet_id} not found")

        conn.execute("DELETE FROM facet_embeddings WHERE facet_id = ?", [facet_id])
        conn.execute("DELETE FROM peer_facets WHERE id = ?", [facet_id])
        conn.commit()
        return MessageResponse(message=f"Facet {facet_id} deleted")
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# MEMORIES
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(body: MemoryCreate):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO memories (content, type, source, confidence, occurred_at,
                                session_id, source_channel, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            body.content, body.type, body.source, body.confidence,
            body.occurred_at, body.session_id, body.source_channel,
            json.dumps(body.metadata),
        ])
        memory_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Auto-embed
        store_memory_embedding(conn, memory_id, body.content)

        # FTS index
        conn.execute("""
            INSERT INTO memory_fts (rowid, content, type, source)
            VALUES (?, ?, ?, ?)
        """, [memory_id, body.content, body.type, body.source])

        # Link to peers
        for peer_id in body.peer_ids:
            peer = conn.execute("SELECT id FROM peers WHERE id = ? AND is_active = 1", [peer_id]).fetchone()
            if peer:
                conn.execute("""
                    INSERT OR IGNORE INTO memory_peers (memory_id, peer_id, relevance)
                    VALUES (?, ?, 0.5)
                """, [memory_id, peer_id])

        conn.commit()

        memory = conn.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()
        d = row_to_dict(memory)
        linked = conn.execute("""
            SELECT peer_id FROM memory_peers WHERE memory_id = ?
        """, [memory_id]).fetchall()
        d["peers"] = [l["peer_id"] for l in linked]
        return d
    finally:
        conn.close()


@app.get("/api/v1/memories", response_model=list[MemoryResponse])
async def list_memories(
    peer_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
):
    conn = get_connection()
    try:
        if peer_id:
            rows = conn.execute("""
                SELECT m.* FROM memories m
                JOIN memory_peers mp ON m.id = mp.memory_id
                WHERE mp.peer_id = ? AND m.is_active = 1
                ORDER BY m.created_at DESC LIMIT ?
            """, [peer_id, limit]).fetchall()
        else:
            query = "SELECT * FROM memories WHERE is_active = 1"
            params = []
            if type:
                query += " AND type = ?"
                params.append(type)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

        result = []
        for r in rows:
            d = row_to_dict(r)
            linked = conn.execute("SELECT peer_id FROM memory_peers WHERE memory_id = ?", [r["id"]]).fetchall()
            d["peers"] = [l["peer_id"] for l in linked]
            result.append(d)
        return result
    finally:
        conn.close()


@app.get("/api/v1/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: int):
    conn = get_connection()
    try:
        memory = conn.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()
        if not memory:
            raise HTTPException(404, f"Memory {memory_id} not found")
        d = row_to_dict(memory)
        linked = conn.execute("SELECT peer_id FROM memory_peers WHERE memory_id = ?", [memory_id]).fetchall()
        d["peers"] = [l["peer_id"] for l in linked]
        return d
    finally:
        conn.close()


@app.put("/api/v1/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(memory_id: int, body: MemoryUpdate):
    conn = get_connection()
    try:
        memory = conn.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()
        if not memory:
            raise HTTPException(404, f"Memory {memory_id} not found")

        updates = []
        params = []
        if body.content is not None:
            updates.append("content = ?")
            params.append(body.content)
        if body.confidence is not None:
            updates.append("confidence = ?")
            params.append(body.confidence)
        if body.metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(body.metadata))

        if updates:
            updates.append("updated_at = datetime('now')")
            params.append(memory_id)
            conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)

            if body.content is not None:
                store_memory_embedding(conn, memory_id, body.content)
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", [memory_id])
                conn.execute("""
                    INSERT INTO memory_fts (rowid, content, type, source)
                    VALUES (?, ?, ?, ?)
                """, [memory_id, body.content, memory["type"], memory["source"]])

        conn.commit()

        updated = conn.execute("SELECT * FROM memories WHERE id = ?", [memory_id]).fetchone()
        d = row_to_dict(updated)
        linked = conn.execute("SELECT peer_id FROM memory_peers WHERE memory_id = ?", [memory_id]).fetchall()
        d["peers"] = [l["peer_id"] for l in linked]
        return d
    finally:
        conn.close()


@app.delete("/api/v1/memories/{memory_id}", response_model=MessageResponse)
async def delete_memory(memory_id: int):
    conn = get_connection()
    try:
        memory = conn.execute("SELECT id FROM memories WHERE id = ?", [memory_id]).fetchone()
        if not memory:
            raise HTTPException(404, f"Memory {memory_id} not found")

        conn.execute("UPDATE memories SET is_active = 0, updated_at = datetime('now') WHERE id = ?", [memory_id])
        conn.commit()
        return MessageResponse(message=f"Memory {memory_id} soft-deleted")
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# SEARCH — Semantic + FTS
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/search", response_model=list[SearchResult])
async def search_memories(body: SearchRequest):
    conn = get_connection()
    try:
        results = []

        if body.method in ("semantic", "hybrid"):
            query_emb = embed(body.query)
            dims = get_embedding_dims(conn)

            # Use sqlite-vec for top-k search
            try:
                vec_results = conn.execute("""
                    SELECT memory_id, distance
                    FROM memory_embeddings
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                """, [struct.pack(f"{dims}f", *query_emb), body.limit * 2]).fetchall()

                for vr in vec_results:
                    # distance from vec0 is cosine distance (1 - similarity)
                    sim = 1.0 - vr["distance"]
                    mem = conn.execute(
                        "SELECT id, content, type FROM memories WHERE id = ? AND is_active = 1",
                        [vr["memory_id"]]
                    ).fetchone()
                    if mem:
                        linked = conn.execute(
                            "SELECT peer_id FROM memory_peers WHERE memory_id = ?",
                            [mem["id"]]
                        ).fetchall()
                        results.append(SearchResult(
                            memory_id=mem["id"],
                            content=mem["content"],
                            type=mem["type"],
                            score=round(sim, 4),
                            peers=[l["peer_id"] for l in linked],
                        ))
            except Exception:
                pass  # sqlite-vec not available, fall through to FTS

        if body.method in ("fts", "hybrid") or not results:
            fts_results = conn.execute("""
                SELECT rowid, content, type, rank
                FROM memory_fts
                WHERE content MATCH ?
                ORDER BY rank
                LIMIT ?
            """, [body.query, body.limit]).fetchall()

            existing_ids = {r.memory_id for r in results}
            for fr in fts_results:
                if fr["rowid"] not in existing_ids:
                    # Check active
                    mem = conn.execute(
                        "SELECT id, content, type FROM memories WHERE id = ? AND is_active = 1",
                        [fr["rowid"]]
                    ).fetchone()
                    if mem:
                        linked = conn.execute(
                            "SELECT peer_id FROM memory_peers WHERE memory_id = ?",
                            [mem["id"]]
                        ).fetchall()
                        results.append(SearchResult(
                            memory_id=mem["id"],
                            content=mem["content"],
                            type=mem["type"],
                            score=round(-fr["rank"], 4),  # FTS rank is negative
                            peers=[l["peer_id"] for l in linked],
                        ))

        # Filter by peer if specified
        if body.peer_id:
            results = [r for r in results if body.peer_id in r.peers]

        return results[:body.limit]
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# EVENTS — Create + auto-route
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/events", response_model=EventResponse, status_code=201)
async def create_event(body: EventCreate):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO events (session_id, type, content, channel, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, [body.session_id, body.type, body.content, body.channel, json.dumps(body.metadata)])

        event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Auto-embed event
        vector = embed(body.content)
        dims = get_embedding_dims(conn)
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        conn.execute("INSERT INTO event_embeddings (event_id, embedding) VALUES (?, ?)", [event_id, vec_bytes])

        # Auto-route if user_message
        activations_list = []
        if body.type == "user_message":
            db_path = os.environ.get("DB_PATH", "./muninn.db")
            activated = route(body.content, db_path=db_path, use_reranker=True)

            for a in activated:
                # Record activation in DB
                conn.execute("""
                    INSERT INTO activations (event_id, peer_id, facet_id, similarity,
                                            bonus_level, bonus_context, total_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    event_id, a["peer_id"], a["facet_id"],
                    a["similarity"], a.get("bonus_level", 0), a.get("bonus_context", 0),
                    a["total_score"],
                ])

                # Increment peer activation count
                conn.execute("""
                    UPDATE peers
                    SET activation_count = activation_count + 1,
                        last_activated_at = datetime('now'),
                        updated_at = datetime('now')
                    WHERE id = ?
                """, [a["peer_id"]])

                activations_list.append(ActivationResponse(
                    peer_id=a["peer_id"],
                    peer_name=a.get("peer_name", ""),
                    peer_type=a.get("peer_type", ""),
                    peer_domain=a.get("peer_domain"),
                    facet_id=a["facet_id"],
                    facet_type=a["facet_type"],
                    similarity=round(a["similarity"], 4),
                    bonus_level=round(a.get("bonus_level", 0), 4),
                    bonus_context=round(a.get("bonus_context", 0), 4),
                    total_score=round(a["total_score"], 4),
                    representation=a.get("representation"),
                    confidence=a.get("confidence", 0.0),
                ))

        conn.commit()

        return EventResponse(event_id=event_id, activations=activations_list)
    finally:
        conn.close()


@app.get("/api/v1/events", response_model=list[dict])
async def list_events(
    session_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(default=50, ge=1, le=200),
):
    conn = get_connection()
    try:
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# SESSIONS
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/sessions", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreate):
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM sessions WHERE id = ?", [body.id]).fetchone()
        if existing:
            raise HTTPException(409, f"Session '{body.id}' already exists")

        conn.execute("""
            INSERT INTO sessions (id, channel, chat_id, metadata)
            VALUES (?, ?, ?, ?)
        """, [body.id, body.channel, body.chat_id, json.dumps(body.metadata)])
        conn.commit()

        session = conn.execute("SELECT * FROM sessions WHERE id = ?", [body.id]).fetchone()
        d = row_to_dict(session)
        d["event_count"] = 0
        return d
    finally:
        conn.close()


@app.get("/api/v1/sessions", response_model=list[SessionResponse])
async def list_sessions(limit: int = Query(default=20, ge=1, le=100)):
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.*, COUNT(e.id) as event_count
            FROM sessions s
            LEFT JOIN events e ON s.id = e.session_id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
        """, [limit]).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    conn = get_connection()
    try:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", [session_id]).fetchone()
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found")
        d = row_to_dict(session)
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        d["event_count"] = count
        return d
    finally:
        conn.close()


@app.put("/api/v1/sessions/{session_id}/close", response_model=SessionResponse)
async def close_session(session_id: str):
    conn = get_connection()
    try:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", [session_id]).fetchone()
        if not session:
            raise HTTPException(404, f"Session '{session_id}' not found")

        conn.execute("""
            UPDATE sessions SET ended_at = datetime('now') WHERE id = ?
        """, [session_id])
        conn.commit()

        updated = conn.execute("SELECT * FROM sessions WHERE id = ?", [session_id]).fetchone()
        d = row_to_dict(updated)
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        d["event_count"] = count
        return d
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════
# CONNECTIONS
# ════════════════════════════════════════════════════════════

@app.post("/api/v1/connections", response_model=ConnectionResponse, status_code=201)
async def create_connection(body: ConnectionCreate):
    conn = get_connection()
    try:
        # Verify peers exist
        for pid in [body.from_peer_id, body.to_peer_id]:
            if not conn.execute("SELECT id FROM peers WHERE id = ?", [pid]).fetchone():
                raise HTTPException(404, f"Peer '{pid}' not found")

        try:
            conn.execute("""
                INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
                VALUES (?, ?, ?, ?, ?)
            """, [body.from_peer_id, body.to_peer_id, body.relation_type, body.strength, body.description])
        except Exception:
            raise HTTPException(409, "Connection already exists")

        conn.commit()
        c = conn.execute("""
            SELECT * FROM connections
            WHERE from_peer_id = ? AND to_peer_id = ? AND relation_type = ?
        """, [body.from_peer_id, body.to_peer_id, body.relation_type]).fetchone()
        return row_to_dict(c)
    finally:
        conn.close()


@app.get("/api/v1/connections", response_model=list[ConnectionResponse])
async def list_connections(
    peer_id: Optional[str] = Query(None),
    relation_type: Optional[str] = Query(None),
):
    conn = get_connection()
    try:
        query = "SELECT * FROM connections WHERE 1=1"
        params = []
        if peer_id:
            query += " AND (from_peer_id = ? OR to_peer_id = ?)"
            params.extend([peer_id, peer_id])
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
        query += " ORDER BY strength DESC"

        rows = conn.execute(query, params).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


@app.delete("/api/v1/connections/{connection_id}", response_model=MessageResponse)
async def delete_connection(connection_id: int):
    conn = get_connection()
    try:
        c = conn.execute("SELECT id FROM connections WHERE id = ?", [connection_id]).fetchone()
        if not c:
            raise HTTPException(404, f"Connection {connection_id} not found")

        conn.execute("DELETE FROM connections WHERE id = ?", [connection_id])
        conn.commit()
        return MessageResponse(message=f"Connection {connection_id} deleted")
    finally:
        conn.close()
