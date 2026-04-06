"""Muninn API — FastAPI application."""

import json
import struct
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .db import get_connection, init_db
from .embeddings import embed
from .models import (
    PeerCreate, PeerUpdate, PeerResponse,
    MemoryCreate, MemoryUpdate, MemoryResponse,
    EventCreate, EventResponse,
    ConnectionCreate, ConnectionResponse,
    SearchRequest, SearchResult,
    SessionCreate, SessionResponse,
    MessageResponse,
)

app = FastAPI(
    title="Muninn",
    description="Memory system for AI agents — inspired by depth psychology",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────

def row_to_peer(row) -> dict:
    """Convert a DB row to a peer dict."""
    d = dict(row)
    d["tags"] = json.loads(d.get("tags", "[]"))
    return d


def row_to_memory(row) -> dict:
    """Convert a DB row to a memory dict."""
    d = dict(row)
    d["metadata"] = json.loads(d.get("metadata", "{}"))
    return d


def store_embedding(conn, table: str, id_col: str, id_val, text: str):
    """Generate embedding for text and store in sqlite-vec table."""
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    # Delete existing embedding if updating
    conn.execute(f"DELETE FROM {table} WHERE {id_col} = ?", [id_val])
    conn.execute(f"INSERT INTO {table} ({id_col}, embedding) VALUES (?, ?)", [id_val, vec_bytes])


# ══════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════

@app.get("/", response_model=dict)
async def root():
    return {
        "status": "ok",
        "name": "Muninn",
        "version": "0.1.0",
    }


@app.get("/stats", response_model=dict)
async def stats():
    conn = get_connection()
    try:
        total_peers = conn.execute("SELECT COUNT(*) FROM peers WHERE is_active=1").fetchone()[0]
        total_memories = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active=1").fetchone()[0]
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        total_activations = conn.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
        return {
            "total_peers": total_peers,
            "total_memories": total_memories,
            "total_events": total_events,
            "total_activations": total_activations,
        }
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
# PEERS
# ══════════════════════════════════════════════════════════

@app.post("/api/v1/peers", response_model=PeerResponse, status_code=201)
async def create_peer(body: PeerCreate):
    conn = get_connection()
    try:
        # Check if exists
        existing = conn.execute("SELECT id FROM peers WHERE id = ?", [body.id]).fetchone()
        if existing:
            raise HTTPException(409, f"Peer '{body.id}' already exists")

        conn.execute("""
            INSERT INTO peers (id, name, type, description, representation, confidence, activation_threshold, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            body.id, body.name, body.type, body.description,
            body.representation, body.confidence, body.activation_threshold,
            json.dumps(body.tags),
        ])

        # Auto-embed using embedding_text or fallback to description
        text_to_embed = body.embedding_text or body.description or body.name
        if text_to_embed:
            store_embedding(conn, "peer_embeddings", "peer_id", body.id, text_to_embed)

        conn.commit()

        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [body.id]).fetchone()
        return row_to_peer(peer)
    finally:
        conn.close()


@app.get("/api/v1/peers", response_model=list[PeerResponse])
async def list_peers(
    type: Optional[str] = Query(None),
    is_active: Optional[int] = Query(None),
):
    conn = get_connection()
    try:
        query = "SELECT * FROM peers WHERE 1=1"
        params = []
        if type:
            query += " AND type = ?"
            params.append(type)
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(is_active)
        query += " ORDER BY updated_at DESC"

        rows = conn.execute(query, params).fetchall()
        return [row_to_peer(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/v1/peers/{peer_id}", response_model=PeerResponse)
async def get_peer(peer_id: str):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")
        return row_to_peer(peer)
    finally:
        conn.close()


@app.put("/api/v1/peers/{peer_id}", response_model=PeerResponse)
async def update_peer(peer_id: str, body: PeerUpdate):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")

        # Build dynamic UPDATE
        updates = []
        params = []
        if body.name is not None:
            updates.append("name = ?")
            params.append(body.name)
        if body.description is not None:
            updates.append("description = ?")
            params.append(body.description)
        if body.representation is not None:
            updates.append("representation = ?")
            params.append(body.representation)
        if body.confidence is not None:
            updates.append("confidence = ?")
            params.append(body.confidence)
        if body.activation_threshold is not None:
            updates.append("activation_threshold = ?")
            params.append(body.activation_threshold)
        if body.tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(body.tags))

        if updates:
            updates.append("updated_at = datetime('now')")
            params.append(peer_id)
            conn.execute(f"UPDATE peers SET {', '.join(updates)} WHERE id = ?", params)

        # Re-embed if embedding_text provided
        if body.embedding_text:
            store_embedding(conn, "peer_embeddings", "peer_id", peer_id, body.embedding_text)

        conn.commit()

        updated = conn.execute("SELECT * FROM peers WHERE id = ?", [peer_id]).fetchone()
        return row_to_peer(updated)
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


@app.get("/api/v1/peers/{peer_id}/memories", response_model=list[MemoryResponse])
async def get_peer_memories(peer_id: str, limit: int = Query(default=20, ge=1, le=100)):
    conn = get_connection()
    try:
        peer = conn.execute("SELECT id FROM peers WHERE id = ?", [peer_id]).fetchone()
        if not peer:
            raise HTTPException(404, f"Peer '{peer_id}' not found")

        rows = conn.execute("""
            SELECT m.*, mp.relevance
            FROM memories m
            JOIN memory_peers mp ON m.id = mp.memory_id
            WHERE mp.peer_id = ? AND m.is_active = 1
            ORDER BY m.created_at DESC
            LIMIT ?
        """, [peer_id, limit]).fetchall()

        result = []
        for r in rows:
            d = row_to_memory(r)
            d["peers"] = [{"peer_id": peer_id, "relevance": r["relevance"]}]
            result.append(d)
        return result
    finally:
        conn.close()
