"""Dreaming/Consolidation — El proceso onírico de Muninn v0.2.

Adaptado a arquitectura Disco Elysium (facetas).
Sin LLM externo. Usa embeddings + reglas para:
1. Clasificar eventos por relevancia para cada peer (via facetas)
2. Crear memorias vinculadas a peers
3. Descubrir conexiones entre peers co-activados
4. Actualizar representaciones de peers
5. Olvidar memorias irrelevantes (decay)

Inspiración: Honcho Dreaming + KAIROS autoDream + Jung (función onírica)
"""

import json
import os
import struct
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from .db import get_connection, get_embedding_dims
from .embeddings_v2 import embed, embed_batch
from .router_v2 import route


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

# Relevancia mínima para crear memoria permanente
MEMORY_THRESHOLD_HIGH = 0.50  # alta → memoria permanente
MEMORY_THRESHOLD_LOW = 0.35  # media → memoria temporal (baja confianza)
MEMORY_THRESHOLD_FORGET = 0.25  # bajo → no crear memoria

# Decay: memorias con confianza < esto después de N días se olvidan
FORGET_CONFIDENCE = 0.2
FORGET_AFTER_DAYS = 30

# Co-activación: si 2 peers se activan juntos > N veces, sugerir conexión
COACTIVATION_THRESHOLD = 3

# Actualización de representación: cada N activaciones
REPRESENTATION_UPDATE_INTERVAL = 10


# ══════════════════════════════════════════════════════════════
# DREAMING: PROCESO PRINCIPAL
# ══════════════════════════════════════════════════════════════


def dream(
    db_path: Optional[str] = None,
    session_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the dreaming process.

    1. Take unprocessed events
    2. Route each through semantic router (facetas)
    3. Create memories for relevant events
    4. Find co-activation patterns
    5. Update peer representations
    6. Forget old low-confidence memories

    Returns consolidation stats.
    """
    conn = get_connection(db_path)

    # Start consolidation log
    log_id = _start_log(conn, session_id)

    stats = {
        "events_processed": 0,
        "memories_created": 0,
        "memories_skipped": 0,
        "connections_found": 0,
        "peers_updated": 0,
        "memories_forgotten": 0,
        "errors": [],
    }

    try:
        # ── PASO 1: Obtener eventos no procesados ─────────────
        events = _get_unprocessed_events(conn, session_id)
        stats["events_processed"] = len(events)

        if not events:
            _finish_log(conn, log_id, "completed", stats, "No events to process")
            conn.commit()
            conn.close()
            return stats

        # ── PASO 2: Procesar cada evento ──────────────────────
        for event in events:
            try:
                result = _process_event(conn, event, db_path, dry_run)
                if result["action"] == "memory_created":
                    stats["memories_created"] += 1
                elif result["action"] == "skipped":
                    stats["memories_skipped"] += 1
            except Exception as e:
                stats["errors"].append(f"Event {event['id']}: {str(e)}")

        # ── PASO 3: Descubrir conexiones por co-activación ────
        if not dry_run:
            new_connections = _discover_connections(conn)
            stats["connections_found"] = len(new_connections)

        # ── PASO 4: Actualizar representaciones de peers ──────
        if not dry_run:
            updated = _update_representations(conn, db_path)
            stats["peers_updated"] = len(updated)

        # ── PASO 4.5: Merge similar facets ────────────────────
        if not dry_run:
            merged = _merge_similar_facets(conn, db_path)
            stats["facets_merged"] = merged

        # ── PASO 5: Olvido (decay) ────────────────────────────
        if not dry_run:
            forgotten = _forget_memories(conn)
            stats["memories_forgotten"] = forgotten

        _finish_log(conn, log_id, "completed", stats)
        conn.commit()

    except Exception as e:
        stats["errors"].append(f"Fatal: {str(e)}")
        try:
            _finish_log(conn, log_id, "failed", stats, str(e))
            conn.commit()
        except:
            pass

    conn.close()
    return stats


# ══════════════════════════════════════════════════════════════
# PASO 1: Eventos no procesados
# ══════════════════════════════════════════════════════════════


def _get_unprocessed_events(conn, session_id=None):
    """
    Get events that haven't been processed into memories yet.
    An event is unprocessed if it has no corresponding memory
    with source='dreaming' and metadata.event_id matching.
    """
    query = """
        SELECT e.* FROM events e
        WHERE (e.type = 'user_message' OR e.type = 'conversation_turn')
        AND e.id NOT IN (
            SELECT DISTINCT json_extract(m.metadata, '$.event_id')
            FROM memories m
            WHERE m.source = 'dreaming'
            AND json_extract(m.metadata, '$.event_id') IS NOT NULL
        )
    """
    params = []

    if session_id:
        query += " AND e.session_id = ?"
        params.append(session_id)

    query += " ORDER BY e.created_at ASC LIMIT 200"

    return conn.execute(query, params).fetchall()


# ══════════════════════════════════════════════════════════════
# PASO 2: Procesar evento individual
# ══════════════════════════════════════════════════════════════


def _process_event(conn, event, db_path=None, dry_run=False):
    """
    Process a single event:
    - Route through semantic router (facetas) to find activated peers
    - Classify relevance (high/medium/low)
    - Create memory if relevant
    - Record activations (with facet info)
    """
    content = event["content"]
    event_id = event["id"]

    # Route event to find activated peers (via facetas)
    activated = route(content, db_path=db_path, use_reranker=True)

    if not activated:
        return {"action": "skipped", "reason": "no peers activated"}

    # Get the best activation (highest total_score)
    best = activated[0]
    best_score = best["total_score"]

    # Classify relevance
    if best_score >= MEMORY_THRESHOLD_HIGH:
        memory_type = "episodio"
        confidence = min(0.9, best_score)
        relevance = "high"
    elif best_score >= MEMORY_THRESHOLD_LOW:
        memory_type = "episodio"
        confidence = 0.3
        relevance = "medium"
    elif best_score >= MEMORY_THRESHOLD_FORGET:
        # Record activation but don't create memory
        _record_activations(conn, event_id, activated)
        return {"action": "skipped", "reason": f"low relevance ({best_score:.3f})"}
    else:
        return {"action": "skipped", "reason": "below threshold"}

    if dry_run:
        return {
            "action": "would_create_memory",
            "relevance": relevance,
            "peers": [a["peer_id"] for a in activated],
            "best_score": best_score,
        }

    # Record activations
    _record_activations(conn, event_id, activated)

    # Create memory
    peer_ids = [a["peer_id"] for a in activated]
    metadata = {
        "event_id": event_id,
        "relevance": relevance,
        "best_score": round(best_score, 4),
        "activated_peers": peer_ids,
        "activated_facets": [
            {
                "peer_id": a["peer_id"],
                "facet_id": a.get("facet_id"),
                "facet_type": a.get("facet_type"),
            }
            for a in activated
        ],
        "session_id": event["session_id"],
    }

    conn.execute(
        """
        INSERT INTO memories (content, type, source, confidence, occurred_at, session_id, source_channel, metadata)
        VALUES (?, ?, 'dreaming', ?, ?, ?, ?, ?)
    """,
        [
            content,
            memory_type,
            confidence,
            event["created_at"],
            event["session_id"],
            event["channel"],
            json.dumps(metadata),
        ],
    )

    memory_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Embed the memory
    _store_memory_embedding(conn, memory_id, content)

    # Index in FTS
    conn.execute(
        """
        INSERT INTO memory_fts (rowid, content, type, source)
        VALUES (?, ?, ?, ?)
    """,
        [memory_id, content, memory_type, "dreaming"],
    )

    # Link memory to activated peers
    for a in activated:
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_peers (memory_id, peer_id, relevance)
            VALUES (?, ?, ?)
        """,
            [memory_id, a["peer_id"], a["similarity"]],
        )

    conn.commit()

    return {
        "action": "memory_created",
        "memory_id": memory_id,
        "relevance": relevance,
        "peers": peer_ids,
        "best_score": best_score,
    }


def _record_activations(conn, event_id, activated):
    """Record which peers+facets activated for an event."""
    for a in activated:
        conn.execute(
            """
            INSERT INTO activations (event_id, peer_id, facet_id, similarity,
                                    bonus_level, bonus_context, total_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                event_id,
                a["peer_id"],
                a.get("facet_id"),  # may be None in composite/hybrid
                a["similarity"],
                a.get("bonus_level", 0.0),
                a.get("bonus_context", 0.0),
                a["total_score"],
            ],
        )

        # Increment activation count
        conn.execute(
            """
            UPDATE peers
            SET activation_count = activation_count + 1,
                last_activated_at = datetime('now'),
                updated_at = datetime('now')
            WHERE id = ?
        """,
            [a["peer_id"]],
        )


def _store_memory_embedding(conn, memory_id, text):
    """Generate and store embedding for a memory."""
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", [memory_id])
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
        [memory_id, vec_bytes],
    )


# ══════════════════════════════════════════════════════════════
# PASO 3: Descubrir conexiones por co-activación
# ══════════════════════════════════════════════════════════════


def _discover_connections(conn):
    """
    Find peers that co-activate frequently.
    If peer A and peer B activate together > COACTIVATION_THRESHOLD times,
    create a 'conecta' connection.
    """
    pairs = conn.execute(
        """
        SELECT
            a1.peer_id AS peer_a,
            a2.peer_id AS peer_b,
            COUNT(*) AS coactivation_count
        FROM activations a1
        JOIN activations a2 ON a1.event_id = a2.event_id AND a1.peer_id < a2.peer_id
        GROUP BY a1.peer_id, a2.peer_id
        HAVING COUNT(*) >= ?
    """,
        [COACTIVATION_THRESHOLD],
    ).fetchall()

    new_connections = []
    for pair in pairs:
        existing = conn.execute(
            """
            SELECT id FROM connections
            WHERE (from_peer_id = ? AND to_peer_id = ?)
               OR (from_peer_id = ? AND to_peer_id = ?)
        """,
            [pair["peer_a"], pair["peer_b"], pair["peer_b"], pair["peer_a"]],
        ).fetchone()

        if not existing:
            strength = min(1.0, pair["coactivation_count"] / 20.0)
            conn.execute(
                """
                INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
                VALUES (?, ?, 'conecta', ?, ?)
            """,
                [
                    pair["peer_a"],
                    pair["peer_b"],
                    strength,
                    f"Co-activadas {pair['coactivation_count']} veces en los mismos eventos",
                ],
            )
            new_connections.append(
                {
                    "from": pair["peer_a"],
                    "to": pair["peer_b"],
                    "count": pair["coactivation_count"],
                    "strength": strength,
                }
            )

    conn.commit()
    return new_connections


# ══════════════════════════════════════════════════════════════
# PASO 4: Actualizar representaciones
# ══════════════════════════════════════════════════════════════


def _update_representations(conn, db_path=None):
    """
    For peers with enough activations, update their confidence
    and re-embed their facets based on accumulated memories.

    The representation evolves as more memories are linked.
    Confidence increases with each relevant memory.
    """
    peers = conn.execute(
        """
        SELECT p.* FROM peers p
        WHERE p.is_active = 1
        AND p.activation_count >= ?
    """,
        [REPRESENTATION_UPDATE_INTERVAL],
    ).fetchall()

    updated = []
    for peer in peers:
        # Get top memories for this peer
        memories = conn.execute(
            """
            SELECT m.content, mp.relevance
            FROM memories m
            JOIN memory_peers mp ON m.id = mp.memory_id
            WHERE mp.peer_id = ? AND m.is_active = 1
            ORDER BY mp.relevance DESC
            LIMIT 20
        """,
            [peer["id"]],
        ).fetchall()

        if not memories:
            continue

        # Calculate new confidence based on memory count and relevance
        avg_relevance = sum(m["relevance"] for m in memories) / len(memories)
        memory_factor = min(1.0, len(memories) / 50.0)
        new_confidence = min(0.95, avg_relevance * memory_factor + 0.1)

        # Build updated representation from top memories
        top_3 = memories[:3]
        memory_summaries = [f"- {m['content'][:80]}" for m in top_3]

        new_representation = (
            f"{peer['representation'] or peer['description']}\n\n"
            f"Memorias clave ({len(memories)} total):\n" + "\n".join(memory_summaries)
        )

        # Update peer
        conn.execute(
            """
            UPDATE peers
            SET confidence = ?, representation = ?, updated_at = datetime('now')
            WHERE id = ?
        """,
            [round(new_confidence, 3), new_representation, peer["id"]],
        )

        # Re-embed ALL facets for this peer with enriched representation
        facets = conn.execute(
            """
            SELECT id, text FROM peer_facets WHERE peer_id = ?
        """,
            [peer["id"]],
        ).fetchall()

        # Optionally enrich facet text with memory context
        for facet in facets:
            enriched_text = facet["text"]
            if top_3:
                # Append top memory context to facet embedding text
                context_snippets = " ".join(m["content"][:50] for m in top_3[:2])
                enriched_text = f"{facet['text']} Contexto: {context_snippets}"

            vector = embed(enriched_text)
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)
            conn.execute(
                "DELETE FROM facet_embeddings WHERE facet_id = ?", [facet["id"]]
            )
            conn.execute(
                "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
                [facet["id"], vec_bytes],
            )

        updated.append(
            {
                "peer_id": peer["id"],
                "old_confidence": peer["confidence"],
                "new_confidence": round(new_confidence, 3),
                "memories": len(memories),
                "facets_re_embedded": len(facets),
            }
        )

    conn.commit()
    return updated


# ══════════════════════════════════════════════════════════════
# PASO 5: Olvido (decay)
# ══════════════════════════════════════════════════════════════


def _forget_memories(conn):
    """
    Soft-delete old memories with low confidence.
    Like the brain's pruning during sleep — what's not reinforced fades.
    """
    cutoff = (datetime.now() - timedelta(days=FORGET_AFTER_DAYS)).isoformat()

    to_forget = conn.execute(
        """
        SELECT id FROM memories
        WHERE is_active = 1
        AND confidence < ?
        AND created_at < ?
        AND source = 'dreaming'
        AND id NOT IN (
            SELECT memory_id FROM memory_peers
            WHERE relevance > 0.5
        )
    """,
        [FORGET_CONFIDENCE, cutoff],
    ).fetchall()

    count = len(to_forget)
    for m in to_forget:
        conn.execute(
            "UPDATE memories SET is_active = 0, updated_at = datetime('now') WHERE id = ?",
            [m["id"]],
        )

    conn.commit()
    return count


# ══════════════════════════════════════════════════════════════
# CONSOLIDATION LOG
# ══════════════════════════════════════════════════════════════


def _merge_similar_facets(conn, db_path=None):
    """
    Merge facets within the same peer that are very similar.
    Uses similarity > 0.9 threshold.
    """
    from .learning import merge_facets

    # For each peer, find facets and check similarities
    peers = conn.execute("SELECT id FROM peers WHERE is_active = 1").fetchall()
    merged_count = 0

    for peer in peers:
        pid = peer["id"]
        # Get facets for this peer
        facets = conn.execute(
            """
            SELECT pf.id, fe.embedding
            FROM peer_facets pf
            JOIN facet_embeddings fe ON pf.id = fe.facet_id
            WHERE pf.peer_id = ?
        """,
            [pid],
        ).fetchall()

        if len(facets) < 2:
            continue

        # Compare each pair
        for i in range(len(facets)):
            for j in range(i + 1, len(facets)):
                fid1, emb1 = facets[i]["id"], facets[i]["embedding"]
                fid2, emb2 = facets[j]["id"], facets[j]["embedding"]

                # Unpack and compare
                try:
                    dims = len(embed("test"))  # Get dims
                    vec1 = list(struct.unpack(f"{dims}f", emb1))
                    vec2 = list(struct.unpack(f"{dims}f", emb2))
                    sim = cosine_similarity(vec1, vec2)
                    if sim > 0.9:
                        merge_facets(fid1, fid2, db_path)
                        merged_count += 1
                        break  # Merged one, continue to next
                except:
                    continue

    conn.commit()
    return merged_count


def _start_log(conn, session_id=None):
    """Start a consolidation log entry."""
    notes = f"session={session_id}" if session_id else "auto"
    conn.execute(
        """
        INSERT INTO consolidation_log (started_at, status, notes)
        VALUES (datetime('now'), 'running', ?)
    """,
        [notes],
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _finish_log(conn, log_id, status, stats, notes=None):
    """Finish a consolidation log entry."""
    conn.execute(
        """
        UPDATE consolidation_log
        SET finished_at = datetime('now'),
            status = ?,
            memories_processed = ?,
            peers_updated = ?,
            connections_found = ?,
            memories_added = ?,
            memories_deleted = ?,
            errors = ?,
            notes = ?
        WHERE id = ?
    """,
        [
            status,
            stats["events_processed"],
            stats["peers_updated"],
            stats["connections_found"],
            stats["memories_created"],
            stats["memories_forgotten"],
            json.dumps(stats["errors"]),
            notes,
            log_id,
        ],
    )
