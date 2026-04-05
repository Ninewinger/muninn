"""Dreaming/Consolidation process for Muninn."""

import json
import urllib.request
from datetime import datetime
from typing import Optional


def call_llm(prompt: str, api_url: str, model: str, api_key: str) -> str:
    """Call an OpenAI-compatible LLM API."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def consolidate_peer(
    peer_id: str,
    api_url: str,
    model: str,
    api_key: str,
    db_path: Optional[str] = None,
) -> dict:
    """
    Consolidate memories for a single peer.
    Uses LLM to detect patterns, update representation, find connections.
    """
    from .db import get_connection

    conn = get_connection(db_path)

    # Get peer info
    peer = conn.execute(
        "SELECT * FROM peers WHERE id = ?", (peer_id,)
    ).fetchone()

    if not peer:
        return {"error": f"Peer {peer_id} not found"}

    # Get recent unconsolidated memories for this peer
    memories = conn.execute("""
        SELECT m.* FROM memories m
        JOIN memory_peers mp ON m.id = mp.memory_id
        WHERE mp.peer_id = ? AND m.is_active = 1
        ORDER BY m.created_at DESC
        LIMIT 50
    """, (peer_id,)).fetchall()

    if not memories:
        return {"status": "skipped", "reason": "no memories to consolidate"}

    # Build prompt for LLM
    memories_text = "\n".join([
        f"- [{m['type']}] [{m['occurred_at'] or m['created_at']}] {m['content']}"
        for m in memories
    ])

    prompt = f"""You are Muninn, a memory consolidation system. Analyze the following memories for the peer "{peer['name']}" (type: {peer['type']}).

Current representation:
{peer['representation'] or 'None yet'}

Current confidence: {peer['confidence']}

Recent memories:
{memories_text}

Tasks:
1. Write an updated representation (2-4 sentences) that incorporates new learnings
2. Identify any new patterns or insights
3. Rate the overall confidence (0.0-1.0) based on how well we understand this peer
4. Suggest any connections to other potential peers

Respond in JSON format:
{{
    "representation": "updated representation text",
    "confidence": 0.7,
    "patterns_found": ["pattern 1", "pattern 2"],
    "suggested_connections": [
        {{"peer_name": "...", "relation_type": "conecta", "description": "..."}}
    ]
}}"""

    try:
        response = call_llm(prompt, api_url, model, api_key)

        # Parse JSON response
        # Handle cases where LLM adds markdown formatting
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]

        result = json.loads(response)

        # Update peer in database
        conn.execute("""
            UPDATE peers
            SET representation = ?, confidence = ?, updated_at = ?
            WHERE id = ?
        """, (
            result.get("representation", peer["representation"]),
            result.get("confidence", peer["confidence"]),
            datetime.now().isoformat(),
            peer_id,
        ))

        conn.commit()
        conn.close()

        return {
            "status": "completed",
            "peer_id": peer_id,
            "patterns_found": result.get("patterns_found", []),
            "suggested_connections": result.get("suggested_connections", []),
        }

    except Exception as e:
        conn.close()
        return {"status": "failed", "error": str(e)}
