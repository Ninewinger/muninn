"""Muninn — API reference.

Base URL: http://localhost:8000/api/v1

## Event Stream

### POST /events
Send an event (user message, bot response, tool call, etc.).
Triggers semantic router and returns activated peers.

**Request:**
```json
{
    "session_id": "session_001",
    "type": "user_message",
    "content": "I felt weird today",
    "channel": "cli",
    "metadata": {}
}
```

**Response:**
```json
{
    "event_id": 42,
    "activations": [
        {
            "peer_id": "sombra_ansiedad",
            "peer_name": "Sombra Ansiedad",
            "similarity": 0.78,
            "representation": "..."
        }
    ]
}
```

## Memories

### POST /memories
Store a new memory. Lifecycle classifier decides ADD/UPDATE/DELETE/NOOP.

### GET /memories/{id}
Get a memory with linked peers and similar memories.

### GET /memories?peer_id=x&limit=10&type=episodio
List memories with filters.

### PUT /memories/{id}
Update a memory's content or confidence.

### DELETE /memories/{id}
Soft delete (is_active = 0).

## Peers

### POST /peers
Create a new peer. Embedding calculated automatically.

### GET /peers?type=sombra&is_active=true
List peers with filters.

### GET /peers/{id}
Get full peer details + connections + recent memories.

### PUT /peers/{id}
Update representation, confidence, tags.

### DELETE /peers/{id}
Soft delete.

### POST /peers/{id}/activate
Force manual activation. Returns representation + memories + connections.

### GET /peers/{id}/memories
Get memories linked to this peer.

## Connections

### POST /connections
Create a connection between two peers.

### GET /connections?peer_id=x
Get all connections for a peer.

## Search

### POST /search
Hybrid search (semantic + full-text).

**Request:**
```json
{
    "query": "emotional variables in combat",
    "peer_id": null,
    "limit": 10,
    "method": "hybrid"
}
```

## Consolidation

### POST /consolidate
Trigger consolidation (dreaming) for specific peers or all.

### GET /consolidate/{job_id}
Check consolidation job status.

### GET /consolidate/log
View consolidation history.

## Sessions

### POST /sessions
Create a new session.

### PUT /sessions/{id}
Close a session.

### GET /sessions/{id}/context
Get active context (active peers + relevant memories).

## Health

### GET /
Health check.

### GET /stats
System statistics.
"""
