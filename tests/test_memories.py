"""Test Memories and Connections API endpoints."""

import os
import pytest
import tempfile
from httpx import AsyncClient, ASGITransport

DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DB_PATH"] = DB_PATH


@pytest.fixture(autouse=True)
def setup_db():
    from muninn.db import init_db
    if os.path.exists(DB_PATH):
        try:
            os.unlink(DB_PATH)
        except PermissionError:
            import time
            time.sleep(0.5)
            try:
                os.unlink(DB_PATH)
            except Exception:
                pass
    conn = init_db(DB_PATH)
    conn.close()
    yield
    import time
    for _ in range(3):
        try:
            os.unlink(DB_PATH)
            break
        except PermissionError:
            time.sleep(0.5)


@pytest.fixture
async def client():
    from muninn.api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def sample_peer(client):
    """Create a sample peer for memory tests."""
    r = await client.post("/api/v1/peers", json={
        "id": "sombra_test",
        "name": "Sombra Test",
        "type": "sombra",
        "description": "Una sombra de prueba",
    })
    return r.json()


# ── Memories ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_create_memory(client, sample_peer):
    r = await client.post("/api/v1/memories", json={
        "content": "Diego sintio rechazo al hablar con ella",
        "type": "episodio",
        "source": "conversation",
        "peer_ids": ["sombra_test"],
    })
    assert r.status_code == 201
    data = r.json()
    assert data["content"] == "Diego sintio rechazo al hablar con ella"
    assert data["type"] == "episodio"
    assert len(data["peers"]) == 1
    assert data["peers"][0]["peer_id"] == "sombra_test"


@pytest.mark.anyio
async def test_create_memory_without_peer(client):
    r = await client.post("/api/v1/memories", json={
        "content": "Un hecho cualquiera",
        "type": "hecho",
    })
    assert r.status_code == 201
    assert len(r.json()["peers"]) == 0


@pytest.mark.anyio
async def test_list_memories(client, sample_peer):
    # Create two memories
    await client.post("/api/v1/memories", json={
        "content": "Memoria A", "type": "hecho", "peer_ids": ["sombra_test"],
    })
    await client.post("/api/v1/memories", json={
        "content": "Memoria B", "type": "episodio",
    })

    # List all
    r = await client.get("/api/v1/memories")
    assert r.status_code == 200
    assert len(r.json()) == 2

    # Filter by peer
    r = await client.get("/api/v1/memories", params={"peer_id": "sombra_test"})
    assert r.status_code == 200
    assert len(r.json()) == 1

    # Filter by type
    r = await client.get("/api/v1/memories", params={"type": "episodio"})
    assert r.status_code == 200
    assert len(r.json()) == 1


@pytest.mark.anyio
async def test_get_memory(client, sample_peer):
    created = await client.post("/api/v1/memories", json={
        "content": "Memoria especifica", "type": "hecho",
    })
    mem_id = created.json()["id"]

    r = await client.get(f"/api/v1/memories/{mem_id}")
    assert r.status_code == 200
    assert r.json()["content"] == "Memoria especifica"

    r = await client.get("/api/v1/memories/9999")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_update_memory(client):
    created = await client.post("/api/v1/memories", json={
        "content": "Original", "type": "hecho",
    })
    mem_id = created.json()["id"]

    r = await client.put(f"/api/v1/memories/{mem_id}", json={
        "content": "Actualizada",
        "confidence": 0.9,
    })
    assert r.status_code == 200
    assert r.json()["content"] == "Actualizada"
    assert r.json()["confidence"] == 0.9


@pytest.mark.anyio
async def test_delete_memory(client):
    created = await client.post("/api/v1/memories", json={
        "content": "Para borrar", "type": "hecho",
    })
    mem_id = created.json()["id"]

    r = await client.delete(f"/api/v1/memories/{mem_id}")
    assert r.status_code == 200

    # Verify soft delete
    r = await client.get(f"/api/v1/memories/{mem_id}")
    assert r.json()["is_active"] == 0


# ── Connections ────────────────────────────────────────────

@pytest.mark.anyio
async def test_create_connection(client):
    await client.post("/api/v1/peers", json={"id": "peer_x", "name": "X", "type": "sombra"})
    await client.post("/api/v1/peers", json={"id": "peer_y", "name": "Y", "type": "tema"})

    r = await client.post("/api/v1/connections", json={
        "from_peer_id": "peer_x",
        "to_peer_id": "peer_y",
        "relation_type": "conecta",
        "strength": 0.8,
        "description": "Test connection",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["from_peer_id"] == "peer_x"
    assert data["strength"] == 0.8


@pytest.mark.anyio
async def test_list_connections(client):
    await client.post("/api/v1/peers", json={"id": "conn_a", "name": "A", "type": "tema"})
    await client.post("/api/v1/peers", json={"id": "conn_b", "name": "B", "type": "tema"})
    await client.post("/api/v1/peers", json={"id": "conn_c", "name": "C", "type": "tema"})

    await client.post("/api/v1/connections", json={
        "from_peer_id": "conn_a", "to_peer_id": "conn_b",
        "relation_type": "conecta",
    })
    await client.post("/api/v1/connections", json={
        "from_peer_id": "conn_a", "to_peer_id": "conn_c",
        "relation_type": "activa",
    })

    # All connections
    r = await client.get("/api/v1/connections")
    assert len(r.json()) == 2

    # Filter by peer
    r = await client.get("/api/v1/connections", params={"peer_id": "conn_a"})
    assert len(r.json()) == 2


@pytest.mark.anyio
async def test_connection_nonexistent_peer(client):
    r = await client.post("/api/v1/connections", json={
        "from_peer_id": "nope", "to_peer_id": "also_nope",
        "relation_type": "conecta",
    })
    assert r.status_code == 404
