"""Test Muninn API endpoints."""

import os
import pytest
import tempfile
from httpx import AsyncClient, ASGITransport

os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize DB for all tests."""
    from muninn.db import init_db, get_db_path
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.unlink(db_path)
    conn = init_db(db_path)
    conn.close()
    yield
    # Cleanup — retry with delay for Windows file locks
    import time
    for _ in range(3):
        try:
            os.unlink(db_path)
            break
        except PermissionError:
            time.sleep(0.5)


@pytest.fixture
async def client():
    from muninn.api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Muninn"
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_stats_empty(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_peers"] == 0
    assert data["total_memories"] == 0


@pytest.mark.anyio
async def test_create_peer(client):
    r = await client.post("/api/v1/peers", json={
        "id": "sombra_rechazo",
        "name": "Sombra de Rechazo",
        "type": "sombra",
        "description": "Rechazo maternal y su proyeccion en relaciones",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == "sombra_rechazo"
    assert data["name"] == "Sombra de Rechazo"
    assert data["type"] == "sombra"
    assert data["is_active"] == 1
    assert data["confidence"] == 0.1


@pytest.mark.anyio
async def test_create_peer_duplicate(client):
    await client.post("/api/v1/peers", json={
        "id": "test_dup", "name": "Test", "type": "tema",
    })
    r = await client.post("/api/v1/peers", json={
        "id": "test_dup", "name": "Test 2", "type": "tema",
    })
    assert r.status_code == 409


@pytest.mark.anyio
async def test_list_peers(client):
    # Create two peers
    await client.post("/api/v1/peers", json={
        "id": "peer_a", "name": "Peer A", "type": "sombra",
    })
    await client.post("/api/v1/peers", json={
        "id": "peer_b", "name": "Peer B", "type": "tema",
    })

    # List all
    r = await client.get("/api/v1/peers")
    assert r.status_code == 200
    assert len(r.json()) == 2

    # Filter by type
    r = await client.get("/api/v1/peers", params={"type": "sombra"})
    assert r.status_code == 200
    assert len(r.json()) == 1


@pytest.mark.anyio
async def test_get_peer(client):
    await client.post("/api/v1/peers", json={
        "id": "get_test", "name": "Get Test", "type": "tema",
    })
    r = await client.get("/api/v1/peers/get_test")
    assert r.status_code == 200
    assert r.json()["name"] == "Get Test"

    r = await client.get("/api/v1/peers/nonexistent")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_update_peer(client):
    await client.post("/api/v1/peers", json={
        "id": "update_test", "name": "Original", "type": "tema",
    })
    r = await client.put("/api/v1/peers/update_test", json={
        "name": "Updated",
        "confidence": 0.8,
        "tags": ["test", "updated"],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Updated"
    assert data["confidence"] == 0.8
    assert data["tags"] == ["test", "updated"]


@pytest.mark.anyio
async def test_delete_peer(client):
    await client.post("/api/v1/peers", json={
        "id": "del_test", "name": "Delete Me", "type": "tema",
    })
    r = await client.delete("/api/v1/peers/del_test")
    assert r.status_code == 200

    # Verify soft delete
    r = await client.get("/api/v1/peers/del_test")
    assert r.status_code == 200
    assert r.json()["is_active"] == 0
