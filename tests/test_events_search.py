"""Test Events and Search API endpoints."""

import os
import pytest
import tempfile
from httpx import AsyncClient, ASGITransport

DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DB_PATH"] = DB_PATH


@pytest.fixture(autouse=True)
def setup_db():
    from muninn.db import init_db, get_db_path
    db_path = get_db_path()
    if os.path.exists(db_path):
        import time
        for _ in range(5):
            try:
                os.unlink(db_path)
                break
            except PermissionError:
                time.sleep(0.5)
    conn = init_db(db_path)
    conn.close()
    yield
    import time
    for _ in range(5):
        try:
            os.unlink(db_path)
            break
        except PermissionError:
            time.sleep(1)


@pytest.fixture
async def client():
    from muninn.api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def seeded_db(client):
    """Seed DB with peers and memories for event/search tests."""
    # Create peers with embeddings
    await client.post("/api/v1/peers", json={
        "id": "sombra_rechazo",
        "name": "Sombra de Rechazo",
        "type": "sombra",
        "description": "Rechazo maternal dolor profundo",
        "embedding_text": "rechazo de la madre dolor profundo abandono",
        "activation_threshold": 0.3,
    })
    await client.post("/api/v1/peers", json={
        "id": "sombra_muerte",
        "name": "Sombra de Muerte",
        "type": "sombra",
        "description": "Miedo a la muerte perder seres queridos",
        "embedding_text": "miedo a la muerte perder a alguien fallecimiento",
        "activation_threshold": 0.3,
    })
    await client.post("/api/v1/peers", json={
        "id": "sombra_ansiedad",
        "name": "Sombra Ansiedad",
        "type": "sombra",
        "description": "Ansiedad generalizada preocupacion",
        "embedding_text": "ansiedad preocupacion nervioso estres angustia",
        "activation_threshold": 0.3,
    })

    # Create memories linked to peers
    await client.post("/api/v1/memories", json={
        "content": "Diego sintio rechazo al intentar hablar con ella en la fiesta",
        "type": "episodio",
        "peer_ids": ["sombra_rechazo"],
    })
    await client.post("/api/v1/memories", json={
        "content": "La muerte de su hermano antes de nacer impacto toda la familia",
        "type": "hecho",
        "peer_ids": ["sombra_muerte", "sombra_rechazo"],
    })
    await client.post("/api/v1/memories", json={
        "content": "Le genera mucha ansiedad pensar en el futuro de la IA",
        "type": "episodio",
        "peer_ids": ["sombra_ansiedad"],
    })


# ── Events ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_event_activates_peers(client, seeded_db):
    r = await client.post("/api/v1/events", json={
        "session_id": "test_session_1",
        "type": "user_message",
        "content": "me siento rechazado por las mujeres",
        "channel": "cli",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["event_id"] > 0
    assert isinstance(data["activations"], list)

    # sombra_rechazo should be the top activation
    if data["activations"]:
        top = data["activations"][0]
        assert "peer_id" in top
        assert "similarity" in top
        print(f"Top activation: {top['peer_id']} ({top['similarity']})")


@pytest.mark.anyio
async def test_event_without_session(client, seeded_db):
    r = await client.post("/api/v1/events", json={
        "type": "user_message",
        "content": "hola mundo",
    })
    assert r.status_code == 200
    assert r.json()["event_id"] > 0


@pytest.mark.anyio
async def test_event_increments_activation_count(client, seeded_db):
    # Send event that should activate rechazo
    await client.post("/api/v1/events", json={
        "type": "user_message",
        "content": "rechazo maternal",
    })

    # Check peer was updated
    peer = await client.get("/api/v1/peers/sombra_rechazo")
    assert peer.json()["activation_count"] >= 1


# ── Search ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_semantic_search(client, seeded_db):
    r = await client.post("/api/v1/search", json={
        "query": "rechazo de las mujeres",
        "method": "semantic",
        "limit": 5,
    })
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
    # Top result should be about rechazo
    assert "rechazo" in results[0]["content"].lower()


@pytest.mark.anyio
async def test_fts_search(client, seeded_db):
    r = await client.post("/api/v1/search", json={
        "query": "ansiedad",
        "method": "fts",
        "limit": 5,
    })
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
    assert "ansiedad" in results[0]["content"].lower()


@pytest.mark.anyio
async def test_hybrid_search(client, seeded_db):
    r = await client.post("/api/v1/search", json={
        "query": "rechazo",
        "method": "hybrid",
        "limit": 10,
    })
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
