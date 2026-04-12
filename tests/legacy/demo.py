"""Muninn End-to-End Demo - prueba todos los endpoints."""

import json
import os
import sys
import time
import subprocess

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Set DB path
DB_PATH = os.path.join(os.path.dirname(__file__), "demo_muninn.db")
os.environ["DB_PATH"] = DB_PATH

# Clean previous
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)

from muninn.db import init_db
import httpx

print("=" * 60)
print("  MUNINN - End-to-End Demo")
print("=" * 60)

# Init DB
print("\n[1/8] Inicializando database...")
conn = init_db(DB_PATH)
conn.close()
print("  OK - Schema creado")

# Start server
print("\n[2/8] Levantando servidor...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "muninn.api:app", "--host", "127.0.0.1", "--port", "8901"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
time.sleep(6)

BASE = "http://127.0.0.1:8901"

try:
    # Health check
    print("\n[3/8] Health check...")
    r = httpx.get(f"{BASE}/")
    print(f"  {r.json()}")

    r = httpx.get(f"{BASE}/stats")
    print(f"  Stats: {r.json()}")

    # CREATE PEERS
    print("\n[4/8] Creando Peers (sombras)...")

    peers_data = [
        {
            "id": "sombra_rechazo",
            "name": "Sombra de Rechazo",
            "type": "sombra",
            "description": "Rechazo maternal proyectado en relaciones femeninas",
            "embedding_text": "rechazo maternal abandono dolor mujeres no me quieren",
            "activation_threshold": 0.3,
            "tags": ["relaciones", "madre", "femenino"],
        },
        {
            "id": "sombra_muerte",
            "name": "Sombra de Muerte",
            "type": "sombra",
            "description": "Miedo a perder seres queridos, hermano fallecido",
            "embedding_text": "muerte perder alguien fallecimiento hermano duelo",
            "activation_threshold": 0.3,
            "tags": ["muerte", "familia", "duelo"],
        },
        {
            "id": "sombra_ansiedad",
            "name": "Sombra Ansiedad IA",
            "type": "sombra",
            "description": "Ansiedad por cantidad de informacion sobre IA",
            "embedding_text": "ansiedad inteligencia artificial abrumado informacion tecnologia",
            "activation_threshold": 0.3,
            "tags": ["IA", "tecnologia", "ansiedad"],
        },
        {
            "id": "sombra_fortaleza",
            "name": "Mi Fortaleza",
            "type": "sombra",
            "description": "Mascara de perfeccion, raiz en rechazo materno",
            "embedding_text": "fortaleza mascara perfeccion fuerte proteger demostrar valer",
            "activation_threshold": 0.3,
            "tags": ["fortaleza", "identidad", "proteccion"],
        },
    ]

    for p in peers_data:
        r = httpx.post(f"{BASE}/api/v1/peers", json=p, timeout=30)
        if r.status_code == 201:
            print(f"  + {p['name']} (id: {p['id']})")
        else:
            print(f"  ERROR: {r.status_code} {r.text}")

    r = httpx.get(f"{BASE}/api/v1/peers")
    print(f"  Total peers: {len(r.json())}")

    # CREATE MEMORIES
    print("\n[5/8] Creando Memories...")

    memories_data = [
        {
            "content": "Diego sintio rechazo al hablar con ella en la fiesta. Patron de buscar amor donde no florece.",
            "type": "episodio",
            "source": "conversation",
            "peer_ids": ["sombra_rechazo"],
        },
        {
            "content": "La madre perdio un hijo un ano antes de que Diego naciera. El bebe murio a los 6 meses de gestacion.",
            "type": "hecho",
            "source": "obsidian",
            "peer_ids": ["sombra_muerte", "sombra_rechazo", "sombra_fortaleza"],
        },
        {
            "content": "Para un bebe, rechazo materno equivale a muerte. Esa ecuacion se transfirio a todas las relaciones femeninas.",
            "type": "hipotesis",
            "source": "conversation",
            "confidence": 0.8,
            "peer_ids": ["sombra_rechazo", "sombra_muerte"],
        },
        {
            "content": "Le genera mucha ansiedad pensar en el futuro de la IA. Siente que hay demasiada informacion.",
            "type": "episodio",
            "source": "telegram",
            "peer_ids": ["sombra_ansiedad"],
        },
        {
            "content": "Descubrio que hacer ejercicios de piernas en el gimnasio fue la respuesta a su debilidad historica.",
            "type": "episodio",
            "source": "conversation",
            "confidence": 0.9,
            "peer_ids": ["sombra_fortaleza"],
        },
    ]

    for m in memories_data:
        r = httpx.post(f"{BASE}/api/v1/memories", json=m, timeout=30)
        if r.status_code == 201:
            data = r.json()
            print(f"  + [{m['type']}] id:{data['id']} -> {m['content'][:55]}...")
        else:
            print(f"  ERROR: {r.status_code} {r.text}")

    r = httpx.get(f"{BASE}/stats")
    print(f"  Stats: {r.json()}")

    # CONNECTIONS
    print("\n[6/8] Creando Connections entre sombras...")

    connections_data = [
        {"from_peer_id": "sombra_rechazo", "to_peer_id": "sombra_muerte",
         "relation_type": "conecta", "strength": 0.9,
         "description": "Rechazo materno = muerte para el bebe"},
        {"from_peer_id": "sombra_rechazo", "to_peer_id": "sombra_fortaleza",
         "relation_type": "evoluciona_de", "strength": 0.7,
         "description": "Fortaleza nace como mascara contra el rechazo"},
        {"from_peer_id": "sombra_muerte", "to_peer_id": "sombra_ansiedad",
         "relation_type": "activa", "strength": 0.5,
         "description": "Miedo a la muerte amplifica ansiedad existencial"},
    ]

    for c in connections_data:
        r = httpx.post(f"{BASE}/api/v1/connections", json=c)
        if r.status_code == 201:
            print(f"  + {c['from_peer_id']} --[{c['relation_type']}]--> {c['to_peer_id']} ({c['strength']})")
        else:
            print(f"  ERROR: {r.status_code} {r.text}")

    # EVENTS (Router test)
    print("\n[7/8] Probando Events (Semantic Router)...")
    print("  " + "-" * 50)

    test_events = [
        ("Me siento rechazado por las mujeres", "sombra_rechazo"),
        ("Tengo miedo de perder a mi familia", "sombra_muerte"),
        ("La IA me abruma con tanta informacion", "sombra_ansiedad"),
        ("Me siento fuerte hoy, hice piernas", "sombra_fortaleza"),
        ("Hola, como estas?", "NINGUNO"),
    ]

    for event_text, expected in test_events:
        r = httpx.post(f"{BASE}/api/v1/events", json={
            "session_id": "demo_session",
            "type": "user_message",
            "content": event_text,
            "channel": "cli",
        }, timeout=30)
        if r.status_code == 200:
            data = r.json()
            activations = data["activations"]
            top = activations[0] if activations else None
            top_name = top["name"] if top else "NINGUNO"
            top_sim = top["similarity"] if top else 0
            match = "OK" if (expected in top_name if top else expected == "NINGUNO") else "MISS"
            print(f"  [{match}] \"{event_text[:40]}\"")
            print(f"       Expected: {expected}")
            print(f"       Got:      {top_name} (sim={top_sim:.3f})")
            if len(activations) > 1:
                for a in activations[1:]:
                    print(f"       Also:     {a['name']} (sim={a['similarity']:.3f})")
        else:
            print(f"  ERROR: {r.status_code} {r.text}")

    # SEARCH
    print("\n[8/8] Probando Search (hybrid)...")
    print("  " + "-" * 50)

    test_searches = [
        "rechazo mujeres",
        "miedo muerte familia",
        "ansiedad tecnologia IA",
        "fortaleza gym piernas",
    ]

    for query in test_searches:
        r = httpx.post(f"{BASE}/api/v1/search", json={
            "query": query,
            "method": "hybrid",
            "limit": 3,
        }, timeout=30)
        if r.status_code == 200:
            results = r.json()
            print(f"  QUERY: \"{query}\"")
            for res in results:
                peers_str = ", ".join(res["peers"]) if res["peers"] else "none"
                print(f"    -> [{res['type']}] score={res['score']:.3f} | {res['content'][:50]}...")
                print(f"       peers: [{peers_str}]")
            if not results:
                print(f"    -> No results")
        else:
            print(f"  ERROR: {r.status_code} {r.text}")

    # Final stats
    print("\n" + "=" * 60)
    r = httpx.get(f"{BASE}/stats")
    stats = r.json()
    print(f"  STATS FINALES:")
    print(f"    Peers:       {stats['total_peers']}")
    print(f"    Memories:    {stats['total_memories']}")
    print(f"    Events:      {stats['total_events']}")
    print(f"    Activations: {stats['total_activations']}")
    print("=" * 60)
    print("\n  DEMO COMPLETADA!")

finally:
    proc.terminate()
    proc.wait(timeout=5)
    if os.path.exists(DB_PATH):
        try:
            os.unlink(DB_PATH)
        except Exception:
            pass
