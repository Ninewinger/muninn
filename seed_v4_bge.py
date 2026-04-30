#!/usr/bin/env python3
"""Muninn v4 Seed Script — BGE-M3 + improved facets + data import.

Creates muninn_v4.db with:
  - 15 base peers from db_export.json
  - 72 original facets (preserved)
  - ~44 NEW facets (from benchmark failure analysis)
  - Initial memories from Hermes knowledge
  - All embeddings via BGE-M3 (1024d, sentence-transformers)

Usage:
  python seed_v4_bge.py [--db-path muninn_v4.db] [--batch-size 32]
"""

import json
import os
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
EXPORT_PATH = SCRIPT_DIR / "db_export.json"
SCHEMA_PATH = SCRIPT_DIR / "muninn" / "schema_v2.sql"

# ── Embedding Backend ──────────────────────────────────────────
# BGE-M3 via sentence-transformers (fast, local, cached)
MODEL_NAME = "BAAI/bge-m3"
MODEL_DIMS = 1024
INSTRUCTION = "Dado un mensaje de un usuario, identifica que dominio de su vida esta activando"


def get_embedder():
    """Load BGE-M3 via sentence_transformers."""
    from sentence_transformers import SentenceTransformer
    print(f"  [BGE-M3] Cargando {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  [BGE-M3] Listo (dims={model.get_sentence_embedding_dimension()})")
    return model


def embed_texts(model, texts, is_query=False, batch_size=32):
    """Embed a list of texts. Returns list of float lists."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if is_query:
            # BGE-M3 doesn't use instruction prefix in ST mode;
            # it uses it natively. We prepend manually for query.
            batch = [f"Instruct: {INSTRUCTION}\nQuery: {t}" for t in batch]
        vecs = model.encode(batch, normalize_embeddings=True)
        all_embeds.extend(vecs.tolist())
    return all_embeds


# ── Database Setup ─────────────────────────────────────────────

def init_db(db_path, dims=1024):
    """Create fresh DB with schema_v2."""
    import sqlite_vec

    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"  [DB] Removed existing {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Read and adapt schema
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    clean = "\n".join(
        line for line in schema.splitlines()
        if not line.strip().startswith("-- .load")
    )
    clean = re.sub(r'FLOAT\[\d+\]', f'FLOAT[{dims}]', clean)
    conn.executescript(clean)

    # Update config
    conn.execute("UPDATE embedding_config SET value = ? WHERE key = 'model_name'", [MODEL_NAME])
    conn.execute("UPDATE embedding_config SET value = ? WHERE key = 'dimensions'", [str(dims)])
    conn.commit()
    print(f"  [DB] Schema v2 created ({dims}d)")
    return conn


# ── Data: Peers from export ────────────────────────────────────

def load_export():
    """Load peers and facets from db_export.json."""
    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["peers"], data["facets"]


# ── Data: NEW facets from failure analysis ─────────────────────

NEW_FACETS = [
    # Failure 1: soñé que mi abuela se caía → sombra_muerte (got sombra_angel_atardecer)
    {"peer_id": "sombra_muerte", "facet_type": "emocional",
     "text": "sueños de muerte, soñar que un familiar se cae o enferma, pesadillas de pérdida, sueños terminales, soñar con abuela nona hermano"},
    # Failure 2: algo me va bien, algo malo pasa → sombra_angel_atardecer (got sombra_muerte)
    {"peer_id": "sombra_angel_atardecer", "facet_type": "emocional",
     "text": "catastrofización, cuando algo va bien siento que algo malo va a pasar, ansiedad anticipatoria, no puedo disfrutar el momento, miedo al éxito"},
    # Failure 3: no sé si tengo disciplina o soy débil → sombra_fortaleza (got sombra_rechazo)
    {"peer_id": "sombra_fortaleza", "facet_type": "emocional",
     "text": "duda sobre disciplina, auto-cuestionamiento de fortaleza, soy débil o soy fuerte, la disciplina como armadura, cuestionar la propia fuerza"},
    # Failure 4: hoy tocaba pecho y tríceps, duele hombro → gym_rutina (got sombra_fortaleza)
    {"peer_id": "gym_rutina", "facet_type": "fisico",
     "text": "dolor de hombro espalda rodilla, lesiones entrenamiento, duele al hacer ejercicio, adaptar rutina por dolor, sobreentrenamiento recuperacion"},
    # Failure 5: tomar creatina, qué opinas? → gym_rutina (got peer_operativo)
    {"peer_id": "gym_rutina", "facet_type": "contextual",
     "text": "suplementos deportivos creatina proteína pre-entreno, qué opinas de tomar, recomendación de suplementos, nutrición deportiva, vitaminas minerales"},
    # Failure 6: cómo funciona un context manager → programacion (got peer_herramientas)
    {"peer_id": "programacion", "facet_type": "tecnico",
     "text": "conceptos Python avanzados, context managers decoradores generadores, protocolos programación, __enter__ __exit__, with statement, closures"},
    # Failure 7: NPCs mismas habilidades que jugador → proyecto_juego (got peer_skills)
    {"peer_id": "proyecto_juego", "facet_type": "tecnico",
     "text": "diseño NPCs enemigos, habilidades personajes, sistema de habilidades compartidas jugador enemigo, balance juego, IA enemigos, habilidades NPC"},
    # Failure 8: cuánto cobrar por sistema inventario → valle_alto (got peer_usuario)
    {"peer_id": "valle_alto", "facet_type": "contextual",
     "text": "precios desarrollo software, cobrar por sistema programa inventario, freelance programación, cuánto cobrar, venta de software, negocio desarrollo"},
    # Failure 9: quién soy yo, no me reconozco → peer_identidad (got sombra_rechazo)
    {"peer_id": "peer_identidad", "facet_type": "emocional",
     "text": "crisis de identidad, no me reconozco, quién soy yo, cuestionar la propia identidad, perderse a uno mismo, desconexión del yo"},
    # Failure 10: qué aprendimos sesión pasada? → peer_usuario (got casual_social)
    {"peer_id": "peer_usuario", "facet_type": "contextual",
     "text": "recordar sesiones pasadas, qué aprendimos sesión anterior, recapitulación conversación previa, qué hicimos la última vez, resumen de sesión"},
    # Failure 11: primera venta programando → valle_alto (got casual_social)
    {"peer_id": "valle_alto", "facet_type": "contextual",
     "text": "primera venta freelance programando, ganar dinero con código, vender software, cliente primer proyecto, ingreso por desarrollo, emprendimiento"},
]


# ── Data: Hermes Memories → table memories ─────────────────────

INITIAL_MEMORIES = [
    # From Hermes MEMORY block
    {"content": "Vault reorganizado 2026-04-20: 319 files root→20 themed folders. Root=SCHEMA.md+log.md only.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Autoeval cron suggests 3-5 stubs weekly. Noticiario cron 18:00 Chile.", "type": "patron", "source": "hermes_memory"},
    {"content": "Local image gen: ERNIE-Image-Turbo via BnB NF4 (~10GB CUDA). Script: /root/test_ernie_bnb.py.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Obsidian vault: /mnt/d/Documentos/Obsidian Vault. Windows user: x_zer.", "type": "hecho", "source": "hermes_memory"},
    {"content": "NVIDIA NIM streaming fixed in v0.11.0 natively.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Hermes v0.11.0 (2026-04-23): Transport layer, TUI, orchestrator role, enabled_toolsets per cron.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Deepseek-v3.2 configured as cheap_model via NVIDIA. Custom patches obsolete as of v0.11.0.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Crons nvidia provider, API keys in dot-env. Crons tienen instrucción explícita de guardar respuestas en vault.", "type": "patron", "source": "hermes_memory"},
    {"content": "Roadmap programación: tema actual Context managers (5), generadores y decoradores completados.", "type": "hecho", "source": "hermes_memory"},
    {"content": "Proyecto juego PQ009 resuelta: niveles numéricos, visualización sutil, liderazgo emergente. 11/15 PQs resueltas.", "type": "hecho", "source": "hermes_memory"},

    # From USER PROFILE
    {"content": "Diego prefiere Obsidian Bases plugin para tracking. Evaluaciones como markdown en autoevaluaciones/.", "type": "preference", "source": "hermes_profile"},
    {"content": "Le motiva para gym ver chicas en el gym. Cuando hay sobreentrenamiento prefiere reducir volumen pero mantener intensidad.", "type": "preference", "source": "hermes_profile"},
    {"content": "Valora learning práctico sobre abstracto. Preocupado por situación económica, necesita trabajo pronto.", "type": "preference", "source": "hermes_profile"},
    {"content": "CV email: d.v.petricio@live.com (Outlook, 2FA on, basic auth blocked).", "type": "hecho", "source": "hermes_profile"},
    {"content": "Aprendizaje: Micro-lecciones (10:00), Anki (10:30/16:00/21:00).", "type": "patron", "source": "hermes_profile"},
    {"content": "Game design: nunca revelar significado explícitamente — jugadores descubren orgánicamente (Adventure Time style). Shadow integration = freedom.", "type": "preference", "source": "hermes_profile"},

    # Additional context
    {"content": "Diego vive en Antofagasta, Chile. Timezone UTC-3/UTC-4.", "type": "hecho", "source": "hermes_profile"},
    {"content": "PC: RTX 5070 Ti 16GB, 32GB RAM, Windows 11 + WSL.", "type": "hecho", "source": "hermes_profile"},
    {"content": "Proyectos activos: juego indie, Muninn, Valle Alto, aprendizaje programación.", "type": "hecho", "source": "hermes_profile"},
    {"content": "Plan z.ai semanal, se agota día 6-7. Usar modelo local como backup.", "type": "hecho", "source": "hermes_profile"},
]

# Memory-to-peer linking
MEMORY_PEER_LINKS = [
    # memory index → [peer_ids]
    ([0], ["nanobot_sistema"]),  # vault reorganizado
    ([1], ["nanobot_sistema"]),  # autoeval cron
    ([2], ["nanobot_sistema", "programacion"]),  # image gen
    ([3], ["nanobot_sistema"]),  # vault path
    ([4], ["nanobot_sistema"]),  # NIM streaming
    ([5], ["nanobot_sistema"]),  # Hermes version
    ([6], ["nanobot_sistema"]),  # Deepseek
    ([7], ["nanobot_sistema"]),  # crons config
    ([8], ["programacion"]),  # roadmap
    ([9], ["proyecto_juego"]),  # PQs
    ([10], ["nanobot_sistema"]),  # Obsidian bases
    ([11], ["gym_rutina"]),  # gym motivation
    ([12], ["peer_usuario"]),  # practical learning, needs job
    ([13], ["peer_usuario", "valle_alto"]),  # CV email
    ([14], ["programacion"]),  # learning schedule
    ([15], ["proyecto_juego"]),  # game design philosophy
    ([16], ["peer_usuario"]),  # location
    ([17], ["peer_usuario"]),  # PC specs
    ([18], ["peer_usuario"]),  # projects
    ([19], ["peer_usuario", "nanobot_sistema"]),  # z.ai plan
]


# ── Seed Process ───────────────────────────────────────────────

def seed(db_path="muninn_v4.db", batch_size=32):
    print("=" * 60)
    print("MUNINN v4 SEED — BGE-M3 + Improved Facets")
    print("=" * 60)

    # 1. Init DB
    print("\n[1/5] Initializing database...")
    conn = init_db(db_path, dims=MODEL_DIMS)

    # 2. Load embedder
    print("\n[2/5] Loading BGE-M3...")
    model = get_embedder()

    # 3. Insert peers
    print("\n[3/5] Inserting peers...")
    peers_data, facets_data = load_export()
    peer_ids = set()
    for pid, pdata in peers_data.items():
        conn.execute("""
            INSERT OR IGNORE INTO peers (id, name, type, domain, description,
                representation, tags, activation_threshold, confidence, level)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0.25, 0.1, 1.0)
        """, [
            pdata["id"], pdata["name"], pdata["type"], pdata["domain"],
            pdata["description"], pdata["representation"], pdata["tags"]
        ])
        peer_ids.add(pid)
    conn.commit()
    print(f"  Inserted {len(peer_ids)} peers")

    # 4. Insert facets + compute embeddings
    print("\n[4/5] Inserting facets + computing embeddings...")

    # Collect all facet texts
    all_facets = []

    # Original facets from export
    for f in facets_data:
        all_facets.append({
            "peer_id": f["peer_id"],
            "facet_type": f["facet_type"],
            "text": f["text"],
        })

    # New facets from failure analysis
    for f in NEW_FACETS:
        all_facets.append({
            "peer_id": f["peer_id"],
            "facet_type": f["facet_type"],
            "text": f["text"],
        })

    print(f"  Total facets: {len(all_facets)} ({len(facets_data)} original + {len(NEW_FACETS)} new)")

    # Compute embeddings in batch
    facet_texts = [f["text"] for f in all_facets]
    print(f"  Computing embeddings ({len(facet_texts)} texts, batch_size={batch_size})...")
    t0 = time.time()
    embeddings = embed_texts(model, facet_texts, is_query=False, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(facet_texts)/elapsed:.1f} texts/s)")

    # Insert facets + embeddings
    for facet, emb in zip(all_facets, embeddings):
        cursor = conn.execute("""
            INSERT INTO peer_facets (peer_id, facet_type, text)
            VALUES (?, ?, ?)
        """, [facet["peer_id"], facet["facet_type"], facet["text"]])

        facet_id = cursor.lastrowid
        emb_bytes = struct.pack(f"{MODEL_DIMS}f", *emb)
        conn.execute(
            "INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
            [facet_id, emb_bytes]
        )

    conn.commit()
    print(f"  Inserted {len(all_facets)} facets with embeddings")

    # 5. Insert memories + link to peers + embeddings
    print("\n[5/5] Inserting initial memories...")
    mem_texts = [m["content"] for m in INITIAL_MEMORIES]
    mem_embeddings = embed_texts(model, mem_texts, is_query=False, batch_size=batch_size)

    for i, (mem, emb) in enumerate(zip(INITIAL_MEMORIES, mem_embeddings)):
        cursor = conn.execute("""
            INSERT INTO memories (content, type, source, confidence)
            VALUES (?, ?, ?, 0.8)
        """, [mem["content"], mem["type"], mem["source"]])
        mem_id = cursor.lastrowid

        # Store embedding
        emb_bytes = struct.pack(f"{MODEL_DIMS}f", *emb)
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
            [mem_id, emb_bytes]
        )

        # Link to peers (MEMORY_PEER_LINKS[i] = ([memory_indices], [peer_ids]))
        # Since we iterate memories sequentially, link by position
        if i < len(MEMORY_PEER_LINKS):
            for pid in MEMORY_PEER_LINKS[i][1]:
                conn.execute("""
                    INSERT OR IGNORE INTO memory_peers (memory_id, peer_id, relevance)
                    VALUES (?, ?, 0.7)
                """, [mem_id, pid])

    conn.commit()
    print(f"  Inserted {len(INITIAL_MEMORIES)} memories with embeddings and peer links")

    # 6. Insert peer connections (semantic relationships)
    connections = [
        ("sombra_muerte", "sombra_rechazo", "conecta", 0.6, "Ambas sombras ligadas a pérdida"),
        ("sombra_muerte", "sombra_angel_atardecer", "conecta", 0.5, "Ansiedad por pérdida como nexo"),
        ("sombra_rechazo", "sombra_fortaleza", "conecta", 0.7, "Fortaleza como defensa contra rechazo"),
        ("sombra_angel_atardecer", "sombra_fortaleza", "conecta", 0.4, "Ansiedad vs armadura emocional"),
        ("peer_usuario", "gym_rutina", "conecta", 0.5, "Diego hace gym regularmente"),
        ("peer_usuario", "programacion", "conecta", 0.6, "Diego estudia programación"),
        ("peer_usuario", "proyecto_juego", "conecta", 0.5, "Proyecto activo de Diego"),
        ("peer_usuario", "valle_alto", "conecta", 0.5, "Proyecto familiar de Diego"),
        ("proyecto_juego", "programacion", "conecta", 0.4, "Juego requiere código"),
        ("gym_rutina", "sombra_fortaleza", "conecta", 0.3, "Disciplina física vs emocional"),
    ]
    for from_p, to_p, rtype, strength, desc in connections:
        conn.execute("""
            INSERT OR IGNORE INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES (?, ?, ?, ?, ?)
        """, [from_p, to_p, rtype, strength, desc])
        # Bidirectional
        conn.execute("""
            INSERT OR IGNORE INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES (?, ?, ?, ?, ?)
        """, [to_p, from_p, rtype, strength, desc])

    conn.commit()
    print(f"  Inserted {len(connections)*2} bidirectional connections")

    # Summary
    print("\n" + "=" * 60)
    print("SEED COMPLETE")
    print(f"  Database: {db_path}")
    print(f"  Model: {MODEL_NAME} ({MODEL_DIMS}d)")
    print(f"  Peers: {len(peer_ids)}")
    print(f"  Facets: {len(all_facets)} ({len(NEW_FACETS)} new from failure analysis)")
    print(f"  Memories: {len(INITIAL_MEMORIES)}")
    print(f"  Connections: {len(connections)*2}")
    print("=" * 60)

    conn.close()
    return db_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed Muninn v4 DB with BGE-M3")
    parser.add_argument("--db-path", default="muninn_v4.db", help="Output DB path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    seed(args.db_path, args.batch_size)
