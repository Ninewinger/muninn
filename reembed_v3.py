"""Re-embed all 72 facets from muninn_v3.db with Qwen3-8B (1024d).

The original embeddings were made with Qwen3-0.6B — incompatible space.
This regenerates them with Qwen3-8B which is the current backend.
"""
import os, sys, struct, time

sys.path.insert(0, os.path.dirname(__file__))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ['HF_HOME'] = 'D:/hf_cache'

from muninn.db import init_db, get_connection
from muninn.embeddings_v2 import embed

DB_PATH = os.path.join(os.path.dirname(__file__), "muninn.db")

def reembed():
    print("=" * 70)
    print("  MUNINN — Re-embedding 72 facetas con Qwen3-8B")
    print("=" * 70)

    # 1. Read all data from current DB (muninn_v3.db copied as muninn.db)
    conn_old = get_connection(DB_PATH)

    peers = [dict(r) for r in conn_old.execute("SELECT * FROM peers").fetchall()]
    facets = [dict(r) for r in conn_old.execute("SELECT * FROM peer_facets").fetchall()]
    connections = [dict(r) for r in conn_old.execute("SELECT * FROM connections").fetchall()]

    # Get embedding_config
    config = {}
    for row in conn_old.execute("SELECT key, value FROM embedding_config").fetchall():
        config[row['key']] = row['value']

    conn_old.close()

    print(f"\n  Datos leídos: {len(peers)} peers, {len(facets)} facetas, {len(connections)} conexiones")

    # 2. Recreate DB with 1024d (Qwen3-8B)
    print(f"\n  Recreando DB con 1024d...")
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)

    conn = init_db(DB_PATH, dimensions=1024)

    # Restore config
    for k, v in config.items():
        if k != 'dimensions':  # already set by init_db
            conn.execute("UPDATE embedding_config SET value = ? WHERE key = ?", [v, k])
    conn.commit()

    # 3. Restore peers
    print(f"\n  Restaurando {len(peers)} peers...")
    for p in peers:
        conn.execute("""
            INSERT INTO peers (id, name, type, domain, description, representation,
                             confidence, activation_threshold, level, max_activations, tags,
                             is_active, activation_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            p["id"], p["name"], p["type"], p["domain"],
            p["description"], p["representation"], p["confidence"],
            p["activation_threshold"], p["level"], p["max_activations"],
            p["tags"], p.get("is_active", 1), p.get("activation_count", 0),
            p.get("metadata"),
        ])
    conn.commit()
    print(f"  ✅ {len(peers)} peers restaurados")

    # 4. Re-embed facets with Qwen3-8B
    print(f"\n  Re-embeddiendo {len(facets)} facetas con Qwen3-8B...")
    start = time.time()

    for i, f in enumerate(facets):
        # Insert facet
        conn.execute("""
            INSERT INTO peer_facets (id, peer_id, facet_type, text, weight, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [f["id"], f["peer_id"], f["facet_type"], f["text"],
              f["weight"], f.get("created_at"), f.get("updated_at")])

        # Generate embedding with Qwen3-8B
        vector = embed(f["text"])
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
                     [f["id"], vec_bytes])

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (elapsed / (i + 1)) * (len(facets) - i - 1) if i > 0 else 0
        print(f"     [{i+1:2d}/{len(facets)}] {f['peer_id']:25s} {f['facet_type']:12s} "
              f"({rate:.1f}/s ETA:{eta:.0f}s)")

    conn.commit()
    elapsed = time.time() - start
    print(f"  ✅ {len(facets)} facetas re-embeddias en {elapsed:.1f}s")

    # 5. Restore connections
    print(f"\n  Restaurando conexiones...")
    for c in connections:
        conn.execute("""
            INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES (?, ?, ?, ?, ?)
        """, [c["from_peer_id"], c["to_peer_id"], c["relation_type"],
              c["strength"], c["description"]])
    conn.commit()
    print(f"  ✅ {len(connections)} conexiones restauradas")

    # 6. Verify routing
    print(f"\n{'─' * 70}")
    print("  TEST DE ROUTING:")
    from muninn.router_v2 import route
    test_queries = [
        ("escribir codigo python", "Programación"),
        ("como me llamo", "Usuario"),
        ("quien soy", "Identidad"),
        ("que herramientas tengo", "Herramientas"),
        ("diseñar combate", "Proyecto Juego"),
        ("miedo a la muerte", "Muerte"),
        ("ir al gym", "Gym/Rutina"),
        ("que skills tienes", "Skills"),
        ("mi abuela esta enferma", "Muerte"),
        ("me rechazaron", "Rechazo"),
    ]
    print(f"\n  {'Query':35s} {'→ Peer':20s} {'Score':>6s} {'Expected':>15s} {'OK':>4s}")
    print(f"  {'-'*80}")
    for q, expected in test_queries:
        results = route(q, db_path=DB_PATH)
        if results:
            top = results[0]
            ok = "✅" if expected.lower() in top['peer_name'].lower() else "❌"
            print(f"  {q:35s} {top['peer_name']:20s} {top['total_score']:6.3f} {expected:>15s} {ok:>4s}")
        else:
            print(f"  {q:35s} {'SIN ACTIVACION':20s} {'':>6s} {expected:>15s}    ❌")

    # Summary
    count_p = conn.execute("SELECT COUNT(*) FROM peers").fetchone()[0]
    count_f = conn.execute("SELECT COUNT(*) FROM peer_facets").fetchone()[0]
    count_c = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    print(f"\n  Final: {count_p} peers, {count_f} facetas, {count_c} conexiones")

    conn.close()
    print(f"\n  ✅ Migración completada! DB: {DB_PATH}")


if __name__ == "__main__":
    reembed()
