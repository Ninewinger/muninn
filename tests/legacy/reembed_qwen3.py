"""Re-embed all facets with Qwen3-Embedding-8B (384d → 1024d migration).

This script:
1. Opens existing muninn.db
2. Re-initializes schema for 1024d
3. Re-embeds all 47 facets with Qwen3
4. Verifies routing works
"""
import os, sys, struct, time

sys.path.insert(0, os.path.dirname(__file__))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ['HF_HOME'] = 'D:/hf_cache'

from muninn.db import init_db, get_connection, get_embedding_dims
from muninn.embeddings_v2 import embed

DB_PATH = os.path.join(os.path.dirname(__file__), "muninn.db")

def reembed():
    print("=" * 70)
    print("  MUNINN — Re-embedding 384d → 1024d (Qwen3)")
    print("=" * 70)

    # 1. Leer facetas existentes
    conn_old = get_connection(DB_PATH)
    facets = conn_old.execute("""
        SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, pf.weight
        FROM peer_facets pf
    """).fetchall()
    peers = conn_old.execute("SELECT * FROM peers").fetchall()
    connections = conn_old.execute("SELECT * FROM connections").fetchall()
    print(f"\n  Datos leídos: {len(peers)} peers, {len(facets)} facetas, {len(connections)} conexiones")

    # Guardar datos en memoria
    facet_data = [dict(f) for f in facets]
    peer_data = [dict(p) for p in peers]
    conn_data = [dict(c) for c in connections]
    conn_old.close()

    # 2. Recrear DB con 1024d
    print(f"\n  Recreando DB con 1024d...")
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)
        print(f"  DB anterior eliminada")

    conn = init_db(DB_PATH, dimensions=1024)
    dims = get_embedding_dims(conn)
    print(f"  Dimensiones configuradas: {dims}")

    # 3. Restaurar peers
    print(f"\n  Restaurando peers...")
    for p in peer_data:
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
    print(f"  ✅ {len(peer_data)} peers restaurados")

    # 4. Re-embed facets con Qwen3
    print(f"\n  Re-embeddiendo {len(facet_data)} facetas con Qwen3...")
    start = time.time()
    total = len(facet_data)

    for i, f in enumerate(facet_data):
        # Insert facet
        conn.execute("""
            INSERT INTO peer_facets (id, peer_id, facet_type, text, weight)
            VALUES (?, ?, ?, ?, ?)
        """, [f["id"], f["peer_id"], f["facet_type"], f["text"], f["weight"]])

        # Generate Qwen3 embedding
        vector = embed(f["text"])
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)",
                     [f["id"], vec_bytes])

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"     [{i+1}/{total}] {f['peer_id']:25s} {f['facet_type']:12s} ({rate:.1f} facetas/s)")

    conn.commit()
    elapsed = time.time() - start
    print(f"  ✅ {total} facetas re-embeddias en {elapsed:.1f}s")

    # 5. Restaurar conexiones
    print(f"\n  Restaurando conexiones...")
    for c in conn_data:
        conn.execute("""
            INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES (?, ?, ?, ?, ?)
        """, [c["from_peer_id"], c["to_peer_id"], c["relation_type"],
              c["strength"], c["description"]])
    conn.commit()
    print(f"  ✅ {len(conn_data)} conexiones restauradas")

    # 6. Verificación
    print(f"\n{'─' * 70}")
    print("  VERIFICACIÓN:")
    count_p = conn.execute("SELECT COUNT(*) FROM peers").fetchone()[0]
    count_f = conn.execute("SELECT COUNT(*) FROM peer_facets").fetchone()[0]
    count_e = conn.execute("SELECT COUNT(*) FROM facet_embeddings").fetchone()[0]
    count_c = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    print(f"  Peers: {count_p}, Facetas: {count_f}, Embeddings: {count_e}, Conexiones: {count_c}")

    # 7. Test de routing
    print(f"\n  Test de routing:")
    from muninn.router_v2 import route
    test_queries = [
        "escribir código python",
        "cómo está mi abuela",
        "quiero ir al gimnasio",
        "diseñar sistema de combate",
    ]
    for q in test_queries:
        results = route(q, db_path=DB_PATH)
        if results:
            top = results[0]
            print(f"     '{q}' → {top['peer_name']} ({top['total_score']:.3f}) via [{top['facet_type']}]")
        else:
            print(f"     '{q}' → sin activación")

    conn.close()
    print(f"\n  ✅ Migración completada! DB: {DB_PATH}")


if __name__ == "__main__":
    reembed()
