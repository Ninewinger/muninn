"""Quick debug: why is the router returning empty?"""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from muninn.router import route
from muninn.db import get_connection
import struct

DB = "muninn_dream_test.db"
conn = get_connection(DB)

# Check peers
peers = conn.execute("SELECT id, name, activation_threshold FROM peers WHERE is_active=1").fetchall()
print(f"Peers: {len(peers)}")
for p in peers:
    emb = conn.execute("SELECT peer_id FROM peer_embeddings WHERE peer_id=?", [p["id"]]).fetchone()
    print(f"  {p['name']}: threshold={p['activation_threshold']}, has_embedding={bool(emb)}")

# Check if sqlite-vec works
try:
    dims = 384
    dummy = struct.pack(f"{dims}f", *([0.0]*dims))
    test = conn.execute("SELECT peer_id, distance FROM peer_embeddings WHERE embedding MATCH ? AND k = 4", [dummy]).fetchall()
    print(f"\nsqlite-vec query works: {len(test)} results")
    for r in test:
        sim = 1.0 - r["distance"]
        print(f"  {r['peer_id']}: sim={sim:.4f}")
except Exception as e:
    print(f"\nsqlite-vec query FAILED: {e}")

conn.close()

# Test route
result = route("Tengo miedo de que mi nona muera", db_path=DB)
print(f"\nRoute 'miedo nona muerte': {len(result)} activations")
for r in result:
    print(f"  {r['name']}: {r['similarity']:.4f}")
