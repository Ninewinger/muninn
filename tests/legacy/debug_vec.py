"""Debug: sqlite-vec matching issue"""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from muninn.embeddings import embed
from muninn.db import get_connection
import struct

DB = "muninn_dream_test.db"
dims = 384

# Get a real embedding
text = "Tengo miedo de la muerte"
query_vec = embed(text)
query_bytes = struct.pack(f"{dims}f", *query_vec)
print(f"Query vector norm: {sum(x**2 for x in query_vec)**0.5:.4f}")
print(f"Query bytes length: {len(query_bytes)}")

conn = get_connection(DB)

# Get stored embedding for sombra_muerte
row = conn.execute("SELECT peer_id, embedding FROM peer_embeddings WHERE peer_id='sombra_muerte'").fetchone()
raw = row["embedding"]
stored_vec = struct.unpack(f"{dims}f", raw)
print(f"\nStored vector for {row['peer_id']}:")
print(f"  Norm: {sum(x**2 for x in stored_vec)**0.5:.4f}")
print(f"  Bytes length: {len(raw)}")

# Manual cosine similarity
dot = sum(a*b for a,b in zip(query_vec, stored_vec))
print(f"\nManual cosine similarity: {dot:.4f}")

# Now test sqlite-vec query
print("\n--- sqlite-vec query ---")
results = conn.execute(
    "SELECT peer_id, distance FROM peer_embeddings WHERE embedding MATCH ? AND k = 4",
    [query_bytes]
).fetchall()
for r in results:
    sim = 1.0 - r["distance"]
    print(f"  {r['peer_id']}: distance={r['distance']:.6f}, sim={sim:.6f}")

# Try with query_bytes as blob explicitly
print("\n--- sqlite-vec with explicit blob ---")
import sqlite3
results2 = conn.execute(
    "SELECT peer_id, distance FROM peer_embeddings WHERE embedding MATCH ? AND k = 4",
    [sqlite3.Binary(query_bytes)]
).fetchall()
for r in results2:
    sim = 1.0 - r["distance"]
    print(f"  {r['peer_id']}: distance={r['distance']:.6f}, sim={sim:.6f}")

conn.close()
