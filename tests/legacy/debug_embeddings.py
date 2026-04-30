"""Debug: are embeddings actually non-zero?"""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from muninn.embeddings import embed
from muninn.db import get_connection
import struct

# Test embed function
print("Testing embed()...")
v = embed("Tengo miedo de la muerte")
print(f"  Vector length: {len(v)}")
print(f"  First 5 values: {v[:5]}")
print(f"  Norm: {sum(x**2 for x in v)**0.5:.4f}")
print(f"  Are all zeros? {all(x == 0 for x in v)}")

# Check what's actually stored in DB
conn = get_connection("muninn_dream_test.db")
row = conn.execute("SELECT peer_id, embedding FROM peer_embeddings LIMIT 1").fetchone()
if row:
    raw = row["embedding"]
    dims = len(raw) // 4
    vec = struct.unpack(f"{dims}f", raw)
    print(f"\nDB embedding for {row['peer_id']}: dims={dims}")
    print(f"  First 5: {vec[:5]}")
    print(f"  Are all zeros? {all(x == 0 for x in vec)}")
else:
    print("No embeddings in DB!")
conn.close()
