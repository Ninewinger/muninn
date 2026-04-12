"""Debug: sqlite-vec with TEXT primary key vs the actual DB"""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import sqlite3
import sqlite_vec
import struct
from muninn.embeddings import embed

# 1) Test with fresh in-memory DB using TEXT pk (like schema.sql)
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
db.row_factory = sqlite3.Row

dims = 384
db.execute(f"CREATE VIRTUAL TABLE test_peers USING vec0(peer_id TEXT PRIMARY KEY, embedding FLOAT[{dims}])")

# Insert with text key
text = "miedo a la muerte"
vec = embed(text)
vec_bytes = struct.pack(f"{dims}f", *vec)
db.execute("INSERT INTO test_peers(peer_id, embedding) VALUES (?, ?)", ["sombra_muerte", vec_bytes])

# Query with same text
results = db.execute("SELECT peer_id, distance FROM test_peers WHERE embedding MATCH ? AND k = 1", [vec_bytes]).fetchall()
for r in results:
    print(f"In-memory TEXT PK: peer_id={r['peer_id']}, distance={r['distance']:.6f}, sim={1-r['distance']:.6f}")

db.close()

# 2) Now test with the actual test DB
conn = sqlite3.connect("muninn_dream_test.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
conn.row_factory = sqlite3.Row

# Query
results2 = conn.execute("SELECT peer_id, distance FROM peer_embeddings WHERE embedding MATCH ? AND k = 4", [vec_bytes]).fetchall()
for r in results2:
    print(f"Test DB: peer_id={r['peer_id']}, distance={r['distance']:.6f}, sim={1-r['distance']:.6f}")

# Check what's in the table
print("\n--- Table info ---")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'").fetchall()
print(f"Tables: {[t['name'] for t in tables]}")

# Check row count
try:
    count = conn.execute("SELECT count(*) FROM peer_embeddings").fetchone()
    print(f"peer_embeddings rows: {count}")
except:
    print("Can't count virtual table")

# Check the actual stored data
row = conn.execute("SELECT peer_id, embedding FROM peer_embeddings WHERE peer_id = 'sombra_muerte'").fetchone()
if row:
    raw = row["embedding"]
    stored = struct.unpack(f"{dims}f", raw)
    # Compare
    dot = sum(a*b for a,b in zip(vec, stored))
    print(f"\nDirect comparison (query vs stored sombra_muerte):")
    print(f"  Manual cosine: {dot:.6f}")
    print(f"  Stored norm: {sum(x**2 for x in stored)**0.5:.6f}")
    print(f"  Query norm: {sum(x**2 for x in vec)**0.5:.6f}")
else:
    print("\nNo row found for sombra_muerte!")

conn.close()
