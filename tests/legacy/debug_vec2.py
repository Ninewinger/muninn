"""Debug: Check how sqlite-vec stores and retrieves"""
import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import sqlite3
import sqlite_vec
import struct

# Create a fresh in-memory DB to test sqlite-vec
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

dims = 384

# Create virtual table
db.execute(f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{dims}])")

# Create a simple test vector (normalized)
import math
test_vec = [0.1] * dims
norm = math.sqrt(sum(x**2 for x in test_vec))
test_vec = [x/norm for x in test_vec]
test_bytes = struct.pack(f"{dims}f", *test_vec)

# Insert
db.execute("INSERT INTO vec_items(rowid, embedding) VALUES (1, ?)", [test_bytes])

# Query with same vector (should give distance ~0, sim ~1.0)
results = db.execute("SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? AND k = 1", [test_bytes]).fetchall()
for r in results:
    print(f"Self-match: rowid={r[0]}, distance={r[1]:.6f}, sim={1-r[1]:.6f}")

# Now try with a slightly different vector
test_vec2 = [0.11] * dims
norm2 = math.sqrt(sum(x**2 for x in test_vec2))
test_vec2 = [x/norm2 for x in test_vec2]
test_bytes2 = struct.pack(f"{dims}f", *test_vec2)

# Manual similarity
dot = sum(a*b for a,b in zip(test_vec, test_vec2))
print(f"Manual cosine sim: {dot:.6f}")

results2 = db.execute("SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? AND k = 1", [test_bytes2]).fetchall()
for r in results2:
    print(f"Similar vec: rowid={r[0]}, distance={r[1]:.6f}, sim={1-r[1]:.6f}")

db.close()
print("\nDone - if self-match distance is ~0, sqlite-vec works correctly")
