import sys, os, struct
sys.path.insert(0, 'D:/github/muninn')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

from muninn.db import get_connection
from muninn.embeddings_v2 import embed, cosine_similarity, get_backend

DB = 'D:/github/muninn/muninn.db'
conn = get_connection(DB)

# 1. Check columns
print("PEER_FACETS columns:")
for col in conn.execute("PRAGMA table_info(peer_facets)").fetchall():
    print(f"  {col['name']} ({col['type']})")

print("\nFACET_EMBEDDINGS columns:")
for col in conn.execute("PRAGMA table_info(facet_embeddings)").fetchall():
    print(f"  {col['name']} ({col['type']})")

# 2. Sample facets
print("\nFIRST 5 FACETS:")
facets = conn.execute("SELECT id, peer_id, facet_type, text FROM peer_facets LIMIT 5").fetchall()
for f in facets:
    print(f"  id={f['id']} peer={f['peer_id']} type={f['facet_type']} text={f['text'][:50]}")

# 3. Check if embeddings match facet IDs
print("\nEMBEDDING IDs vs FACET IDs:")
emb_ids = [r['facet_id'] for r in conn.execute("SELECT facet_id FROM facet_embeddings LIMIT 5").fetchall()]
facet_ids = [r['id'] for r in conn.execute("SELECT id FROM peer_facets LIMIT 5").fetchall()]
print(f"  Facet IDs: {facet_ids}")
print(f"  Embedding IDs: {emb_ids}")

# 4. Manual embedding test
print("\nMANUAL EMBEDDING TEST:")
backend = get_backend()
print(f"  Backend: {backend.model_name} ({backend.dimensions}d)")

# Pick facet about "muerte"
facet = conn.execute("SELECT id, peer_id, text FROM peer_facets WHERE peer_id='sombra_muerte' LIMIT 1").fetchone()
if facet:
    print(f"  Facet: id={facet['id']} text={facet['text'][:60]}")
    emb = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [facet['id']]).fetchone()
    if emb:
        byte_len = len(emb['embedding'])
        actual_dims = byte_len // 4
        print(f"  Embedding bytes: {byte_len} = {actual_dims}d")
        
        # Try unpack with backend dims
        try:
            stored = list(struct.unpack(f"{backend.dimensions}f", emb['embedding']))
            print(f"  Unpack with {backend.dimensions}d: OK")
        except struct.error as e:
            print(f"  Unpack with {backend.dimensions}d: FAILED ({e})")
            print(f"  Need {actual_dims}d but backend is {backend.dimensions}d")
            stored = list(struct.unpack(f"{actual_dims}f", emb['embedding']))
        
        # Embed query and compare
        query_vec = embed("miedo a la muerte", is_query=True)
        print(f"  Query dims: {len(query_vec)}, Stored dims: {len(stored)}")
        
        if len(query_vec) == len(stored):
            sim = cosine_similarity(query_vec, stored)
            print(f"  COSINE SIMILARITY: {sim:.6f}")
        else:
            print(f"  DIMENSION MISMATCH — cannot compare!")
            print(f"  This is the root cause: stored embeddings are {len(stored)}d but backend generates {len(query_vec)}d")
    else:
        print(f"  No embedding found for facet_id={facet['id']}")
else:
    print("  No muerte facets found")

conn.close()
