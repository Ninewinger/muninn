import sys, os, struct
sys.path.insert(0, 'D:/github/muninn')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

from muninn.db import get_connection
from muninn.embeddings_v2 import embed, cosine_similarity, get_backend
from muninn.router_v2 import route

DB = 'D:/github/muninn/muninn.db'

conn = get_connection(DB)

# 1. Check thresholds
print("THRESHOLDS:")
peers = conn.execute("SELECT id, name, activation_threshold FROM peers").fetchall()
for p in peers:
    print(f"  {p['name']:25s} threshold={p['activation_threshold']}")

# 2. Raw similarity test — manual
print("\nRAW SIMILARITY TEST:")
backend = get_backend()
print(f"  Backend dims: {backend.dimensions}")

# Get a sample facet
facet = conn.execute("SELECT pf.id, pf.peer_id, pf.text FROM peer_facets LIMIT 1").fetchone()
row = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [facet['id']]).fetchone()
print(f"  Sample facet: [{facet['peer_id']}] {facet['text'][:50]}...")
print(f"  Embedding bytes: {len(row['embedding'])} = {len(row['embedding'])//4} dims")

# Unpack with backend dims
try:
    stored = list(struct.unpack(f"{backend.dimensions}f", row['embedding']))
    print(f"  Unpack OK: {len(stored)} values")
except Exception as e:
    print(f"  Unpack FAILED: {e}")
    # Try with actual byte count
    actual_dims = len(row['embedding']) // 4
    print(f"  Actual dims from bytes: {actual_dims}")
    stored = list(struct.unpack(f"{actual_dims}f", row['embedding']))
    print(f"  Unpack with actual: {len(stored)} values")

# Embed query
query_vec = embed("miedo a la muerte", is_query=True)
print(f"  Query vec: {len(query_vec)} dims")

if len(stored) == len(query_vec):
    sim = cosine_similarity(query_vec, stored)
    print(f"  Similarity: {sim:.4f}")
else:
    print(f"  MISMATCH: stored={len(stored)}, query={len(query_vec)}")

# 3. Try with ALL queries against a few known facets
print("\nDETAILED SCORES:")
queries = ["miedo a la muerte", "escribir codigo python", "ir al gym"]
for q in queries:
    qvec = embed(q, is_query=True)
    print(f"\n  Query: '{q}'")
    # Check top 5 facets by manual similarity
    facet_rows = conn.execute("""
        SELECT pf.id, pf.peer_id, pf.text 
        FROM peer_facets pf 
        LIMIT 10
    """).fetchall()
    scores = []
    for f in facet_rows:
        emb_row = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [f['id']]).fetchone()
        if not emb_row:
            continue
        actual_dims = len(emb_row['embedding']) // 4
        try:
            svec = list(struct.unpack(f"{actual_dims}f", emb_row['embedding']))
            if len(svec) == len(qvec):
                sim = cosine_similarity(qvec, svec)
                scores.append((f['peer_id'], f['text'][:30], sim))
        except:
            pass
    scores.sort(key=lambda x: x[2], reverse=True)
    for pid, txt, sim in scores[:3]:
        print(f"    {pid:25s} {txt:30s} sim={sim:.4f}")

conn.close()
