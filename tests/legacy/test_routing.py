"""Test completo de Muninn routing — diagnóstico."""
import sys, os, struct
sys.path.insert(0, 'D:/github/muninn')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

from muninn.db import get_connection
from muninn.embeddings_v2 import get_backend, embed

DB_PATH = 'D:/github/muninn/muninn.db'

# 1. Verificar datos en DB
print("=" * 50)
print("1. ESTADO DE LA DB")
conn = get_connection(DB_PATH)
peers = conn.execute("SELECT id, name, confidence FROM peers").fetchall()
print(f"   Peers: {len(peers)}")
for p in peers:
    print(f"     - {p['name']} (confidence: {p['confidence']})")

facets = conn.execute("SELECT id, peer_id, facet_type, text FROM peer_facets LIMIT 5").fetchall()
print(f"\n   Facetas (primeras 5 de 47):")
for f in facets:
    print(f"     - [{f['peer_id']}] {f['facet_type']}: {f['text'][:50]}...")

# 2. Verificar embeddings almacenados
print("\n" + "=" * 50)
print("2. EMBEDDINGS ALMACENADOS")
emb_count = conn.execute("SELECT COUNT(*) FROM facet_embeddings").fetchone()[0]
print(f"   Embeddings en DB: {emb_count}")

if emb_count > 0:
    sample = conn.execute("SELECT facet_id, LENGTH(embedding) as bytes_len FROM facet_embeddings LIMIT 1").fetchone()
    byte_len = sample['bytes_len']
    dims_stored = byte_len // 4  # float32 = 4 bytes
    print(f"   Tamaño por embedding: {byte_len} bytes = {dims_stored} dimensiones")

# 3. Verificar backend actual
print("\n" + "=" * 50)
print("3. BACKEND ACTUAL")
backend = get_backend()
print(f"   Modelo: {backend.model_name}")
print(f"   Dimensiones: {backend.dimensions}")

# 4. Verificar compatibilidad
print("\n" + "=" * 50)
print("4. COMPATIBILIDAD")
if emb_count > 0:
    if dims_stored != backend.dimensions:
        print(f"   ❌ INCOMPATIBLE: DB tiene {dims_stored}d pero backend genera {backend.dimensions}d")
        print(f"   Los embeddings necesitan ser regenerados con Qwen3")
        print(f"   Solución: correr seed_v2.py con el nuevo backend")
    else:
        print(f"   ✅ Compatible: {dims_stored}d = {backend.dimensions}d")

# 5. Test de embedding (sin DB)
print("\n" + "=" * 50)
print("5. TEST DE EMBEDDING")
test_vec = embed("escribir código python")
print(f"   Vector generado: {len(test_vec)} dimensiones")
print(f"   Primeros 5 valores: {test_vec[:5]}")

conn.close()
print("\n✅ Diagnóstico completo.")
