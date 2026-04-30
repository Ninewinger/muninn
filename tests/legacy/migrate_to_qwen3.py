"""Regenerar embeddings de la DB v3 con Qwen3-0.6B (384d → 1024d).
Batches de 4 para no saturar memoria."""
import sys, os, struct
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DB_PATH = r"D:\github\muninn\muninn_v3.db"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 4

print(f"Cargando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left', local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
model.eval()
print("Modelo cargado (CPU)")

def encode_batch(texts):
    """Encode texts with mean pooling."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.numpy()

# Connect to DB
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
except: pass

# Get all facets
facets = conn.execute("SELECT id, text FROM peer_facets ORDER BY id").fetchall()
print(f"Facetas a regenerar: {len(facets)}")

# Delete old embeddings
conn.execute("DELETE FROM facet_embeddings")
conn.commit()

# Re-generate in batches
done = 0
for i in range(0, len(facets), BATCH_SIZE):
    batch = facets[i:i + BATCH_SIZE]
    texts = [f['text'] for f in batch]
    ids = [f['id'] for f in batch]
    
    embs = encode_batch(texts)
    
    for j, fid in enumerate(ids):
        emb_bytes = struct.pack("1024f", *embs[j])
        conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)", [fid, emb_bytes])
    
    done += len(batch)
    print(f"  {done}/{len(facets)}...", flush=True)

# Update config
conn.execute("DELETE FROM embedding_config WHERE key = 'dimensions'")
conn.execute("INSERT INTO embedding_config (key, value) VALUES ('dimensions', '1024')")
conn.execute("DELETE FROM embedding_config WHERE key = 'model_name'")
conn.execute("INSERT INTO embedding_config (key, value) VALUES ('model_name', ?)", [MODEL_NAME])

conn.commit()
conn.close()

# Verify
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec; sqlite_vec.load(conn)
except: pass
sample = conn.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
dims = len(struct.unpack("1024f", sample['embedding']))
count = conn.execute("SELECT COUNT(*) as c FROM facet_embeddings").fetchone()['c']
conn.close()

print(f"\n✅ Migración completa: {count} embeddings de {dims}d con {MODEL_NAME}")
