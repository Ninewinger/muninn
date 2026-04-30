"""Regenerar embeddings DB v3 con Qwen3-0.6B — via sqlite-vec API correcta."""
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
NEW_DIMS = 1024

print(f"Cargando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left', local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
model.eval()
print("Modelo cargado (CPU)")

def encode_batch(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.numpy()

# Connect with sqlite-vec
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
conn.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(conn)

# Check current dims
sample = conn.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
current_bytes = len(sample['embedding'])
current_dims = current_bytes // 4
print(f"DB actual: {current_dims}d ({current_bytes} bytes)")

if current_dims == NEW_DIMS:
    print("Ya está en 1024d, verificando si es MiniLM o Qwen3...")
    # Just re-embed to be safe

# Get all facets
facets = conn.execute("SELECT pf.id, pf.text FROM peer_facets pf ORDER BY pf.id").fetchall()
print(f"Regenerando {len(facets)} embeddings...")

# Drop and recreate the virtual table with correct dims
conn.execute("DROP TABLE IF EXISTS facet_embeddings")
conn.execute(f"CREATE VIRTUAL TABLE facet_embeddings USING vec0(facet_id INTEGER PRIMARY KEY, embedding FLOAT[{NEW_DIMS}])")
conn.commit()

# Insert in batches
done = 0
for i in range(0, len(facets), BATCH_SIZE):
    batch = facets[i:i + BATCH_SIZE]
    texts = [f['text'] for f in batch]
    ids = [f['id'] for f in batch]
    
    embs = encode_batch(texts)
    
    for j, fid in enumerate(ids):
        emb_bytes = struct.pack(f"{NEW_DIMS}f", *embs[j])
        conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)", [fid, emb_bytes])
    
    done += len(batch)
    conn.commit()
    print(f"  {done}/{len(facets)}", flush=True)

# Update config
conn.execute("DELETE FROM embedding_config WHERE key IN ('dimensions', 'model_name')")
conn.execute("INSERT INTO embedding_config (key, value) VALUES ('dimensions', ?)", [str(NEW_DIMS)])
conn.execute("INSERT INTO embedding_config (key, value) VALUES ('model_name', ?)", [MODEL_NAME])
conn.commit()

# Verify
sample = conn.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
final_dims = len(sample['embedding']) // 4
count = conn.execute("SELECT COUNT(*) as c FROM facet_embeddings").fetchone()['c']
conn.close()

print(f"\n✅ Migración completa: {count} embeddings de {final_dims}d con {MODEL_NAME}")
