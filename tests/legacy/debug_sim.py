import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import sqlite3, struct, numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Load model
model_name = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left', local_files_only=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
model.eval()

def encode(texts):
    if isinstance(texts, str): texts = [texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return F.normalize(pooled, p=2, dim=1).numpy()

# Load DB
conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
conn.enable_load_extension(True)
import sqlite_vec; sqlite_vec.load(conn)

# Test: "Me siento ignorado" vs rechazo facet
query = encode(["Me siento ignorado, nadie me contesta"])[0]

facets = conn.execute("""
    SELECT pf.id, pf.text, pf.facet_type, p.name, p.domain
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id
    WHERE p.id = 'sombra_rechazo'
""").fetchall()

print("Rechazo facets vs 'Me siento ignorado':")
for f in facets:
    row = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [f['id']]).fetchone()
    stored = np.array(struct.unpack('1024f', row['embedding']))
    sim = float(np.dot(query, stored))
    print(f"  {f['facet_type']:12s} ({sim:.4f}) {f['text'][:60]}")

# Also check top-3 across ALL peers
print("\nTop-5 across ALL peers:")
all_facets = conn.execute("""
    SELECT pf.id, pf.text, pf.facet_type, p.name, p.domain
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id
""").fetchall()

scores = []
for f in all_facets:
    row = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [f['id']]).fetchone()
    stored = np.array(struct.unpack('1024f', row['embedding']))
    sim = float(np.dot(query, stored))
    scores.append((sim, f['name'], f['domain'], f['facet_type']))

scores.sort(reverse=True)
for sim, name, domain, ftype in scores[:5]:
    print(f"  {sim:.4f} {name} ({domain}) [{ftype}]")

conn.close()
