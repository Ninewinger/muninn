"""Test C10: Reranker bge-reranker-v2-m3 sobre sistema Disco Elysium DB v3.

Flujo:
1. Embeddings → top 50 candidatos
2. Reranker (cross-encoder) → reordena
3. Top-3 final
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '0'  # Need to download reranker

import sqlite3, struct, numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

# ═══════════════════════════════════════════
# EMBEDDING MODEL (Qwen3-0.6B)
# ═══════════════════════════════════════════
model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Cargando {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left', local_files_only=True)
emb_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
emb_model.eval()
print("Embedding model cargado")

def encode(texts, batch_size=4):
    if isinstance(texts, str): texts = [texts]
    all_out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            out = emb_model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            pooled = F.normalize(pooled, p=2, dim=1)
            all_out.append(pooled.numpy())
    return all_out[0] if len(all_out) == 1 else np.vstack(all_out)

# ═══════════════════════════════════════════
# RERANKER (BGE-reranker-v2-m3)
# ═══════════════════════════════════════════
print("Cargando reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
print("Reranker cargado")

# ═══════════════════════════════════════════
# LOAD DB
# ═══════════════════════════════════════════
conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
conn.enable_load_extension(True)
import sqlite_vec; sqlite_vec.load(conn)

facets = conn.execute("""
    SELECT pf.id, pf.peer_id, pf.text, pf.facet_type, p.name, p.domain, p.activation_threshold
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id WHERE p.is_active = 1
""").fetchall()

facet_data = {}
for f in facets:
    row = conn.execute("SELECT embedding FROM facet_embeddings WHERE facet_id = ?", [f['id']]).fetchone()
    if row:
        emb = np.array(struct.unpack('1024f', row['embedding']), dtype=np.float32)
        facet_data[f['id']] = {
            'emb': emb, 'peer_id': f['peer_id'], 'name': f['name'],
            'domain': f['domain'], 'threshold': f['activation_threshold'],
            'text': f['text'], 'facet_type': f['facet_type']
        }

print(f"DB: {len(facet_data)} facetas cargadas\n")

# ═══════════════════════════════════════════
# TEST PHRASES
# ═══════════════════════════════════════════
test_phrases = [
    ("Me siento ignorado, nadie me contesta", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona esta mejor pero sigue en cama", "sombra_muerte"),
    ("Si lloro es debilidad", "sombra_fortaleza"),
    ("Tengo miedo de perder a alguien cercano", "sombra_muerte"),
    ("Una chica que me gustaba me dejo en visto", "sombra_rechazo"),
    ("La ansiedad no me deja dormir", "sombra_angel_atardecer"),
    ("El gimnasio es mi templo", "sombra_fortaleza"),
    ("Siento que no encajo en ninguna parte", "sombra_rechazo"),
    ("Necesito ayuda pero no puedo pedirla", "sombra_fortaleza"),
    ("Me da terror abrirme emocionalmente", "sombra_rechazo"),
    ("Cierro los ojos y veo oscuridad", "sombra_muerte"),
    ("La tecnologia me estresa", "sombra_angel_atardecer"),
    ("Las piernas que antes eran mi debilidad", "sombra_fortaleza"),
    ("El grupo hace planes sin mi", "sombra_rechazo"),
    ("El doctor dijo que hay que hacer mas examenes", "sombra_muerte"),
    ("Saque un nuevo PR en sentadilla hoy", "sombra_fortaleza"),
    ("Me borraron del grupo de WhatsApp", "sombra_rechazo"),
    ("El funeral de mi tio me dejo pensando", "sombra_muerte"),
    ("Me quede scrolleando hasta las 3am", "sombra_angel_atardecer"),
    ("Me dieron un abrazo y casi me echo a llorar", "sombra_fortaleza"),
    ("Mi ex novia cambio su estado", "sombra_rechazo"),
    ("Sueno que se me caen los dientes", "sombra_muerte"),
    ("Como deberia funcionar el sistema de combate?", "proyecto_juego"),
    ("Tengo un error en la linea 42", "programacion"),
    ("Como funcionan los decoradores en Python", "programacion"),
    ("Configura un recordatorio para manana", "nanobot_sistema"),
    ("Instalar una skill nueva desde clawhub", "nanobot_sistema"),
    ("Las patentes mineras estan vigentes", "valle_alto"),
    ("Hoy toca pierna, bulgaras con 35 kilos", "gym_rutina"),
    ("Hola weon, como estai", "casual_social"),
    ("Quien soy yo como asistente?", "peer_identidad"),
    ("Que hora es?", "casual_social"),
    ("Necesito ejecutar un comando en la terminal", "peer_herramientas"),
    ("Antes de empezar, cual es la vision del proyecto?", "peer_operativo"),
]

# ═══════════════════════════════════════════
# COMPARISON: Without vs With Reranker
# ═══════════════════════════════════════════
print(f"{'Frase':<47s} {'Sin Reranker':<35s} {'Con Reranker':<35s} {'OK?'}")
print("-" * 130)

total = len(test_phrases)

for mode in ["baseline", "reranker"]:
    r1, r3, ra = 0, 0, 0
    
    for text, expected in test_phrases:
        q = encode([text])[0]
        
        # Step 1: Get all peer scores (embedding)
        peer_best = {}
        for fid, fd in facet_data.items():
            sim = float(np.dot(q, fd['emb']))
            pid = fd['peer_id']
            if pid not in peer_best or sim > peer_best[pid]['score']:
                peer_best[pid] = {'score': sim, 'name': fd['name'], 'domain': fd['domain'], 'text': fd['text']}
        
        # Filter by threshold
        filtered = {pid: d for pid, d in peer_best.items() if d['score'] >= 0.25}
        candidates = sorted(filtered.items(), key=lambda x: -x[1]['score'])
        
        if mode == "reranker" and len(candidates) > 3:
            # Step 2: Rerank top candidates
            top_n = min(15, len(candidates))  # Rerank top 15
            pairs = [(text, c[1]['text']) for c in candidates[:top_n]]
            scores = reranker.predict(pairs)
            
            # Re-sort by reranker score
            reranked = list(zip(candidates[:top_n], scores))
            reranked.sort(key=lambda x: -x[1])
            final = [(pid, {'name': d['name'], 'score': float(s)}) for (pid, d), s in reranked]
        else:
            final = candidates
        
        top3 = final[:3]
        top_ids = [a[0] for a in top3]
        any_ids = [a[0] for a in final]
        
        if top3 and top3[0][0] == expected: r1 += 1
        if expected in top_ids: r3 += 1
        if expected in any_ids: ra += 1
    
    label = "SIN Reranker" if mode == "baseline" else "CON Reranker"
    print(f"\n  {label}:")
    print(f"    R@1={r1}/{total}({r1/total:.1%})  R@3={r3}/{total}({r3/total:.1%})  R@any={ra}/{total}({ra/total:.1%})")

# Now show per-phrase comparison
print(f"\n{'='*130}")
print(f"  DETALLE POR FRASE")
print(f"{'='*130}")

for text, expected in test_phrases:
    q = encode([text])[0]
    
    # Embedding scores
    peer_best = {}
    for fid, fd in facet_data.items():
        sim = float(np.dot(q, fd['emb']))
        pid = fd['peer_id']
        if pid not in peer_best or sim > peer_best[pid]['score']:
            peer_best[pid] = {'score': sim, 'name': fd['name'], 'text': fd['text']}
    
    filtered = {pid: d for pid, d in peer_best.items() if d['score'] >= 0.25}
    candidates = sorted(filtered.items(), key=lambda x: -x[1]['score'])
    
    # Without reranker
    top3_base = [c[0] for c in candidates[:3]]
    
    # With reranker
    if len(candidates) > 3:
        top_n = min(15, len(candidates))
        pairs = [(text, c[1]['text']) for c in candidates[:top_n]]
        scores = reranker.predict(pairs)
        reranked = list(zip(candidates[:top_n], scores))
        reranked.sort(key=lambda x: -x[1])
        top3_rerank = [r[0][0] for r in reranked[:3]]
    else:
        top3_rerank = top3_base
    
    base_str = ",".join(f"{pid[:8]}" for pid in top3_base[:3])
    rerank_str = ",".join(f"{pid[:8]}" for pid in top3_rerank[:3])
    
    hit_base = "✅" if expected in top3_base else "❌"
    hit_rerank = "✅" if expected in top3_rerank else "❌"
    improved = "↑" if expected not in top3_base and expected in top3_rerank else "↓" if expected in top3_base and expected not in top3_rerank else "="
    
    print(f"  {hit_base}{hit_rerank}{improved} {text[:42]:<42s} → {base_str:>30s} | {rerank_str}")

conn.close()
