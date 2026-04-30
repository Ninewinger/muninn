import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import sqlite3, struct, numpy as np
sys.path.insert(0, r'D:\github\muninn')
from muninn.embeddings_v2 import get_backend

print("Cargando backend...")
backend = get_backend()
print(f"Modelo: {backend.model_name}, dims: {backend.dimensions}")

conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
except: pass

facets = conn.execute('''SELECT pf.id, pf.peer_id, pf.text, p.name, p.domain
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id WHERE p.is_active = 1''').fetchall()

# Load embeddings
facet_data = {}
for f in facets:
    row = conn.execute('SELECT embedding FROM facet_embeddings WHERE facet_id = ?', [f['id']]).fetchone()
    if row:
        emb = np.array(struct.unpack('1024f', row['embedding']), dtype=np.float32)
        facet_data[f['id']] = {'emb': emb, 'peer_id': f['peer_id'], 'name': f['name'], 'domain': f['domain']}

print(f"Facetas: {len(facet_data)}")

instruction = "Dado un mensaje de un usuario, identifica que dominio de su vida esta activando"

tests = [
    "Me siento ignorado, nadie me contesta",
    "Tengo un KeyError en python",
    "Configura un recordatorio para manana",
    "Hola weon como estai",
    "Quien soy yo como asistente?",
    "Como va el combate del juego?",
    "Voy a cargar creatina hoy",
]

for text in tests:
    q = backend.embed([text], is_query=True, instruction=instruction)
    q = np.array(q).flatten()
    
    peer_best = {}
    for fid, fd in facet_data.items():
        sim = float(np.dot(q, fd['emb']))
        pid = fd['peer_id']
        if pid not in peer_best or sim > peer_best[pid]['score']:
            peer_best[pid] = {'score': sim, 'name': fd['name'], 'domain': fd['domain']}
    
    sorted_all = sorted(peer_best.items(), key=lambda x: -x[1]['score'])
    top5 = sorted_all[:5]
    top_str = " | ".join(f"{a[1]['name']}({a[1]['score']:.3f})" for a in top5)
    print(f"  {text[:45]:<45s} → {top_str}")

conn.close()
