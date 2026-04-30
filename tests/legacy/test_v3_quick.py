"""Test rápido DB v3 — verificar que los 15 peers activan correctamente."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import sqlite3, struct, numpy as np
from sentence_transformers import SentenceTransformer

print("Cargando modelo...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                            cache_folder=r"C:\Users\x_zer\.cache\huggingface")

conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
except Exception:
    pass

facets = conn.execute('''
    SELECT pf.id, pf.peer_id, pf.facet_type, pf.text, p.name, p.domain, p.activation_threshold
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id
    WHERE p.is_active = 1
''').fetchall()

# Load facet embeddings
facet_data = {}
for f in facets:
    row = conn.execute('SELECT embedding FROM facet_embeddings WHERE facet_id = ?', [f['id']]).fetchone()
    if row:
        # Try 1024d first, then 384d
        try:
            emb = np.array(struct.unpack('1024f', row['embedding']), dtype=np.float32)
        except struct.error:
            emb = np.array(struct.unpack('384f', row['embedding']), dtype=np.float32)
        facet_data[f['id']] = {
            'emb': emb, 'peer_id': f['peer_id'], 'type': f['facet_type'],
            'name': f['name'], 'domain': f['domain'], 'threshold': f['activation_threshold']
        }

print(f"Facetas cargadas: {len(facet_data)}")
dims = list(facet_data.values())[0]['emb'].shape[0]
print(f"Dimensiones DB: {dims}")

# But our model produces 384d. If DB has 1024d, we need to re-encode facets.
if dims != 384:
    print(f"DB tiene {dims}d pero modelo produce 384d. Re-encodeando facetas con MiniLM...")
    for fid, fd in facet_data.items():
        text = conn.execute('SELECT text FROM peer_facets WHERE id = ?', [fid]).fetchone()['text']
        fd['emb'] = model.encode(text, normalize_embeddings=True)
    print("Facetas re-encodeadas.")

tests = [
    ("Hola nanobot, como estai", ["casual_social", "peer_identidad"]),
    ("Que hora es?", ["casual_social", "peer_usuario"]),
    ("Necesito ejecutar un comando en la terminal", ["peer_herramientas"]),
    ("Configura un recordatorio para manana", ["peer_operativo", "peer_herramientas"]),
    ("Toma un screenshot de mi pantalla", ["peer_skills"]),
    ("Me siento ignorado", ["sombra_rechazo"]),
    ("Tengo un KeyError en python", ["programacion"]),
    ("Como va el sistema de combate del juego?", ["proyecto_juego"]),
    ("Quiero loggear mi entreno de piernas", ["gym_rutina"]),
    ("Quien soy yo como asistente?", ["peer_identidad"]),
    ("Antes de empezar, cual es la vision del proyecto?", ["peer_operativo"]),
    ("Voy a cargar creatina hoy", ["sombra_fortaleza", "gym_rutina"]),
    ("Me da terror abrirme emocionalmente", ["sombra_rechazo", "sombra_fortaleza"]),
    ("Instalar dependencias con pip", ["programacion", "peer_herramientas"]),
]

print(f"\n{'Frase':<47s} {'Top-3 Activaciones'}")
print("-" * 100)

r1, r3, total = 0, 0, len(tests)

for text, expected in tests:
    q = model.encode([text], normalize_embeddings=True)[0]

    peer_best = {}
    thresholds = {}
    for fid, fd in facet_data.items():
        sim = float(np.dot(q, fd['emb']))
        pid = fd['peer_id']
        thresholds[pid] = fd['threshold']
        if pid not in peer_best or sim > peer_best[pid]['score']:
            peer_best[pid] = {'score': sim, 'name': fd['name'], 'domain': fd['domain']}

    # Filter by threshold
    activated = {pid: d for pid, d in peer_best.items() if d['score'] >= thresholds.get(pid, 0.25)}
    sorted_act = sorted(activated.items(), key=lambda x: -x[1]['score'])[:3]

    top_str = " | ".join(f"{a[1]['name']}({a[1]['score']:.3f})" for a in sorted_act)

    # Check if any expected is in top-3
    top_ids = [a[0] for a in sorted_act]
    hit = any(e in top_ids for e in expected)
    mark = "✅" if hit else "❌"
    if hit: r3 += 1
    if sorted_act and sorted_act[0][0] in expected: r1 += 1

    print(f"  {mark} {text[:45]:<45s} → {top_str}")

print(f"\nR@1: {r1}/{total} ({r1/total:.1%})  R@3: {r3}/{total} ({r3/total:.1%})")

conn.close()
