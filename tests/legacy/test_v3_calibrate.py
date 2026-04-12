import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

import sqlite3, struct, numpy as np
sys.path.insert(0, r'D:\github\muninn')
from muninn.embeddings_v2 import get_backend

backend = get_backend()
conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec; sqlite_vec.load(conn)
except: pass

facets = conn.execute('''SELECT pf.id, pf.peer_id, pf.text, p.name, p.domain, p.activation_threshold
    FROM peer_facets pf JOIN peers p ON pf.peer_id = p.id WHERE p.is_active = 1''').fetchall()

facet_data = {}
for f in facets:
    row = conn.execute('SELECT embedding FROM facet_embeddings WHERE facet_id = ?', [f['id']]).fetchone()
    if row:
        emb = np.array(struct.unpack('1024f', row['embedding']), dtype=np.float32)
        facet_data[f['id']] = {'emb': emb, 'peer_id': f['peer_id'], 'name': f['name'], 'domain': f['domain'], 'threshold': f['activation_threshold']}

instruction = "Dado un mensaje de un usuario, identifica que dominio de su vida esta activando"

test_phrases = [
    ("Me mandaron un mensaje y no me contestaron", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona esta mejor pero sigue en cama", "sombra_muerte"),
    ("Si lloro es debilidad", "sombra_fortaleza"),
    ("Tengo miedo de perder a alguien cercano", "sombra_muerte"),
    ("Una chica que me gustaba me dejo en visto", "sombra_rechazo"),
    ("La ansiedad no me deja dormir", "sombra_angel_atardecer"),
    ("El gimnasio es mi templo", "sombra_fortaleza"),
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
    ("Que hora es?", "peer_usuario"),
    ("Necesito ejecutar un comando en la terminal", "peer_herramientas"),
    ("Antes de empezar, cual es la vision?", "peer_operativo"),
    ("Toma un screenshot de mi pantalla", "peer_skills"),
]

total = len(test_phrases)
for thresh in [0.25, 0.35, 0.40, 0.45, 0.50]:
    r1, r3, ra = 0, 0, 0
    for text, expected in test_phrases:
        q = np.array(backend.embed([text], is_query=True, instruction=instruction)).flatten()
        peer_best = {}
        for fid, fd in facet_data.items():
            sim = float(np.dot(q, fd['emb']))
            pid = fd['peer_id']
            if pid not in peer_best or sim > peer_best[pid]['score']:
                peer_best[pid] = {'score': sim, 'name': fd['name']}
        filtered = {pid: d for pid, d in peer_best.items() if d['score'] >= thresh}
        sorted_a = sorted(filtered.items(), key=lambda x: -x[1]['score'])[:3]
        any_a = sorted(filtered.items(), key=lambda x: -x[1]['score'])
        top_ids = [a[0] for a in sorted_a]
        any_ids = [a[0] for a in any_a]
        if top_ids and top_ids[0] == expected: r1 += 1
        if expected in top_ids: r3 += 1
        if expected in any_ids: ra += 1
    print(f"  th={thresh:.2f}  R@1={r1:2d}/{total}({r1/total:.0%})  R@3={r3:2d}/{total}({r3/total:.0%})  R@any={ra:2d}/{total}({ra/total:.0%})")

conn.close()
