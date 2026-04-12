"""Test end-to-end del dreaming de Muninn.

Flujo:
1. Crear DB con 4 peers conceptuales (seed)
2. Ingresar eventos (mensajes del usuario)
3. Ejecutar dreaming
4. Verificar: memorias creadas, activaciones, conexiones

No usa la API HTTP, usa los módulos directamente.
"""

import json
import os
import sys
import struct

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from muninn.db import init_db, get_connection
from muninn.embeddings import embed
from muninn.router import route
from muninn.dreaming import dream


DB_PATH = os.path.join(os.path.dirname(__file__), "muninn_dream_test.db")
os.environ["DB_PATH"] = DB_PATH


# ══════════════════════════════════════════════════════════════
# PEERS (descripciones narrativas de seed_peers.py)
# ══════════════════════════════════════════════════════════════

PEERS = [
    {
        "id": "sombra_muerte",
        "name": "Muerte",
        "type": "sombra",
        "description": "La muerte como presencia constante",
        "representation": "Se activa con temas de pérdida, fragilidad, finalidad, duelo.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "tags": ["muerte", "pérdida", "duelo", "fragilidad"],
        "embedding_text": (
            "El hermano que murió antes de nacer, el cordón que se enroscó, el silencio que "
            "dejó un vacío en la familia antes de que yo existiera. La muerte como presencia "
            "constante, no como evento lejano. El miedo a que los seres queridos desaparezcan "
            "sin aviso. La fragilidad del cuerpo, lo fácil que es que algo se corte. Los sueños "
            "donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar "
            "de estar en cualquier momento. La relación con la nona, esperando una operación, "
            "la familia acompañando en silencio."
        ),
    },
    {
        "id": "sombra_rechazo",
        "name": "Rechazo",
        "type": "sombra",
        "description": "El patrón de rechazo originado en la relación materna",
        "representation": "Se activa con dinámicas de exclusión, indiferencia, no pertenencia.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "tags": ["rechazo", "abandono", "madre", "mujeres"],
        "embedding_text": (
            "La madre que perdió un hijo antes de que él naciera, y ese bebé que nació "
            "sintiendo que el amor materno tenía una grieta invisible. Para un niño, que "
            "mamá no te mire es desaparecer. Esa ecuación se grabó tan profundo que después "
            "toda mujer que no lo mira siente igual. La fiesta donde ella no le hizo caso. "
            "El grupo que no lo incluyó. La sensación de no pertenecer, de estar siempre un "
            "paso afuera. El patrón de buscar amor donde no florece, de acercarse esperando "
            "la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar."
        ),
    },
    {
        "id": "sombra_angel_atardecer",
        "name": "Ángel del Atardecer",
        "type": "sombra",
        "description": "La ansiedad transformada en guía",
        "representation": "Se activa con ansiedad, catastrofización, exceso de información.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "tags": ["ansiedad", "atardecer", "duelo", "catastrofización"],
        "embedding_text": (
            "La ansiedad que llega cuando el sol baja, no como enemiga sino como guardiana. "
            "Nació el día que murió el tata, el padre de su padre, cuando el dolor era tan "
            "grande que nadie podía entenderlo y ella se encargó de esconderlo. Después el "
            "padre también se fue, y ella se quedó, activa cada tarde, recordándole que había "
            "algo debajo de la alfombra. Por mucho tiempo fue sufrimiento puro: la catastrofización, "
            "la neblina mental, la sensación de que algo malo estaba por ocurrir. La tecnología "
            "y la inteligencia artificial eran catalizadores, pantallas que no se apagaban, "
            "demasiada información, el suelo moviéndose bajo los pies. Pero ella no quería "
            "asustar, quería ser mirada. Un día dijo: ya no puedo seguir protegiéndote, es "
            "hora de entrar al agua helada y nadar. No es miedo a la muerte, es miedo a la "
            "vida. La misma IA que generaba ansiedad ahora es aliada en el aprendizaje. La "
            "sombra no desapareció, se transformó en guía. Su mensaje: no escondas más "
            "sentimientos, todo lo que ocurre dentro ocurre afuera, mírame cuando te llame."
        ),
    },
    {
        "id": "sombra_fortaleza",
        "name": "Mi Fortaleza",
        "type": "sombra",
        "description": "La armadura de protección",
        "representation": "Se activa con dinámicas de demostrar valor, perfeccionismo, gym.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "tags": ["fortaleza", "máscara", "perfección", "gym"],
        "embedding_text": (
            "La máscara que se construyó tan temprano que ya no sabe si es máscara o rostro. "
            "Ser fuerte para no sentir, demostrar que vale para no ser abandonado. El gimnasio "
            "como templo, el cuerpo como prueba viviente de que no eres débil. El niño que "
            "aprendió que si llora es debilidad, que si muestra vulnerabilidad pierde protección. "
            "Pero la fortaleza real no es la armadura, es el coraje de bajarla. Las piernas "
            "que antes eran la debilidad ahora son la respuesta — el breakthrough del gym fue "
            "encontrar que la verdadera fuerza está en el territorio que más miedo daba. La "
            "fortaleza como mecanismo de supervivencia que cumplió su función y ahora busca "
            "evolucionar: de escudo a presencia, de demostrar a ser."
        ),
    },
]


def store_embedding(conn, peer_id, text):
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM peer_embeddings WHERE peer_id = ?", [peer_id])
    conn.execute("INSERT INTO peer_embeddings (peer_id, embedding) VALUES (?, ?)", [peer_id, vec_bytes])


def seed_peers(conn):
    """Create the 4 peers with embeddings."""
    for p in PEERS:
        conn.execute("""
            INSERT INTO peers (id, name, type, description, representation, confidence, activation_threshold, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [p["id"], p["name"], p["type"], p["description"], p["representation"],
              p["confidence"], p["activation_threshold"], json.dumps(p["tags"])])
        store_embedding(conn, p["id"], p["embedding_text"])
    conn.commit()


# ══════════════════════════════════════════════════════════════
# EVENTOS SIMULADOS (una sesión típica de Diego por Telegram)
# ══════════════════════════════════════════════════════════════

EVENTS = [
    # type, content, expected_activation
    ("user_message", "Hoy me sentí muy fuerte en el gym, subí el peso en sentadillas", "sombra_fortaleza"),
    ("user_message", "Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("user_message", "Qué lindo día hace hoy", None),  # ruido
    ("user_message", "Me entró ansiedad mirando el atardecer, demasiada información en la pantalla", "sombra_angel_atardecer"),
    ("user_message", "Pensé en mi nona, tengo miedo de que no sobreviva la operación", "sombra_muerte"),
    ("user_message", "Alguien me dijo que soy demasiado intenso, que debería calmarme", "sombra_fortaleza"),
    ("user_message", "No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("user_message", "Tengo hambre, voy a cocinar algo", None),  # ruido
    ("user_message", "Soñé que perdía a alguien cercano, me desperté con taquicardia", "sombra_muerte"),
    ("user_message", "Voy a cargar creatina hoy, se me acabó la semana pasada", "sombra_fortaleza"),
    ("user_message", "Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("user_message", "Una chica que me gustaba me dejó en visto", "sombra_rechazo"),
    ("bot_response", "Te entiendo, Diego. ¿Cómo te hizo sentir eso?", None),  # bot, no se procesa
    ("user_message", "Tengo que demostrar que soy capaz, no puedo fallar", "sombra_fortaleza"),
    ("user_message", "A veces pienso que si bajo la guardia me van a lastimar", "sombra_fortaleza"),
    ("user_message", "La nona está mejor pero sigue en cama, me da pena verla así", "sombra_muerte"),
]


def test():
    print("=" * 70)
    print("  MUNINN DREAMING TEST — End-to-End")
    print("=" * 70)
    
    # ── Setup ─────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)
    
    print("\n  [1] Creando DB con 4 peers...")
    conn = init_db(DB_PATH)
    seed_peers(conn)
    
    peers = conn.execute("SELECT id, name FROM peers WHERE is_active=1").fetchall()
    print(f"      {len(peers)} peers creados")
    conn.close()
    
    # ── Ingestar eventos (via API directa) ────────────────────
    print(f"\n  [2] Ingresando {len(EVENTS)} eventos...")
    conn = get_connection(DB_PATH)
    
    session_id = "test_2026-04-07_telegram"
    
    # Create session
    conn.execute("""
        INSERT OR IGNORE INTO sessions (id, channel, chat_id)
        VALUES (?, 'telegram', 'test_user')
    """, [session_id])
    conn.commit()
    
    activation_results = []
    
    for i, (event_type, content, expected) in enumerate(EVENTS):
        # Insert event
        conn.execute("""
            INSERT INTO events (session_id, type, content, channel, metadata)
            VALUES (?, ?, ?, 'telegram', '{}')
        """, [session_id, event_type, content])
        event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        if event_type != "user_message":
            conn.commit()
            continue
        
        # Route event
        activated = route(content, db_path=DB_PATH)
        
        # Record activations
        activated_names = []
        for a in activated:
            conn.execute("""
                INSERT INTO activations (event_id, peer_id, similarity)
                VALUES (?, ?, ?)
            """, [event_id, a["peer_id"], a["similarity"]])
            conn.execute("""
                UPDATE peers SET activation_count = activation_count + 1,
                    last_activated_at = datetime('now') WHERE id = ?
            """, [a["peer_id"]])
            activated_names.append(f"{a['name']}({a['similarity']:.2f})")
        
        conn.commit()
        
        match = "✅" if (expected and activated and activated[0]["peer_id"] == expected) or (not expected and not activated) else "❌" if expected else "—"
        act_str = ", ".join(activated_names) if activated_names else "(ruido)"
        exp_str = expected or "ruido"
        print(f"      [{i+1:2d}] {match} \"{content[:45]:<45s}\" → {act_str:<40s} (esperado: {exp_str})")
        
        activation_results.append({
            "content": content[:40],
            "expected": expected,
            "got": activated[0]["peer_id"] if activated else None,
            "match": match == "✅",
        })
    
    conn.close()
    
    # ── Stats pre-dreaming ────────────────────────────────────
    conn = get_connection(DB_PATH)
    pre_memories = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active=1").fetchone()[0]
    pre_activations = conn.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
    conn.close()
    
    print(f"\n  Pre-dreaming: {pre_memories} memorias, {pre_activations} activaciones")
    
    # ── Ejecutar Dreaming ─────────────────────────────────────
    print(f"\n  [3] Ejecutando Dreaming...")
    stats = dream(db_path=DB_PATH)
    
    print(f"\n      Resultado del dreaming:")
    print(f"        Eventos procesados: {stats['events_processed']}")
    print(f"        Memorias creadas:   {stats['memories_created']}")
    print(f"        Memorias saltadas:  {stats['memories_skipped']}")
    print(f"        Conexiones nuevas:  {stats['connections_found']}")
    print(f"        Peers actualizados: {stats['peers_updated']}")
    print(f"        Memorias olvidadas: {stats['memories_forgotten']}")
    if stats['errors']:
        print(f"        Errores: {stats['errors']}")
    
    # ── Verificación post-dreaming ────────────────────────────
    conn = get_connection(DB_PATH)
    
    post_memories = conn.execute("SELECT COUNT(*) FROM memories WHERE is_active=1").fetchone()[0]
    post_connections = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    
    print(f"\n  Post-dreaming: {post_memories} memorias, {post_connections} conexiones")
    
    # Show memories created
    print(f"\n  [4] Memorias creadas por el dreaming:")
    memories = conn.execute("""
        SELECT m.*, GROUP_CONCAT(mp.peer_id) AS peers
        FROM memories m
        LEFT JOIN memory_peers mp ON m.id = mp.memory_id
        WHERE m.source = 'dreaming' AND m.is_active = 1
        GROUP BY m.id
        ORDER BY m.confidence DESC
    """).fetchall()
    
    for m in memories:
        print(f"      [{m['id']:2d}] conf={m['confidence']:.2f} peers={m['peers'] or 'none':30s} \"{m['content'][:50]}\"")
    
    # Show connections
    if post_connections > 0:
        print(f"\n  [5] Conexiones entre peers:")
        connections = conn.execute("""
            SELECT c.*, p1.name AS from_name, p2.name AS to_name
            FROM connections c
            JOIN peers p1 ON c.from_peer_id = p1.id
            JOIN peers p2 ON c.to_peer_id = p2.id
        """).fetchall()
        for c in connections:
            print(f"      {c['from_name']} ←→ {c['to_name']} ({c['relation_type']}, str={c['strength']:.2f})")
    
    # Show consolidation log
    print(f"\n  [6] Consolidation log:")
    logs = conn.execute("SELECT * FROM consolidation_log ORDER BY id DESC LIMIT 3").fetchall()
    for log in logs:
        print(f"      #{log['id']} status={log['status']} processed={log['memories_processed']} added={log['memories_added']} connections={log['connections_found']}")
    
    # Summary
    print(f"\n{'═'*70}")
    correct = sum(1 for r in activation_results if r["match"])
    total = len(activation_results)
    print(f"  RESUMEN:")
    print(f"    Routing:     {correct}/{total} correctos ({correct/total*100:.0f}%)")
    print(f"    Memorias:    {post_memories} creadas de {stats['events_processed']} eventos")
    print(f"    Conexiones:  {post_connections} descubiertas")
    print(f"    Errores:     {len(stats['errors'])}")
    print(f"{'═'*70}")
    
    conn.close()
    print(f"\n  DB: {DB_PATH}")


if __name__ == "__main__":
    test()
