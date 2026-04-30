"""Muninn Seed — Crear los 4 peers conceptuales con descripciones ricas."""

import json
import os
import sys
import time

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add muninn package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "muninn"))

from db import init_db, get_connection
from embeddings import embed
import struct


DB_PATH = os.path.join(os.path.dirname(__file__), "muninn_seed.db")
os.environ["DB_PATH"] = DB_PATH


# ══════════════════════════════════════════════════════════════
# PEERS CONCEPTUALES
# ══════════════════════════════════════════════════════════════

peers = [
    {
        "id": "sombra_muerte",
        "name": "Muerte",
        "type": "sombra",
        "description": "La muerte como presencia constante en la vida de Diego",
        "representation": "La sombra de muerte se activa cuando aparecen temas de pérdida, fragilidad, finalidad, duelo o la conciencia de lo efímero.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "tags": ["muerte", "pérdida", "duelo", "fragilidad", "hermano", "nona"],
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
        "representation": "La sombra de rechazo se activa cuando hay dinámicas de exclusión, indiferencia, no pertenencia, puertas cerradas, amor no correspondido.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "tags": ["rechazo", "abandono", "madre", "mujeres", "no pertenencia"],
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
        "description": "La ansiedad transformada en guía, la guardiana del duelo congelado",
        "representation": "El Ángel del Atardecer se activa cuando hay ansiedad, catastrofización, exceso de información, tecnología abrumadora, o la necesidad de mirar lo que se esconde debajo de la alfombra.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "tags": ["ansiedad", "atardecer", "duelo", "catastrofización", "IA", "tecnología", "guía"],
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
        "description": "La armadura de protección que busca evolucionar a presencia auténtica",
        "representation": "Mi Fortaleza se activa cuando hay dinámicas de demostrar valor, perfeccionismo, miedo a la vulnerabilidad, gym como templo, o la tensión entre escudo y autenticidad.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "tags": ["fortaleza", "máscara", "perfección", "gym", "vulnerabilidad", "armadura"],
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


def store_embedding(conn, peer_id: str, text: str):
    """Generate and store embedding for a peer."""
    vector = embed(text)
    vec_bytes = struct.pack(f"{len(vector)}f", *vector)
    conn.execute("DELETE FROM peer_embeddings WHERE peer_id = ?", [peer_id])
    conn.execute("INSERT INTO peer_embeddings (peer_id, embedding) VALUES (?, ?)", [peer_id, vec_bytes])


def seed():
    print("=" * 70)
    print("  MUNINN SEED — Creando peers conceptuales")
    print("=" * 70)

    # Clean start
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)

    conn = init_db(DB_PATH)

    for p in peers:
        print(f"\n  Creando: {p['name']} ({p['id']})")
        print(f"  Tags: {', '.join(p['tags'])}")
        print(f"  Threshold: {p['activation_threshold']}")
        print(f"  Embedding text: {p['embedding_text'][:60]}...")

        conn.execute("""
            INSERT INTO peers (id, name, type, description, representation, confidence, activation_threshold, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            p["id"], p["name"], p["type"], p["description"],
            p["representation"], p["confidence"], p["activation_threshold"],
            json.dumps(p["tags"]),
        ])

        store_embedding(conn, p["id"], p["embedding_text"])
        conn.commit()
        print(f"  ✅ Embedding generado ({len(embed(p['embedding_text']))}D)")

    # Verify
    print(f"\n{'─'*70}")
    print("  VERIFICACIÓN:")
    rows = conn.execute("SELECT id, name, activation_threshold FROM peers WHERE is_active=1").fetchall()
    for r in rows:
        print(f"    {r['id']:30s} {r['name']:20s} thr={r['activation_threshold']}")

    # Quick activation test
    print(f"\n  TEST RÁPIDO DE ACTIVACIÓN:")
    test_phrases = [
        "Si lloro es debilidad",
        "Tengo hambre",
        "Me siento rechazado",
        "Tengo miedo de perder a alguien",
    ]

    import numpy as np
    for phrase in test_phrases:
            # Fallback: manual cosine sim
            phrase_emb = embed(phrase)
            peer_embs = {}
            for p_data in peers:
                peer_embs[p_data["id"]] = embed(p_data["embedding_text"])
            import numpy as np
            best_name = ""
            best_sim = -1
            for pid, pemb in peer_embs.items():
                sim = float(np.dot(phrase_emb, pemb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(pemb)))
                if sim > best_sim:
                    best_sim = sim
                    best_name = next(p["name"] for p in peers if p["id"] == pid)
            status = "ACTIVA" if best_sim >= 0.25 else "ruido"
            print(f"    \"{phrase[:40]}\" -> {best_name} ({best_sim:.4f}) [{status}]")

    conn.close()
    print(f"\n  DB guardada en: {DB_PATH}")
    print(f"  ✅ Seed completado!")


if __name__ == "__main__":
    seed()
