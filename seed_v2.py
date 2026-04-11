"""Muninn Seed v0.2 — Disco Elysium architecture.

Creates peers with facets and connections.
Based on test C7 facetas (best performing configuration).
Uses embeddings_v2 (sentence-transformers MiniLM for seed, swappable later).
"""

import json
import os
import sys
import struct
import time

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from muninn.db import init_db, get_connection, get_embedding_dims
from muninn.embeddings_v2 import embed
from muninn.models_v2 import FacetCreate

DB_PATH = os.path.join(os.path.dirname(__file__), "muninn.db")


# ══════════════════════════════════════════════════════════════
# PEERS CON FACETAS (del test C7 — mejores resultados)
# ══════════════════════════════════════════════════════════════

PEERS = [
    {
        "id": "sombra_muerte",
        "name": "Muerte",
        "type": "sombra",
        "domain": "Sombras",
        "description": "La muerte como presencia constante en la vida de Diego",
        "representation": "La sombra de muerte se activa cuando aparecen temas de pérdida, fragilidad, finalidad, duelo o la conciencia de lo efímero.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["muerte", "pérdida", "duelo", "fragilidad", "hermano", "nona"],
        "facets": [
            {"facet_type": "contextual", "text": "la nona internada en cama esperando operacion, ver un ser querido en hospital, el doctor dice que hay que hacer mas examenes, noticia de salud"},
            {"facet_type": "emocional", "text": "miedo a perder a alguien cercano, pensar en la edad de los padres, ver arrugas en mama, abuelos envejeciendo, cuenta regresiva"},
            {"facet_type": "emocional", "text": "suenar que un familiar enferma, suenar con dientes que se caen, suenos terminales, la muerte en suenos"},
            {"facet_type": "social", "text": "funerales, cementerio, tumba, el hermano que no nacio, el cordon umbilical, el vacio, el silencio de lo que pudo ser"},
            {"facet_type": "contextual", "text": "accidentes, cuerpo fragil, fragilidad humana, la vida es corta, emergencia familiar, operacion que cuesta millones"},
        ],
    },
    {
        "id": "sombra_rechazo",
        "name": "Rechazo",
        "type": "sombra",
        "domain": "Sombras",
        "description": "El patrón de rechazo originado en la relación materna",
        "representation": "La sombra de rechazo se activa cuando hay dinámicas de exclusión, indiferencia, no pertenencia, puertas cerradas, amor no correspondido.",
        "confidence": 0.7,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["rechazo", "abandono", "madre", "mujeres", "no pertenencia"],
        "facets": [
            {"facet_type": "social", "text": "alguien no responde mi mensaje, me deja en visto, no me contesta, leer y no responder, silencio digital"},
            {"facet_type": "social", "text": "no me invitan a la salida, el grupo hace planes sin mi, me borraron del grupo, no encajo, no pertenezco"},
            {"facet_type": "emocional", "text": "una chica que me gusta me rechaza, mi ex cambio su estado, buscar amor donde no florece, la puerta cerrada romantica"},
            {"facet_type": "emocional", "text": "me da terror abrirme emocionalmente y que me rechacen, si muestro quien soy me van a dejar, miedo al rechazo romantico"},
            {"facet_type": "contextual", "text": "me rechazan de un trabajo, alguien cancela planes, me comparo con otros incluidos, mama no me beso, fiesta donde nadie me hizo caso"},
        ],
    },
    {
        "id": "sombra_angel_atardecer",
        "name": "Ángel del Atardecer",
        "type": "sombra",
        "domain": "Sombras",
        "description": "La ansiedad transformada en guía, la guardiana del duelo congelado",
        "representation": "El Ángel del Atardecer se activa cuando hay ansiedad, catastrofización, exceso de información, tecnología abrumadora, o la necesidad de mirar lo que se esconde debajo.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["ansiedad", "atardecer", "duelo", "catastrofización", "IA", "tecnología", "guía"],
        "facets": [
            {"facet_type": "contextual", "text": "ansiedad al atardecer, inquietud vespertina, el sol baja y la angustia sube, la tarde como trigger"},
            {"facet_type": "emocional", "text": "neblina mental que no se va, catastrofizar, sentir que algo malo va a pasar, la mente va muy rapido, rumiacion"},
            {"facet_type": "tecnico", "text": "sentirse abrumado por tecnologia, pantallas, demasiada informacion, scrollear sin poder parar, leer sobre IA y sentir angustia"},
            {"facet_type": "contextual", "text": "no poder procesar tanta informacion, aprender demasiadas cosas sin pausa, paralizarse ante la cantidad, AGI, el suelo se mueve"},
            {"facet_type": "emocional", "text": "sentimientos escondidos bajo la alfombra, el duelo del tata no procesado, demasiado tiempo en pantallas, la guardiana emocional"},
        ],
    },
    {
        "id": "sombra_fortaleza",
        "name": "Mi Fortaleza",
        "type": "sombra",
        "domain": "Sombras",
        "description": "La armadura de protección que busca evolucionar a presencia auténtica",
        "representation": "Mi Fortaleza se activa cuando hay dinámicas de demostrar valor, perfeccionismo, miedo a la vulnerabilidad, gym como templo, o la tensión entre escudo y autenticidad.",
        "confidence": 0.6,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["fortaleza", "máscara", "perfección", "gym", "vulnerabilidad", "armadura"],
        "facets": [
            {"facet_type": "emocional", "text": "llorar es debilidad, no mostrar vulnerabilidad, sentir que debo ser fuerte siempre, contener las lagrimas"},
            {"facet_type": "emocional", "text": "no poder pedir ayuda, rechazar ayuda, decir que no necesito a nadie, bajar la guardia es perder todo, armarme emocionalmente"},
            {"facet_type": "emocional", "text": "demostrar que valgo para no ser abandonado, el valor personal se mide en fuerza, sentir dolor como fracaso, alguien me dijo que soy fuerte y me senti vacio"},
            {"facet_type": "emocional", "text": "la armadura como identidad, el coraje de bajarla, de escudo a presencia, la mascara que se construyo tan temprano que ya no sabe si es mascara o rostro"},
        ],
    },
    {
        "id": "proyecto_juego",
        "name": "Proyecto Juego",
        "type": "proyecto",
        "domain": "Proyecto Juego",
        "description": "El videojuego indie como expresión creativa y proyecto de vida",
        "representation": "Se activa con temas de diseño de juegos, mecánicas, combate, narrativa, inspiración en Disco Elysium.",
        "confidence": 0.3,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["juego", "indie", "combate", "narrativa", "Diseño Elysium", "prototipo"],
        "facets": [
            {"facet_type": "tecnico", "text": "sistema de combate, armas melee distancia escudos, durabilidad, tipos de dano, equilibrar mecanicas, stats del personaje"},
            {"facet_type": "emocional", "text": "variables emocionales del personaje, determinacion intuicion amor miedo dolor, sombras del personaje, experiencia emocional del jugador"},
            {"facet_type": "tecnico", "text": "bucle de juego explorar recolectar ganar dinero, sistema de economia mono-recurso, crafting, loot, progression"},
            {"facet_type": "tecnico", "text": "construir base, defender base, NPCs y dialogos, disenio de niveles, mundo abierto, arte conceptual, motor grafico, GDD"},
            {"facet_type": "contextual", "text": "disco elysium como inspiracion, juego indie minimalista, prototipo, playtest, iterar mecanicas, historia narrativa"},
        ],
    },
    {
        "id": "programacion",
        "name": "Programación",
        "type": "tema",
        "domain": "Programacion",
        "description": "Desarrollo de software, lenguajes, herramientas y resolución de problemas",
        "representation": "Se activa con temas de código, bugs, lenguajes de programación, arquitectura de software.",
        "confidence": 0.4,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["código", "python", "debug", "API", "git", "arquitectura"],
        "facets": [
            {"facet_type": "tecnico", "text": "error bug stack trace exception, debuggear, linea de codigo, KeyError TypeError IndexError, resolver problema de codigo"},
            {"facet_type": "tecnico", "text": "python javascript C PHP, aprender lenguaje, closures decoradores generadores, async await, clases objetos funciones"},
            {"facet_type": "tecnico", "text": "git commit push pull request, conflictos merge, branch, repositorio, versionar codigo, refactoring"},
            {"facet_type": "tecnico", "text": "API REST FastAPI Flask, base de datos SQL SQLite, frontend backend, arquitectura, patrones de diseno, deploy servidor"},
            {"facet_type": "contextual", "text": "instalar dependencias pip npm, virtualenv entorno virtual, testing unit test, documentacion tecnica, estructuras de datos algoritmos"},
        ],
    },
    {
        "id": "nanobot_sistema",
        "name": "Nanobot",
        "type": "sistema",
        "domain": "Nanobot",
        "description": "El asistente IA, su configuración, skills y funcionamiento",
        "representation": "Se activa con temas de configuración del bot, skills, memoria, cron, Telegram.",
        "confidence": 0.5,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["nanobot", "config", "skills", "telegram", "cron", "memoria"],
        "facets": [
            {"facet_type": "tecnico", "text": "configurar nanobot, provider Z.ai GLM, API key endpoint, config.json, suscripcion coding plan, cambiar provider"},
            {"facet_type": "contextual", "text": "cron job recordatorio programado, heartbeat check, scheduled task, timer, ejecucion periodica, lunes 10AM"},
            {"facet_type": "tecnico", "text": "skill nueva, clawhub, instalar skill, supervisor screenshot OCR, desktop-control, habit-tracker workout-logger"},
            {"facet_type": "contextual", "text": "MEMORY.md HISTORY.md, AGENTS.md SOUL.md, workspace, editar archivo del bot, actualizar memoria, session"},
            {"facet_type": "tecnico", "text": "telegram bot, chat_id canal, el bot no responde, conexion, failover openrouter, CLI vs telegram"},
        ],
    },
    {
        "id": "valle_alto",
        "name": "Valle Alto",
        "type": "proyecto",
        "domain": "Valle Alto",
        "description": "Proyecto inmobiliario familiar en cerros de Antofagasta",
        "representation": "Se activa con temas de terrenos, bienes nacionales, urbanización, inversión.",
        "confidence": 0.2,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["terreno", "inmobiliaria", "bienes nacionales", "inversión", "familia", "antofagasta"],
        "facets": [
            {"facet_type": "contextual", "text": "terreno en cerros de antofagasta, bienes nacionales, terreno fiscal, patentes mineras, hectareas, cota metros, acceso 4x4"},
            {"facet_type": "contextual", "text": "urbanizacion proyecto inmobiliario, plan regulador comunal, vivienda social DS19, constitucion empresa, abogado especializado"},
            {"facet_type": "tecnico", "text": "inversion millones de dolares, retorno de inversion, dashboard financiero, analisis financiero, socio estrategico inmobiliaria"},
            {"facet_type": "contextual", "text": "historia familiar abuelo padre, escrituras, negociacion con gobierno, cambio de gobierno oportunidad, superfice metros cuadrados"},
        ],
    },
    {
        "id": "gym_rutina",
        "name": "Gym/Rutina",
        "type": "tema",
        "domain": "Gym/Rutina",
        "description": "Entrenamiento físico, rutinas, ejercicios y nutrición",
        "representation": "Se activa con temas de ejercicios, rutinas, series, reps, suplementos.",
        "confidence": 0.4,
        "activation_threshold": 0.25,
        "level": 1.0,
        "max_activations": 3,
        "tags": ["gym", "rutina", "pierna", "hombro", "pesas", "nutrición", "series"],
        "facets": [
            {"facet_type": "fisico", "text": "hoy toca pierna, bulgaras mancuernas, sentadillas rack, prensa de piernas, hip thrust maquina, curl femoral, abductor adductor"},
            {"facet_type": "fisico", "text": "press militar barra, hombros con poleas, fondos con lastre, pectoral mancuernas inclinadas, barra y poleas, pesos libres"},
            {"facet_type": "fisico", "text": "pullups chinups muscle up handstand, calistenia planche front lever, fondos, barras con amigos"},
            {"facet_type": "contextual", "text": "series repeticiones, descanso entre series, calentamiento estiramiento, dolor agujetas recuperacion, duracion sesion hora y media"},
            {"facet_type": "contextual", "text": "calorias proteinas macros, proteina post-entreno, suplementos, split rutina lunes pierna martes hombros miercoles pierna"},
        ],
    },
    {
        "id": "casual_social",
        "name": "Casual/Social",
        "type": "sistema",
        "domain": "Casual/Social",
        "description": "Interacciones casuales, saludos, conversación sin tema específico",
        "representation": "Se activa con saludos, charla casual, sin tema específico.",
        "confidence": 0.2,
        "activation_threshold": 0.30,
        "level": 1.0,
        "max_activations": 2,
        "tags": ["casual", "saludos", "chat", "social"],
        "facets": [
            {"facet_type": "social", "text": "hola que tal, buen dia, como estai, wea, que onda, saludos, buenas noches, nos vemos, gracias"},
            {"facet_type": "social", "text": "que hora es, como esta el clima, pelicula recomendada, serie, musica, chiste, dime algo interesante"},
            {"facet_type": "social", "text": "planes para el fin de semana, que vas a hacer hoy, vamos a comer algo, conversacion relajada sin tema"},
            {"facet_type": "social", "text": "cuentame algo, charlar, sin tema especifico, solo hablar, pasar el rato, aburrimiento"},
        ],
    },
]

# ══════════════════════════════════════════════════════════════
# CONNECTIONS entre peers
# ══════════════════════════════════════════════════════════════

CONNECTIONS = [
    ("sombra_muerte", "sombra_angel_atardecer", "conecta", 0.7,
     "El ángel nació el día que murió el tata — muerte y ansiedad están conectadas"),
    ("sombra_muerte", "sombra_rechazo", "conecta", 0.5,
     "Perder a alguien es una forma de rechazo existencial"),
    ("sombra_rechazo", "sombra_fortaleza", "conecta", 0.6,
     "La fortaleza se construyó como protección contra el rechazo"),
    ("sombra_angel_atardecer", "sombra_fortaleza", "conecta", 0.4,
     "La ansiedad se esconde detrás de la armadura"),
    ("sombra_fortaleza", "gym_rutina", "activa", 0.7,
     "El gym es el territorio de la fortaleza, las piernas fueron el breakthrough"),
    ("programacion", "nanobot_sistema", "conecta", 0.8,
     "Nanobot es un proyecto de programación activo"),
    ("programacion", "proyecto_juego", "conecta", 0.6,
     "El juego requiere programación"),
]


def seed():
    print("=" * 70)
    print("  MUNINN SEED v0.2 — Disco Elysium Architecture")
    print("=" * 70)

    # Use sentence-transformers for seeding (fast, no GPU needed)
    os.environ["EMBEDDING_MODEL"] = "paraphrase-multilingual-MiniLM-L12-v2"

    # Clean start
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)
        print(f"  DB anterior eliminada")

    # MiniLM = 384 dimensions
    dims = 384
    conn = init_db(DB_PATH, dimensions=dims)

    # ── Crear peers con facetas ─────────────────────────────
    total_facets = 0
    for p_data in PEERS:
        print(f"\n  📦 {p_data['name']} ({p_data['id']})")
        print(f"     Domain: {p_data['domain']} | Facetas: {len(p_data['facets'])}")

        conn.execute("""
            INSERT INTO peers (id, name, type, domain, description, representation,
                             confidence, activation_threshold, level, max_activations, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            p_data["id"], p_data["name"], p_data["type"], p_data["domain"],
            p_data["description"], p_data["representation"], p_data["confidence"],
            p_data["activation_threshold"], p_data["level"], p_data["max_activations"],
            json.dumps(p_data["tags"]),
        ])

        # Create facets with embeddings
        for facet_data in p_data["facets"]:
            conn.execute("""
                INSERT INTO peer_facets (peer_id, facet_type, text, weight)
                VALUES (?, ?, ?, 1.0)
            """, [p_data["id"], facet_data["facet_type"], facet_data["text"]])

            facet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Generate and store embedding
            vector = embed(facet_data["text"])
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)
            conn.execute("INSERT INTO facet_embeddings (facet_id, embedding) VALUES (?, ?)", [facet_id, vec_bytes])
            total_facets += 1

        conn.commit()
        print(f"     ✅ {len(p_data['facets'])} facetas embebidas")

    # ── Crear connections ─────────────────────────────────────
    print(f"\n  🔗 Creando connections...")
    for from_id, to_id, rel_type, strength, desc in CONNECTIONS:
        conn.execute("""
            INSERT INTO connections (from_peer_id, to_peer_id, relation_type, strength, description)
            VALUES (?, ?, ?, ?, ?)
        """, [from_id, to_id, rel_type, strength, desc])
    conn.commit()
    print(f"     ✅ {len(CONNECTIONS)} conexiones creadas")

    # ── Verificación ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  VERIFICACIÓN:")

    peers = conn.execute("SELECT id, name, domain, activation_threshold FROM peers WHERE is_active=1").fetchall()
    print(f"  Peers: {len(peers)}")
    for r in peers:
        facet_count = conn.execute("SELECT COUNT(*) FROM peer_facets WHERE peer_id = ?", [r["id"]]).fetchone()[0]
        print(f"    {r['id']:30s} {r['name']:20s} domain={r['domain']:15s} facets={facet_count} thr={r['activation_threshold']}")

    total_p = len(peers)
    total_c = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    print(f"\n  Total: {total_p} peers, {total_facets} facetas, {total_c} conexiones")

    conn.close()
    print(f"\n  DB guardada en: {DB_PATH}")
    print(f"  ✅ Seed v0.2 completado!")


if __name__ == "__main__":
    seed()
