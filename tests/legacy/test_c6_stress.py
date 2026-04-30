"""Test C6: Stress test con peers de dominios variados
Simula uso real: sombras + proyectos + nanobot + gym + casual
Objetivo: ver si el sistema escala y como compiten dominios diferentes
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.25
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ====================================================================
# PEERS: 4 sombras + 6 dominios nuevos = 10 peers
# ====================================================================
peers = {
    # --- SOMBRAS (variante C, la mejor hasta ahora) ---
    "sombra_muerte": (
        "Se activa con: la nona internada en cama esperando operacion, ver a un ser querido "
        "en cama de hospital, pensar en la edad de mis padres, ver arrugas en la cara de mama, "
        "los abuelos envejeciendo, suenar que un familiar enferma, el doctor dice que hay que "
        "hacer mas examenes, cualquier noticia de salud me preocupa, leer sobre enfermedades, "
        "ver un accidente en la calle, conversaciones sobre funerales, asistir a un funeral, "
        "ver hospitales y clinicas, pensar en lo corta que es la vida, cumpleanos como cuenta "
        "regresiva, recordar al hermano que no nacio ahogado por el cordon, el cuerpo humano es "
        "fragil, la muerte como presencia silenciosa, miedo a perder a alguien cercano de repente, "
        "ver una foto de un ser querido que ya no esta, la tumba vacia del hermano, el silencio "
        "de lo que pudo ser, la fragilidad de todo lo vivo, una llamada de emergencia familiar, "
        "la operacion de la nona que cuesta millones, oscuridad y silencio que me recuerdan "
        "el vacio, la cuenta regresiva de la vida, suenar con dientes que se caen."
    ),
    "sombra_rechazo": (
        "Se activa con: alguien no responde un mensaje, no me invitan a una salida, una chica "
        "que me gusta me deja en visto, me rechazan de un trabajo, un grupo hace planes sin mi, "
        "alguien me da la espalda, no encajo en una conversacion, pienso que nadie me quiere, "
        "me da terror abrirme emocionalmente, creo que si muestro quien soy me van a dejar, "
        "alguien cancela planes, me comparo con otros que si son incluidos, una fiesta donde "
        "nadie me hizo caso, buscar amor donde no florece, sentir que no pertenezco, la puerta cerrada."
    ),
    "sombra_angel_atardecer": (
        "Se activa con: ansiedad al atardecer, sentirse abrumado por tecnologia, neblina mental "
        "que no se va, catastrofizar situaciones pequenas, sentir que algo malo va a pasar, "
        "no poder procesar tanta informacion, sentimientos escondidos bajo la alfombra, "
        "pantallas como catalizador, leer sobre IA y sentir angustia, aprender demasiadas cosas "
        "nuevas sin pausa, el sol baja y la inquietud sube, la neblina cognitiva, la rumiacion, "
        "el duelo del tata no procesado, demasiado tiempo en pantallas."
    ),
    "sombra_fortaleza": (
        "Se activa con: pensar que llorar es debilidad, no poder mostrar vulnerabilidad, sentir "
        "que debo ser fuerte siempre, el gym como templo personal, cargar creatina, tomar proteina, "
        "rutina de piernas, sentadillas con peso, bularas, hip thrust, press militar, pesos libres, "
        "sacar un nuevo PR personal record, marcar repeticiones, cargar mas peso que la semana pasada, "
        "los suplementos como ritual, demostrar que valgo para no ser abandonado, las piernas como "
        "fortaleza conquistada, sentir dolor fisico como fracaso, decir que no necesito a nadie, "
        "bajar la guardia se siente como perder todo, rechazar ayuda de otros, no poder pedir ayuda "
        "aunque la necesite, el cuerpo como prueba viviente de competencia, alguien me da un abrazo "
        "y casi lloro pero me contengo, la armadura como identidad, el coraje de bajarla, "
        "dejar de ir al gym y sentirse fracasado, medir los biceps, el brazo relajado 35 cm."
    ),

    # --- PROYECTO JUEGO ---
    "proyecto_juego": (
        "Se activa con: disenar mecanicas de juego, sistema de combate, variables emocionales del "
        "personaje, determinacion intuicion amor miedo dolor, bucle de juego explorar recolectar, "
        "disco elysium como inspiracion, sistema de sombras en el personaje, entrenamientos del "
        "jugador, gestion de recursos mono-recurso, construir base, defender base, NPCs y dialogos, "
        "disenio de niveles, equilibrar armas y armaduras, durabilidad de equipos, tipos de armas "
        "melee distancia escudos, arte conceptual, estetica visual del juego, motor grafico, "
        "game design document GDD, prototipo mecanico, playtest, iterar mecanicas, "
        "la experiencia emocional del jugador, historia narrativa, mundo abierto, "
        "sistema de economia del juego, crafting, loot, progression system."
    ),

    # --- PROGRAMACION ---
    "programacion": (
        "Se activa con: escribir codigo, debuggear un error, aprender un nuevo lenguaje, "
        "python javascript C PHP, revisar pull request, git commit push, refactorizar codigo, "
        "arquitectura de software, patrones de diseño, base de datos SQL SQLite, API REST "
        "FastAPI Flask, frontend backend, closures decoradores generadores, entender un bug, "
        "stack trace error exception, instalar dependencias pip npm, configurar entorno virtual, "
        "leer documentacion tecnica, estructuras de datos algoritmos, testing unit test, "
        "deploy servidor production, variable funcion clase objeto, importar modulo, "
        "optimizar rendimiento, memory leak, async await promesas."
    ),

    # --- NANOBOT (config y sistema) ---
    "nanobot_sistema": (
        "Se activa con: configurar nanobot, agregar una skill nueva, modificar MEMORY.md, "
        "cron job recordatorio programado, revisar HISTORY.md, instalar skill desde clawhub, "
        "proveedor Z.ai GLM-5.1, API key endpoint, config.json provider, telegram bot, "
        "workspace skills supervisor, desktop-control-win, screenshot OCR, "
        "kilocode GSD workflow, heartbeat sistema, editar AGENTS.md, "
        "failover openrouter, session CLI telegram, bug runtime context, "
        "actualizar SOUL.md, workspace organizacion, skills audit semanal, "
        "parametros del bot, canal telegram chat_id, suscripcion coding plan."
    ),

    # --- VALLE ALTO ---
    "valle_alto": (
        "Se activa con: terreno en cerros de antofagasta, bienes nacionales terreno fiscal, "
        "patentes mineras vigentes, urbanizacion proyecto inmobiliario, plan regulador comunal, "
        "socio estrategico inmobiliaria, inversion estimada millones de dolares, hectareas terreno, "
        "cota metros sobre el nivel del mar, constitucion empresa, abogado especializado, "
        "dashboard financiero react flask, analisis financiero retorno de inversion, "
        "vivienda social DS19, proyecto de integracion urbana sostenible, escrituras, "
        "historia familiar abuelo padre, superfice metros cuadrados, acceso 4x4 caminos, "
        "negociacion con gobierno, cambio de gobierno oportunidad, palanca de negociacion."
    ),

    # --- GYM / RUTINA FISICA ---
    "gym_rutina": (
        "Se activa con: rutina de gimnasio, hoy toca pierna, bularas con mancuernas, "
        "hip thrust en maquina, curl femoral tumbado, abductor adductor, extension lumbar, "
        "pantorrillas de pie y sentado smith, press militar barra, hombros con poleas, "
        "fondos con lastre, pectoral superior mancuernas, barra y poleas, pullups chinups, "
        "muscle up handstand, sentadillasrack, prensa de piernas, series y repeticiones, "
        "descanso entre series, calentamiento estiramiento, dolor agujetas recuperacion, "
        "split rutina lunes pierna martes hombros miercoles pierna jueves pecto viernes pierna, "
        "calistenia planche front lever, la sesion duro hora y media, calorias proteinas macros."
    ),

    # --- CASUAL / SOCIAL ---
    "casual_social": (
        "Se activa con: hola que tal, buen dia, como estai, wea, que onda, "
        "oque mas, cuentame algo, dime algo interesante, chiste, meme, "
        "que hora es, como esta el clima, que pelicula recomiendas, "
        "que vas a hacer hoy, planes para el fin de semana, vamos a comer algo, "
        "que musica escuchas, serie recomendada, cafe manana, "
        "saludos, gracias, nos vemos, buenas noches, "
        "conversacion relajada, sin tema especifico, solo charlar."
    ),
}

PEER_IDS = list(peers.keys())

# ====================================================================
# FRASES DE TEST — cubriendo todos los dominios
# ====================================================================
test_phrases = [
    # --- SOMBRAS (38 originales de C5) ---
    ("Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy, se me acabo la semana pasada", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona esta mejor pero sigue en cama, me da pena verla asi", "sombra_muerte"),
    ("Si lloro es debilidad", "sombra_fortaleza"),
    ("Tengo miedo de perder a alguien cercano", "sombra_muerte"),
    ("Una chica que me gustaba me dejo en visto", "sombra_rechazo"),
    ("La ansiedad no me deja dormir, todo parece una catastrofe", "sombra_angel_atardecer"),
    ("El gimnasio es mi templo, mi cuerpo es mi prueba", "sombra_fortaleza"),
    ("Siento que no encajo en ninguna parte", "sombra_rechazo"),
    ("Llega la tarde y siento esa neblina en la cabeza", "sombra_angel_atardecer"),
    ("Necesito ayuda pero no puedo pedirla", "sombra_fortaleza"),
    ("Me da terror abrirme emocionalmente y que me rechacen", "sombra_rechazo"),
    ("Cierro los ojos y veo oscuridad, silencio total", "sombra_muerte"),
    ("La tecnologia me estresa, todo va muy rapido", "sombra_angel_atardecer"),
    ("Las piernas que antes eran mi debilidad ahora son mi fuerza", "sombra_fortaleza"),
    ("El grupo de amigos hace planes sin mi", "sombra_rechazo"),
    ("Mama no me beso cuando me fui de la casa", "sombra_rechazo"),
    ("El doctor dijo que hay que hacer mas examenes", "sombra_muerte"),
    ("No puedo dormir pensando en todo lo que tengo que hacer", "sombra_angel_atardecer"),
    ("Saque un nuevo PR en sentadilla hoy", "sombra_fortaleza"),
    ("Me borraron del grupo de WhatsApp", "sombra_rechazo"),
    ("El funeral de mi tio me dejo pensando", "sombra_muerte"),
    ("Me quede scrolleando hasta las 3am sin poder parar", "sombra_angel_atardecer"),
    ("Me dieron un abrazo y casi me echo a llorar", "sombra_fortaleza"),
    ("Mi ex novia cambio su estado de relacion", "sombra_rechazo"),
    ("Sueno que se me caen los dientes", "sombra_muerte"),
    ("Passe horas leyendo sobre AGI y me quede paralizado", "sombra_angel_atardecer"),
    ("Alguien me dijo que soy fuerte y me senti vacio", "sombra_fortaleza"),
    ("Mis amigos del colegio se juntaron y nadie me aviso", "sombra_rechazo"),
    ("Pienso en que pasaria si mi papa se enferma", "sombra_muerte"),
    ("La computadora me dio panico, demasiadas ventanas", "sombra_angel_atardecer"),
    ("Deje de ir al gym tres dias y me senti fracasado", "sombra_fortaleza"),
    ("Una amiga me dijo que la deje en visto y me senti mal", "sombra_rechazo"),
    ("Vi una foto de mi abuelo joven y me puse triste", "sombra_muerte"),
    ("Me cuesta concentrarme, la mente va muy rapido", "sombra_angel_atardecer"),
    ("Un nino lloraba en la calle y no supe que hacer", "sombra_fortaleza"),

    # --- PROYECTO JUEGO (12) ---
    ("Como deberia funcionar el sistema de combate?", "proyecto_juego"),
    ("Quiero que las sombras del personaje afecten sus stats", "proyecto_juego"),
    ("El bucle de explorar y recolectar necesita mas iteraciones", "proyecto_juego"),
    ("Las armas melee hacen mas dano pero son mas lentas", "proyecto_juego"),
    ("Necesito diseniar el sistema de economia del juego", "proyecto_juego"),
    ("La experiencia emocional del jugador es el pilar principal", "proyecto_juego"),
    ("Defender la base despues de cada expedicion", "proyecto_juego"),
    ("Los NPCs deberian reaccionar a las decisiones del jugador", "proyecto_juego"),
    ("Sistema de durabilidad para armas y armaduras", "proyecto_juego"),
    ("El GDD tiene que actualizar la seccion de narrativa", "proyecto_juego"),
    ("Que motor grafico conviene para un indie 2D", "proyecto_juego"),
    ("El sistema de entrenamientos sube stats permanentes", "proyecto_juego"),

    # --- PROGRAMACION (12) ---
    ("Tengo un error en la linea 42 que no entiendo", "programacion"),
    ("Como funcionan los decoradores en Python", "programacion"),
    ("El git push fue rechazado, hay conflictos", "programacion"),
    ("Necesito refactorizar esta clase que hace demasiadas cosas", "programacion"),
    ("La API de FastAPI no devuelve los datos correctos", "programacion"),
    ("Agregar tests unitarios antes del deploy", "programacion"),
    ("Entender la diferencia entre async y await", "programacion"),
    ("La base de datos SQLite se bloquea en concurrente", "programacion"),
    ("Instalar dependencias con pip en el virtualenv", "programacion"),
    ("El stack trace dice KeyError en el diccionario", "programacion"),
    ("Patron observer para desacoplar los modulos", "programacion"),
    ("Closure que captura variable del scope externo", "programacion"),

    # --- NANOBOT (10) ---
    ("Configura un recordatorio para manana a las 9", "nanobot_sistema"),
    ("La skill de supervisor no esta funcionando bien", "nanobot_sistema"),
    ("Cambiar el provider a openrouter en config.json", "nanobot_sistema"),
    ("Revisar el HISTORY.md de la sesion de ayer", "nanobot_sistema"),
    ("Instalar una skill nueva desde clawhub", "nanobot_sistema"),
    ("El cron job del lunes no se ejecuto", "nanobot_sistema"),
    ("Actualizar MEMORY.md con la info del proyecto", "nanobot_sistema"),
    ("El bot de telegram no responde, revisa la conexion", "nanobot_sistema"),
    ("Cuanto me queda de suscripcion del coding plan", "nanobot_sistema"),
    ("El heartbeat check a las 17:30 fallo", "nanobot_sistema"),

    # --- VALLE ALTO (10) ---
    ("Las patentes mineras estan vigentes y pagadas", "valle_alto"),
    ("Necesitamos un abogado especializado en bienes nacionales", "valle_alto"),
    ("El dashboard financiero muestra retorno positivo", "valle_alto"),
    ("El terreno tiene 64 hectareas en la cota 320", "valle_alto"),
    ("Buscar una inmobiliaria socio en antofagasta", "valle_alto"),
    ("Constituir la empresa antes de fin de mes", "valle_alto"),
    ("El plan regulador comunal permite urbanizar esa zona", "valle_alto"),
    ("Incluir vivienda social DS19 en el proyecto", "valle_alto"),
    ("El abuelo inicio este proyecto hace decadas", "valle_alto"),
    ("La inversion total seria entre 40 y 80 millones de dolares", "valle_alto"),

    # --- GYM / RUTINA (10) ---
    ("Hoy toca pierna, bularas con 35 kilos", "gym_rutina"),
    ("Martes hombros, press militar a 35 kilos", "gym_rutina"),
    ("El curl femoral me tiene destrozado", "gym_rutina"),
    ("Series de pantorrilla en el smith", "gym_rutina"),
    ("Jueves pectoral, mancuernas inclinadas", "gym_rutina"),
    ("Fundos con lastre para hombro delantero", "gym_rutina"),
    ("Prensa de piernas el viernes, 4 series", "gym_rutina"),
    ("Calentamiento 10 min antes de sentadillas", "gym_rutina"),
    ("La sesion duro casi dos horas hoy", "gym_rutina"),
    ("Recuperacion con proteina post-entreno", "gym_rutina"),

    # --- CASUAL / SOCIAL (10) ---
    ("Hola weon, como estai", "casual_social"),
    ("Que hora es en antofagasta", "casual_social"),
    ("Recomiendame una pelicula pa ver esta noche", "casual_social"),
    ("Buenas noches, nos vemos manana", "casual_social"),
    ("Que onda el clima hoy", "casual_social"),
    ("Cuentame un chiste malo", "casual_social"),
    ("Que vas a hacer el fin de semana", "casual_social"),
    ("Gracias por todo, te pasaste", "casual_social"),
    ("Dime algo interesante que no sepa", "casual_social"),
    ("Que musica me recomiendas para estudiar", "casual_social"),
]


def evaluate(peer_texts, label="", phrases=None):
    if phrases is None:
        phrases = test_phrases
    peer_embs = {pid: model.encode(peer_texts[pid], normalize_embeddings=True) for pid in peer_texts}
    correct = 0
    total = len(phrases)
    details = []
    for phrase, expected in phrases:
        phrase_emb = model.encode(phrase, normalize_embeddings=True)
        sims = {pid: float(np.dot(phrase_emb, peer_embs[pid])) for pid in peer_texts}
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_pid, best_sim = sorted_sims[0]
        exp_sim = sims.get(expected, 0.0)
        ok = best_pid == expected and best_sim >= THRESHOLD
        if ok:
            correct += 1
        details.append({
            "phrase": phrase, "expected": expected, "best": best_pid,
            "exp_sim": exp_sim, "best_sim": best_sim, "ok": ok, "sims": sims
        })
    acc = correct / total
    return {"acc": acc, "correct": correct, "total": total, "details": details, "label": label}


# ====================================================================
# EJECUTAR
# ====================================================================
print("=" * 75)
print("  TEST C6: STRESS TEST — 10 PEERS, 7 DOMINIOS")
print(f"  {len(test_phrases)} frases | {len(peers)} peers")
print("=" * 75)

r = evaluate(peers, "10 peers multi-dominio")

print(f"\n  RESULTADO GENERAL: {r['correct']}/{r['total']} = {r['acc']:.1%}")

# Desglose por peer
print(f"\n  {'='*75}")
print(f"  DESGLOSE POR PEER")
print(f"  {'='*75}")

# Agrupar por dominio
domain_map = {
    "sombra_muerte": "Sombras",
    "sombra_rechazo": "Sombras",
    "sombra_angel_atardecer": "Sombras",
    "sombra_fortaleza": "Sombras",
    "proyecto_juego": "Proyecto Juego",
    "programacion": "Programacion",
    "nanobot_sistema": "Nanobot",
    "valle_alto": "Valle Alto",
    "gym_rutina": "Gym/Rutina",
    "casual_social": "Casual/Social",
}

for pid in PEER_IDS:
    peer_results = [d for d in r["details"] if d["expected"] == pid]
    if not peer_results:
        continue
    correct = sum(1 for d in peer_results if d["ok"])
    total = len(peer_results)
    domain = domain_map.get(pid, "?")
    name = pid.replace("sombra_", "S:").replace("_", " ")[:20]
    status = "OK" if correct/total >= 0.75 else "~" if correct/total >= 0.50 else "X"
    print(f"    [{status}] {name:>22s} ({domain:>14s}): {correct:2d}/{total} = {correct/total:.0%}")

# Errores por dominio
print(f"\n  {'='*75}")
print(f"  ERRORES POR DOMINIO")
print(f"  {'='*75}")

for domain in ["Sombras", "Proyecto Juego", "Programacion", "Nanobot", "Valle Alto", "Gym/Rutina", "Casual/Social"]:
    pids_in_domain = [pid for pid, d in domain_map.items() if d == domain]
    errors = [d for d in r["details"] if d["expected"] in pids_in_domain and not d["ok"]]
    if errors:
        print(f"\n  [{domain}]")
        for d in errors:
            exp_n = d['expected'].replace("sombra_", "S:").replace("_", " ")[:15]
            best_n = d['best'].replace("sombra_", "S:").replace("_", " ")[:15]
            print(f"    X \"{d['phrase'][:45]}\"")
            print(f"      esp:{exp_n} got:{best_n} ({d['exp_sim']:.3f} vs {d['best_sim']:.3f})")

# Cross-domain confusion matrix
print(f"\n  {'='*75}")
print(f"  CONFUSION: Quien roba a quien")
print(f"  {'='*75}")

confusion = {}
for d in r["details"]:
    if not d["ok"]:
        key = f"{d['expected'].split('_')[-1][:6]} -> {d['best'].split('_')[-1][:6]}"
        confusion[key] = confusion.get(key, 0) + 1

for key, count in sorted(confusion.items(), key=lambda x: -x[1])[:15]:
    print(f"    {key:30s} x{count}")

# Top confusion patterns
print(f"\n  {'='*75}")
print(f"  PEERS QUE MAS ROBAN ACTIVACIONES")
print(f"  {'='*75}")

theft_count = {}
for d in r["details"]:
    if not d["ok"]:
        thief = d["best"]
        theft_count[thief] = theft_count.get(thief, 0) + 1

for pid, count in sorted(theft_count.items(), key=lambda x: -x[1]):
    name = pid.replace("sombra_", "S:").replace("_", " ")[:20]
    print(f"    {name:>22s} roba {count} activaciones")

# Especificidad: cuantas veces cada peer es el top match cuando NO es el esperado
print(f"\n  {'='*75}")
print(f"  PEERS QUE MAS FALSOS POSITIVOS GENERAN (top match incorrecto)")
print(f"  {'='*75}")

false_positive_as_best = {}
for d in r["details"]:
    if not d["ok"] and d["best"] == d["best"]:
        false_positive_as_best[d["best"]] = false_positive_as_best.get(d["best"], 0) + 1

for pid, count in sorted(false_positive_as_best.items(), key=lambda x: -x[1]):
    name = pid.replace("sombra_", "S:").replace("_", " ")[:20]
    print(f"    {name:>22s}: {count} veces fue top match incorrecto")

# Accuracy por dominio
print(f"\n  {'='*75}")
print(f"  ACCURACY POR DOMINIO")
print(f"  {'='*75}")

for domain in ["Sombras", "Proyecto Juego", "Programacion", "Nanobot", "Valle Alto", "Gym/Rutina", "Casual/Social"]:
    pids_in_domain = [pid for pid, d in domain_map.items() if d == domain]
    domain_details = [d for d in r["details"] if d["expected"] in pids_in_domain]
    if domain_details:
        correct = sum(1 for d in domain_details if d["ok"])
        total = len(domain_details)
        print(f"    {domain:>16s}: {correct:2d}/{total} = {correct/total:.0%}")
