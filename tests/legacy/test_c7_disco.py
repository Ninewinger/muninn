"""Test C7: Sistema de Activacion Independiente tipo Disco Elysium
- Multi-activacion (no winner-takes-all)
- Facetas compuestas por peer (3-5 cada uno)
- Bonus de nivel y contexto
- Metricas: precision, recall, activaciones multiples correctas vs ruido
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ====================================================================
# DEFINICION DE PEERS CON FACETAS
# ====================================================================
@dataclass
class Facet:
    text: str
    facet_type: str  # emocional, fisico, social, contextual, tecnico
    embedding: np.ndarray = field(default=None, repr=False)

@dataclass
class Peer:
    id: str
    domain: str
    threshold: float  # umbral personal
    level: float = 1.0  # 0.0 - 2.0, peso del peer (frecuencia/recencia)
    facets: List[Facet] = field(default_factory=list)


def build_peers():
    peers = []

    # --- SOMBRA MUERTE ---
    p = Peer("sombra_muerte", "Sombras", threshold=0.25, level=1.0)
    p.facets = [
        Facet("la nona internada en cama esperando operacion, ver un ser querido en hospital, "
              "el doctor dice que hay que hacer mas examenes, noticia de salud", "contextual"),
        Facet("miedo a perder a alguien cercano, pensar en la edad de los padres, "
              "ver arrugas en mama, abuelos envejeciendo, cuenta regresiva", "emocional"),
        Facet("suenar que un familiar enferma, suenar con dientes que se caen, "
              "suenos terminales, la muerte en suenos", "emocional"),
        Facet("funerales, cementerio, tumba, el hermano que no nacio, "
              "el cordon umbilical, el vacio, el silencio de lo que pudo ser", "social"),
        Facet("accidentes, cuerpo fragil, fragilidad humana, la vida es corta, "
              "emergencia familiar, operacion que cuesta millones", "contextual"),
    ]
    peers.append(p)

    # --- SOMBRA RECHAZO ---
    p = Peer("sombra_rechazo", "Sombras", threshold=0.25, level=1.0)
    p.facets = [
        Facet("alguien no responde mi mensaje, me deja en visto, no me contesta, "
              "leer y no responder, silencio digital", "social"),
        Facet("no me invitan a la salida, el grupo hace planes sin mi, "
              "me borraron del grupo, no encajo, no pertenezco", "social"),
        Facet("una chica que me gusta me rechaza, mi ex cambio su estado, "
              "buscar amor donde no florece, la puerta cerrada romantica", "emocional"),
        Facet("me da terror abrirme emocionalmente y que me rechacen, "
              "si muestro quien soy me van a dejar, miedo al rechazo romantico", "emocional"),
        Facet("me rechazan de un trabajo, alguien cancela planes, "
              "me comparo con otros incluidos, mama no me beso, fiesta donde nadie me hizo caso", "contextual"),
    ]
    peers.append(p)

    # --- SOMBRA ANGEL DEL ATARDECER ---
    p = Peer("sombra_angel_atardecer", "Sombras", threshold=0.25, level=1.0)
    p.facets = [
        Facet("ansiedad al atardecer, inquietud vespertina, el sol baja y la "
              "angustia sube, la tarde como trigger", "contextual"),
        Facet("neblina mental que no se va, catastrofizar, sentir que algo malo "
              "va a pasar, la mente va muy rapido, rumiacion", "emocional"),
        Facet("sentirse abrumado por tecnologia, pantallas, demasiada informacion, "
              "scrollear sin poder parar, leer sobre IA y sentir angustia", "tecnico"),
        Facet("no poder procesar tanta informacion, aprender demasiadas cosas sin pausa, "
              "paralizarse ante la cantidad, AGI, el suelo se mueve", "contextual"),
        Facet("sentimientos escondidos bajo la alfombra, el duelo del tata no procesado, "
              "demasiado tiempo en pantallas, la guardiana emocional", "emocional"),
    ]
    peers.append(p)

    # --- SOMBRA FORTALEZA ---
    p = Peer("sombra_fortaleza", "Sombras", threshold=0.25, level=1.0)
    p.facets = [
        Facet("llorar es debilidad, no mostrar vulnerabilidad, sentir que debo ser "
              "fuerte siempre, contener las lagrimas", "emocional"),
        Facet("no poder pedir ayuda, rechazar ayuda, decir que no necesito a nadie, "
              "bajar la guardia es perder todo, armarme emocionalmente", "emocional"),
        Facet("demostrar que valgo para no ser abandonado, el valor personal se mide "
              "en fuerza, sentir dolor como fracaso, alguien me dijo que soy fuerte y me senti vacio", "emocional"),
        Facet("la armadura como identidad, el coraje de bajarla, de escudo a presencia, "
              "la mascara que se construyo tan temprano que ya no sabe si es mascara o rostro", "emocional"),
    ]
    peers.append(p)

    # --- PROYECTO JUEGO ---
    p = Peer("proyecto_juego", "Proyecto Juego", threshold=0.25, level=1.0)
    p.facets = [
        Facet("sistema de combate, armas melee distancia escudos, durabilidad, "
              "tipos de dano, equilibrar mecanicas, stats del personaje", "tecnico"),
        Facet("variables emocionales del personaje, determinacion intuicion amor "
              "miedo dolor, sombras del personaje, experiencia emocional del jugador", "emocional"),
        Facet("bucle de juego explorar recolectar ganar dinero, sistema de economia "
              "mono-recurso, crafting, loot, progression", "tecnico"),
        Facet("construir base, defender base, NPCs y dialogos, disenio de niveles, "
              "mundo abierto, arte conceptual, motor grafico, GDD", "tecnico"),
        Facet("disco elysium como inspiracion, juego indie minimalista, prototipo, "
              "playtest, iterar mecanicas, historia narrativa", "contextual"),
    ]
    peers.append(p)

    # --- PROGRAMACION ---
    p = Peer("programacion", "Programacion", threshold=0.25, level=1.0)
    p.facets = [
        Facet("error bug stack trace exception, debuggear, linea de codigo, "
              "KeyError TypeError IndexError, resolver problema de codigo", "tecnico"),
        Facet("python javascript C PHP, aprender lenguaje, closures decoradores "
              "generadores, async await, clases objetos funciones", "tecnico"),
        Facet("git commit push pull request, conflictos merge, branch, "
              "repositorio, versionar codigo, refactoring", "tecnico"),
        Facet("API REST FastAPI Flask, base de datos SQL SQLite, frontend backend, "
              "arquitectura, patrones de diseno, deploy servidor", "tecnico"),
        Facet("instalar dependencias pip npm, virtualenv entorno virtual, "
              "testing unit test, documentacion tecnica, estructuras de datos algoritmos", "contextual"),
    ]
    peers.append(p)

    # --- NANOBOT ---
    p = Peer("nanobot_sistema", "Nanobot", threshold=0.25, level=1.0)
    p.facets = [
        Facet("configurar nanobot, provider Z.ai GLM, API key endpoint, "
              "config.json, suscripcion coding plan, cambiar provider", "tecnico"),
        Facet("cron job recordatorio programado, heartbeat check, "
              "scheduled task, timer, ejecucion periodica, lunes 10AM", "contextual"),
        Facet("skill nueva, clawhub, instalar skill, supervisor screenshot OCR, "
              "desktop-control, habit-tracker workout-logger", "tecnico"),
        Facet("MEMORY.md HISTORY.md, AGENTS.md SOUL.md, workspace, "
              "editar archivo del bot, actualizar memoria, session", "contextual"),
        Facet("telegram bot, chat_id canal, el bot no responde, "
              "conexion, failover openrouter, CLI vs telegram", "tecnico"),
    ]
    peers.append(p)

    # --- VALLE ALTO ---
    p = Peer("valle_alto", "Valle Alto", threshold=0.25, level=1.0)
    p.facets = [
        Facet("terreno en cerros de antofagasta, bienes nacionales, terreno fiscal, "
              "patentes mineras, hectareas, cota metros, acceso 4x4", "contextual"),
        Facet("urbanizacion proyecto inmobiliario, plan regulador comunal, "
              "vivienda social DS19, constitucion empresa, abogado especializado", "contextual"),
        Facet("inversion millones de dolares, retorno de inversion, dashboard financiero, "
              "analisis financiero, socio estrategico inmobiliaria", "tecnico"),
        Facet("historia familiar abuelo padre, escrituras, negociacion con gobierno, "
              "cambio de gobierno oportunidad, superfice metros cuadrados", "contextual"),
    ]
    peers.append(p)

    # --- GYM / RUTINA ---
    p = Peer("gym_rutina", "Gym/Rutina", threshold=0.25, level=1.0)
    p.facets = [
        Facet("hoy toca pierna, bulgaras mancuernas, sentadillas rack, prensa de piernas, "
              "hip thrust maquina, curl femoral, abductor adductor", "fisico"),
        Facet("press militar barra, hombros con poleas, fondos con lastre, "
              "pectoral mancuernas inclinadas, barra y poleas, pesos libres", "fisico"),
        Facet("pullups chinups muscle up handstand, calistenia planche front lever, "
              "fondos, barras con amigos", "fisico"),
        Facet("series repeticiones, descanso entre series, calentamiento estiramiento, "
              "dolor agujetas recuperacion, duracion sesion hora y media", "contextual"),
        Facet("calorias proteinas macros, proteina post-entreno, suplementos, "
              "split rutina lunes pierna martes hombros miercoles pierna", "contextual"),
    ]
    peers.append(p)

    # --- CASUAL / SOCIAL ---
    p = Peer("casual_social", "Casual/Social", threshold=0.30, level=1.0)
    p.facets = [
        Facet("hola que tal, buen dia, como estai, wea, que onda, "
              "saludos, buenas noches, nos vemos, gracias", "social"),
        Facet("que hora es, como esta el clima, pelicula recomendada, "
              "serie, musica, chiste, dime algo interesante", "social"),
        Facet("planes para el fin de semana, que vas a hacer hoy, "
              "vamos a comer algo, conversacion relajada sin tema", "social"),
        Facet("cuentame algo, charlar, sin tema especifico, solo hablar, "
              "pasar el rato, aburrimiento", "social"),
    ]
    peers.append(p)

    return peers


# ====================================================================
# SISTEMA DE ACTIVACION
# ====================================================================
@dataclass
class Activation:
    peer_id: str
    facet_idx: int
    facet_type: str
    raw_score: float
    bonus_level: float
    bonus_context: float
    total_score: float


def compute_activations(event_text: str, peers: List[Peer],
                        max_activations: int = 3,
                        context_hour: int = None) -> List[Activation]:
    """Compute all activations for an event across all peers."""
    event_emb = model.encode(event_text, normalize_embeddings=True)
    
    all_activations = []
    
    for peer in peers:
        best_score = -1.0
        best_facet_idx = -1
        best_facet_type = ""
        
        for i, facet in enumerate(peer.facets):
            if facet.embedding is None:
                facet.embedding = model.encode(facet.text, normalize_embeddings=True)
            
            raw = float(np.dot(event_emb, facet.embedding))
            if raw > best_score:
                best_score = raw
                best_facet_idx = i
                best_facet_type = facet.facet_type
        
        # Bonus de nivel (peers mas activos pesan mas)
        bonus_level = (peer.level - 1.0) * 0.05  # rango: -0.05 a +0.05
        
        # Bonus de contexto (hora)
        bonus_context = 0.0
        if context_hour is not None:
            # Angel del atardecer: bonus si es tarde/noche
            if peer.id == "sombra_angel_atardecer" and 17 <= context_hour <= 23:
                bonus_context = 0.05
            # Casual: bonus si es fuera de horario laboral
            if peer.id == "casual_social" and (context_hour < 9 or context_hour > 21):
                bonus_context = 0.03
        
        total = best_score + bonus_level + bonus_context
        
        act = Activation(
            peer_id=peer.id,
            facet_idx=best_facet_idx,
            facet_type=best_facet_type,
            raw_score=best_score,
            bonus_level=bonus_level,
            bonus_context=bonus_context,
            total_score=total,
        )
        
        if total >= peer.threshold:
            all_activations.append(act)
    
    # Ordenar por score y limitar
    all_activations.sort(key=lambda a: -a.total_score)
    return all_activations[:max_activations]


# ====================================================================
# TEST FRASES — las mismas 102 del C6
# ====================================================================
test_phrases = [
    # --- SOMBRAS (38) ---
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
    ("Hoy toca pierna, bulgaras con 35 kilos", "gym_rutina"),
    ("Martes hombros, press militar a 35 kilos", "gym_rutina"),
    ("El curl femoral me tiene destrozado", "gym_rutina"),
    ("Series de pantorrilla en el smith", "gym_rutina"),
    ("Jueves pectoral, mancuernas inclinadas", "gym_rutina"),
    ("Fondos con lastre para hombro delantero", "gym_rutina"),
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


# ====================================================================
# EJECUTAR TEST
# ====================================================================
def main():
    print("=" * 75)
    print("  TEST C7: SISTEMA DE ACTIVACION DISCO ELYSIUM")
    print(f"  {len(test_phrases)} frases | facetas compuestas | multi-activacion")
    print("=" * 75)

    peers = build_peers()

    # Pre-compute all facet embeddings
    print("\n  Calculando embeddings de facetas...")
    total_facets = sum(len(p.facets) for p in peers)
    for p in peers:
        for f in p.facets:
            f.embedding = model.encode(f.text, normalize_embeddings=True)
    print(f"  {total_facets} facetas listas para {len(peers)} peers")

    # ====================================================================
    # METRICA 1: Recall — el peer correcto esta entre los activados?
    # ====================================================================
    print(f"\n\n  {'='*75}")
    print(f"  METRICA 1: RECALL (peer correcto entre activados, top-3)")
    print(f"  {'='*75}")

    recall_top1 = 0
    recall_top3 = 0
    recall_any = 0  # activado en cualquier posicion (sin limite)
    total = len(test_phrases)

    # Para comparar con sistema anterior (winner-takes-all)
    winner_takes_all_correct = 0

    details = []

    for phrase, expected in test_phrases:
        # Sistema nuevo: multi-activacion
        activations = compute_activations(phrase, peers, max_activations=3)
        
        activated_ids = [a.peer_id for a in activations]
        
        # Tambien sin limite para recall_any
        all_acts = compute_activations(phrase, peers, max_activations=10)
        any_ids = [a.peer_id for a in all_acts]
        
        # Recall
        in_top1 = expected == activated_ids[0] if activated_ids else False
        in_top3 = expected in activated_ids
        in_any = expected in any_ids
        
        if in_top1: recall_top1 += 1
        if in_top3: recall_top3 += 1
        if in_any: recall_any += 1
        
        # Winner takes all (top-1)
        if in_top1: winner_takes_all_correct += 1
        
        details.append({
            "phrase": phrase, "expected": expected,
            "activations": activations, "activated_ids": activated_ids,
            "in_top1": in_top1, "in_top3": in_top3, "in_any": in_any,
        })

    print(f"\n  Recall@1 (top-1):           {recall_top1:3d}/{total} = {recall_top1/total:.1%}")
    print(f"  Recall@3 (entre top-3):     {recall_top3:3d}/{total} = {recall_top3/total:.1%}")
    print(f"  Recall@any (sin limite):    {recall_any:3d}/{total} = {recall_any/total:.1%}")

    # ====================================================================
    # METRICA 2: Precision — de los activados, cuantos son correctos?
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  METRICA 2: PRECISION (de los activados, cuantos son correctos)")
    print(f"  {'='*75}")

    total_activations = 0
    correct_activations = 0  # peer esperado activado
    noise_activations = 0    # peer incorrecto activado

    multi_correct = 0  # casos donde multiple activacion es correcta
    multi_noise = 0    # casos donde multiple activacion agrega ruido

    for d in details:
        acts = d["activations"]
        expected = d["expected"]
        
        for a in acts:
            total_activations += 1
            if a.peer_id == expected:
                correct_activations += 1
            else:
                noise_activations += 1
        
        if len(acts) > 1:
            # Multi-activacion: hay peers no-esperados activos?
            non_expected = [a for a in acts if a.peer_id != expected]
            if non_expected:
                # Son "ruido" o son "contextualmente validos"?
                multi_noise += 1

    precision = correct_activations / total_activations if total_activations > 0 else 0
    print(f"\n  Activaciones totales:       {total_activations}")
    print(f"  Correctas (expected):       {correct_activations}")
    print(f"  Ruido (no expected):        {noise_activations}")
    print(f"  Precision:                  {precision:.1%}")
    print(f"  Promedio activaciones/msg:  {total_activations/total:.1f}")
    print(f"  Msgs con multi-activacion:  {sum(1 for d in details if len(d['activations'])>1)}/{total}")

    # ====================================================================
    # METRICA 3: Por peer
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  DESGLOSE POR PEER (Recall@3)")
    print(f"  {'='*75}")

    peer_ids = [p.id for p in peers]
    domain_map = {p.id: p.domain for p in peers}

    for pid in peer_ids:
        peer_details = [d for d in details if d["expected"] == pid]
        if not peer_details:
            continue
        r1 = sum(1 for d in peer_details if d["in_top1"])
        r3 = sum(1 for d in peer_details if d["in_top3"])
        ra = sum(1 for d in peer_details if d["in_any"])
        t = len(peer_details)
        domain = domain_map[pid]
        name = pid.replace("sombra_", "S:").replace("_", " ")[:22]
        status = "OK" if r3/t >= 0.80 else "+" if r3/t >= 0.65 else "~" if r3/t >= 0.50 else "X"
        print(f"    [{status}] {name:>22s} ({domain:>14s}): "
              f"R@1={r1:2d}/{t} R@3={r3:2d}/{t} R@any={ra:2d}/{t}")

    # ====================================================================
    # METRICA 4: Comparacion directa C6 vs C7
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  COMPARACION: C6 (winner-takes-all) vs C7 (multi-activacion)")
    print(f"  {'='*75}")

    # C6 accuracy fue 70.6% (72/102)
    c6_acc = 72/102
    c7_r1 = recall_top1/total
    c7_r3 = recall_top3/total

    print(f"\n  C6 (1 ganador):             {c6_acc:.1%}")
    print(f"  C7 top-1 (mejor activado):  {c7_r1:.1%}  ({c7_r1-c6_acc:+.1%})")
    print(f"  C7 top-3 (entre activados): {c7_r3:.1%}  ({c7_r3-c6_acc:+.1%})")

    # ====================================================================
    # DETALLE: Frases que MEJORARON con multi-activacion
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  FRASES QUE GANAN CON MULTI-ACTIVACION")
    print(f"  {'='*75}")

    for d in details:
        if not d["in_top1"] and d["in_top3"]:
            exp_n = d['expected'].replace("sombra_", "S:").replace("_", " ")[:15]
            acts_str = ", ".join(
                f"{a.peer_id.replace('sombra_','S:').replace('_',' ')[:8]}:{a.total_score:.3f}"
                for a in d["activations"]
            )
            print(f"    + \"{d['phrase'][:40]}\" -> {exp_n}")
            print(f"      activados: {acts_str}")

    # ====================================================================
    # DETALLE: Frases que FALLAN incluso con multi-activacion
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  FRASES QUE FALLAN INCLUSO CON RECALL@ANY (limites del modelo)")
    print(f"  {'='*75}")

    for d in details:
        if not d["in_any"]:
            exp_n = d['expected'].replace("sombra_", "S:").replace("_", " ")[:15]
            # Check si el peer correcto ni siquiera paso su threshold
            all_acts = compute_activations(d["phrase"], peers, max_activations=10)
            all_ids = [a.peer_id for a in all_acts]
            # Compute score for expected peer
            exp_score = 0
            for p in peers:
                if p.id == d["expected"]:
                    event_emb = model.encode(d["phrase"], normalize_embeddings=True)
                    for f in p.facets:
                        s = float(np.dot(event_emb, f.embedding))
                        if s > exp_score:
                            exp_score = s
            print(f"    X \"{d['phrase'][:45]}\" -> {exp_n} (best score: {exp_score:.3f})")
            if all_acts:
                top = all_acts[0]
                print(f"      top: {top.peer_id.replace('sombra_','S:').replace('_',' ')[:12]} "
                      f"({top.total_score:.3f})")

    # ====================================================================
    # ANALISIS DE RUIDO: cuando se activan peers incorrectos, es contextualmente valido?
    # ====================================================================
    print(f"\n  {'='*75}")
    print(f"  ANALISIS DE RUIDO (activaciones no-esperadas, son validas?)")
    print(f"  {'='*75}")

    noise_examples = []
    for d in details:
        for a in d["activations"]:
            if a.peer_id != d["expected"]:
                noise_examples.append((d["phrase"], d["expected"], a))

    # Los 10 mas ruidosos (score mas alto siendo incorrecto)
    noise_examples.sort(key=lambda x: -x[2].total_score)
    print(f"\n  Top 10 activaciones no-esperadas con mayor score:")
    for phrase, expected, act in noise_examples[:10]:
        exp_n = expected.replace("sombra_", "S:").replace("_", " ")[:10]
        act_n = act.peer_id.replace("sombra_", "S:").replace("_", " ")[:10]
        print(f"    \"{phrase[:35]}\" -> esp:{exp_n} got:{act_n} ({act.total_score:.3f})")


if __name__ == "__main__":
    main()
