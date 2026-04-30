"""Test C9: Calibracion Qwen3-8B — barrido de parametros
Bateria de tests para optimizar:
  1. Threshold (0.20 - 0.55)
  2. Dimensiones MRL (64, 128, 256, 512, 1024)
  3. Instructions (sin / generica / especifica / por faceta)
  4. Facetas enriquecidas vs actuales

Cada test corre las 102 frases y mide Recall@1, @3, @any + precision
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_CACHE'] = 'D:/hf_cache/hub'
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from typing import List, Union
import time


# ====================================================================
# QWEN3 EMBEDDING WRAPPER
# ====================================================================
class Qwen3Embedding:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B"):
        print(f"  Cargando {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side='left'
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        # Force garbage collection
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, sentences, is_query=False, instruction=None, dim=-1, batch_size=16):
        if isinstance(sentences, str):
            sentences = [sentences]
        all_outputs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            if is_query and instruction:
                batch = [f'Instruct: {instruction}\nQuery:{s}' for s in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors='pt')
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                pooled = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                if dim != -1:
                    pooled = pooled[:, :dim]
                pooled = F.normalize(pooled, p=2, dim=1)
                all_outputs.append(pooled.cpu().float().numpy())
        return np.vstack(all_outputs)


# ====================================================================
# DATA STRUCTURES
# ====================================================================
@dataclass
class Facet:
    text: str
    facet_type: str
    embedding: np.ndarray = field(default=None, repr=False)

@dataclass
class Peer:
    id: str
    domain: str
    threshold: float
    level: float = 1.0
    facets: list = field(default_factory=list)


def build_peers_standard():
    """Facetas actuales (las del C7/C8)."""
    peers = []

    p = Peer("sombra_muerte", "Sombras", threshold=0.25)
    p.facets = [
        Facet("la nona internada en cama esperando operacion, ver un ser querido en hospital, el doctor dice que hay que hacer mas examenes, noticia de salud", "contextual"),
        Facet("miedo a perder a alguien cercano, pensar en la edad de los padres, ver arrugas en mama, abuelos envejeciendo, cuenta regresiva", "emocional"),
        Facet("suenar que un familiar enferma, suenar con dientes que se caen, suenos terminales, la muerte en suenos", "emocional"),
        Facet("funerales, cementerio, tumba, el hermano que no nacio, el cordon umbilical, el vacio, el silencio de lo que pudo ser", "social"),
        Facet("accidentes, cuerpo fragil, fragilidad humana, la vida es corta, emergencia familiar, operacion que cuesta millones", "contextual"),
    ]
    peers.append(p)

    p = Peer("sombra_rechazo", "Sombras", threshold=0.25)
    p.facets = [
        Facet("alguien no responde mi mensaje, me deja en visto, no me contesta, leer y no responder, silencio digital", "social"),
        Facet("no me invitan a la salida, el grupo hace planes sin mi, me borraron del grupo, no encajo, no pertenezco", "social"),
        Facet("una chica que me gusta me rechaza, mi ex cambio su estado, buscar amor donde no florece, la puerta cerrada romantica", "emocional"),
        Facet("me da terror abrirme emocionalmente y que me rechacen, si muestro quien soy me van a dejar, miedo al rechazo romantico", "emocional"),
        Facet("me rechazan de un trabajo, alguien cancela planes, me comparo con otros incluidos, mama no me beso, fiesta donde nadie me hizo caso", "contextual"),
    ]
    peers.append(p)

    p = Peer("sombra_angel_atardecer", "Sombras", threshold=0.25)
    p.facets = [
        Facet("ansiedad al atardecer, inquietud vespertina, el sol baja y la angustia sube, la tarde como trigger", "contextual"),
        Facet("neblina mental que no se va, catastrofizar, sentir que algo malo va a pasar, la mente va muy rapido, rumiacion", "emocional"),
        Facet("sentirse abrumado por tecnologia, pantallas, demasiada informacion, scrollear sin poder parar, leer sobre IA y sentir angustia", "tecnico"),
        Facet("no poder procesar tanta informacion, aprender demasiadas cosas sin pausa, paralizarse ante la cantidad, AGI, el suelo se mueve", "contextual"),
        Facet("sentimientos escondidos bajo la alfombra, el duelo del tata no procesado, demasiado tiempo en pantallas, la guardiana emocional", "emocional"),
    ]
    peers.append(p)

    p = Peer("sombra_fortaleza", "Sombras", threshold=0.25)
    p.facets = [
        Facet("llorar es debilidad, no mostrar vulnerabilidad, sentir que debo ser fuerte siempre, contener las lagrimas", "emocional"),
        Facet("no poder pedir ayuda, rechazar ayuda, decir que no necesito a nadie, bajar la guardia es perder todo, armarme emocionalmente", "emocional"),
        Facet("demostrar que valgo para no ser abandonado, el valor personal se mide en fuerza, sentir dolor como fracaso, alguien me dijo que soy fuerte y me senti vacio", "emocional"),
        Facet("la armadura como identidad, el coraje de bajarla, de escudo a presencia, la mascara que se construyo tan temprano que ya no sabe si es mascara o rostro", "emocional"),
    ]
    peers.append(p)

    p = Peer("proyecto_juego", "Proyecto Juego", threshold=0.25)
    p.facets = [
        Facet("sistema de combate, armas melee distancia escudos, durabilidad, tipos de dano, equilibrar mecanicas, stats del personaje", "tecnico"),
        Facet("variables emocionales del personaje, determinacion intuicion amor miedo dolor, sombras del personaje, experiencia emocional del jugador", "emocional"),
        Facet("bucle de juego explorar recolectar ganar dinero, sistema de economia mono-recurso, crafting, loot, progression", "tecnico"),
        Facet("construir base, defender base, NPCs y dialogos, disenio de niveles, mundo abierto, arte conceptual, motor grafico, GDD", "tecnico"),
        Facet("disco elysium como inspiracion, juego indie minimalista, prototipo, playtest, iterar mecanicas, historia narrativa", "contextual"),
    ]
    peers.append(p)

    p = Peer("programacion", "Programacion", threshold=0.25)
    p.facets = [
        Facet("error bug stack trace exception, debuggear, linea de codigo, KeyError TypeError IndexError, resolver problema de codigo", "tecnico"),
        Facet("python javascript C PHP, aprender lenguaje, closures decoradores generadores, async await, clases objetos funciones", "tecnico"),
        Facet("git commit push pull request, conflictos merge, branch, repositorio, versionar codigo, refactoring", "tecnico"),
        Facet("API REST FastAPI Flask, base de datos SQL SQLite, frontend backend, arquitectura, patrones de diseno, deploy servidor", "tecnico"),
        Facet("instalar dependencias pip npm, virtualenv entorno virtual, testing unit test, documentacion tecnica, estructuras de datos algoritmos", "contextual"),
    ]
    peers.append(p)

    p = Peer("nanobot_sistema", "Nanobot", threshold=0.25)
    p.facets = [
        Facet("configurar nanobot, provider Z.ai GLM, API key endpoint, config.json, suscripcion coding plan, cambiar provider", "tecnico"),
        Facet("cron job recordatorio programado, heartbeat check, scheduled task, timer, ejecucion periodica, lunes 10AM", "contextual"),
        Facet("skill nueva, clawhub, instalar skill, supervisor screenshot OCR, desktop-control, habit-tracker workout-logger", "tecnico"),
        Facet("MEMORY.md HISTORY.md, AGENTS.md SOUL.md, workspace, editar archivo del bot, actualizar memoria, session", "contextual"),
        Facet("telegram bot, chat_id canal, el bot no responde, conexion, failover openrouter, CLI vs telegram", "tecnico"),
    ]
    peers.append(p)

    p = Peer("valle_alto", "Valle Alto", threshold=0.25)
    p.facets = [
        Facet("terreno en cerros de antofagasta, bienes nacionales, terreno fiscal, patentes mineras, hectareas, cota metros, acceso 4x4", "contextual"),
        Facet("urbanizacion proyecto inmobiliario, plan regulador comunal, vivienda social DS19, constitucion empresa, abogado especializado", "contextual"),
        Facet("inversion millones de dolares, retorno de inversion, dashboard financiero, analisis financiero, socio estrategico inmobiliaria", "tecnico"),
        Facet("historia familiar abuelo padre, escrituras, negociacion con gobierno, cambio de gobierno oportunidad, superfice metros cuadrados", "contextual"),
    ]
    peers.append(p)

    p = Peer("gym_rutina", "Gym/Rutina", threshold=0.25)
    p.facets = [
        Facet("hoy toca pierna, bulgaras mancuernas, sentadillas rack, prensa de piernas, hip thrust maquina, curl femoral, abductor adductor", "fisico"),
        Facet("press militar barra, hombros con poleas, fondos con lastre, pectoral mancuernas inclinadas, barra y poleas, pesos libres", "fisico"),
        Facet("pullups chinups muscle up handstand, calistenia planche front lever, fondos, barras con amigos", "fisico"),
        Facet("series repeticiones, descanso entre series, calentamiento estiramiento, dolor agujetas recuperacion, duracion sesion hora y media", "contextual"),
        Facet("calorias proteinas macros, proteina post-entreno, suplementos, split rutina lunes pierna martes hombros miercoles pierna", "contextual"),
    ]
    peers.append(p)

    p = Peer("casual_social", "Casual/Social", threshold=0.30)
    p.facets = [
        Facet("hola que tal, buen dia, como estai, wea, que onda, saludos, buenas noches, nos vemos, gracias", "social"),
        Facet("que hora es, como esta el clima, pelicula recomendada, serie, musica, chiste, dime algo interesante", "social"),
        Facet("planes para el fin de semana, que vas a hacer hoy, vamos a comer algo, conversacion relajada sin tema", "social"),
        Facet("cuentame algo, charlar, sin tema especifico, solo hablar, pasar el rato, aburrimiento", "social"),
    ]
    peers.append(p)

    return peers


# ====================================================================
# TEST FRASES (102)
# ====================================================================
test_phrases = [
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
# INSTRUCTIONS A PROBAR
# ====================================================================
INSTRUCTIONS = {
    "sin_instruction": None,
    "generica": "Retrieve semantically similar texts",
    "especifica": "Dado un mensaje de un usuario, identifica que dominio de su vida esta activando: emocional, sombras, proyecto, programacion, sistema, negocio, ejercicio, o social",
    "identidad": "Clasifica el siguiente texto en uno de estos dominios de la vida de un usuario: sombra de muerte, sombra de rechazo, sombra del angel del atardecer, sombra de fortaleza, proyecto de videojuego, programacion, sistema nanobot, proyecto inmobiliario valle alto, gimnasio rutina, o conversacion casual",
    "zero_shot": "Dado un mensaje en español, determina el dominio temático más relevante",
}


# ====================================================================
# RUNNER
# ====================================================================
def run_test(emb_model, peers, instruction, dim, threshold_override=None):
    """Run all 102 phrases and return metrics."""
    # Re-encode facets with current dim
    all_facet_texts = []
    facet_map = []
    for pi, p in enumerate(peers):
        for fi, f in enumerate(p.facets):
            all_facet_texts.append(f.text)
            facet_map.append((pi, fi))

    facet_embs = emb_model.encode(all_facet_texts, is_query=False, dim=dim, batch_size=16)
    for idx, (pi, fi) in enumerate(facet_map):
        peers[pi].facets[fi].embedding = facet_embs[idx]

    # Encode all queries
    query_texts = [p[0] for p in test_phrases]
    expected = [p[1] for p in test_phrases]
    query_embs = emb_model.encode(query_texts, is_query=True, instruction=instruction, dim=dim, batch_size=16)

    recall_top1 = 0
    recall_top3 = 0
    recall_any = 0
    total = len(test_phrases)

    for qi, (q_emb, exp) in enumerate(zip(query_embs, expected)):
        # Compute best facet per peer
        peer_scores = {}
        for p in peers:
            best = -1
            for f in p.facets:
                score = float(np.dot(q_emb, f.embedding))
                if score > best:
                    best = score
            peer_scores[p.id] = best

        # Sort by score desc
        sorted_peers = sorted(peer_scores.items(), key=lambda x: -x[1])

        # Apply threshold
        threshold = threshold_override if threshold_override is not None else 0.25
        above = [(pid, sc) for pid, sc in sorted_peers if sc >= threshold]

        top1_id = above[0][0] if above else None
        top3_ids = [pid for pid, _ in above[:3]]
        any_ids = [pid for pid, _ in above]

        if top1_id == exp: recall_top1 += 1
        if exp in top3_ids: recall_top3 += 1
        if exp in any_ids: recall_any += 1

    return {
        "r1": recall_top1 / total,
        "r3": recall_top3 / total,
        "ra": recall_any / total,
    }


def main():
    print("=" * 80)
    print("  TEST C9: CALIBRACION QWEN3-8B — BARRIDO DE PARAMETROS")
    print("=" * 80)

    model = Qwen3Embedding()

    # ====================================================================
    # TEST 1: Barrido de THRESHOLD (con instruction especifica, dim 1024)
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 1: BARRIDO DE THRESHOLD")
    print(f"  Instruction: especifica | Dim: 1024")
    print(f"{'='*80}")

    peers = build_peers_standard()
    # Pre-encode facets once
    all_texts = []
    facet_map = []
    for pi, p in enumerate(peers):
        for fi, f in enumerate(p.facets):
            all_texts.append(f.text)
            facet_map.append((pi, fi))
    facet_embs = model.encode(all_texts, is_query=False, dim=-1, batch_size=16)
    for idx, (pi, fi) in enumerate(facet_map):
        peers[pi].facets[fi].embedding = facet_embs[idx]

    # Pre-encode queries
    query_texts = [p[0] for p in test_phrases]
    expected = [p[1] for p in test_phrases]
    query_embs = model.encode(query_texts, is_query=True,
                              instruction=INSTRUCTIONS["especifica"], dim=-1, batch_size=16)

    print(f"\n  {'Threshold':>10s} | {'R@1':>8s} | {'R@3':>8s} | {'R@any':>8s} | {'F1@3':>8s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    best_f1 = 0
    best_thresh = 0
    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        r1 = r3 = ra = 0
        total = len(test_phrases)
        total_acts = 0
        correct_acts = 0

        for qi, (q_emb, exp) in enumerate(zip(query_embs, expected)):
            peer_scores = {}
            for p in peers:
                best = -1
                for f in p.facets:
                    score = float(np.dot(q_emb, f.embedding))
                    if score > best:
                        best = score
                peer_scores[p.id] = best

            sorted_peers = sorted(peer_scores.items(), key=lambda x: -x[1])
            above = [(pid, sc) for pid, sc in sorted_peers if sc >= thresh]

            top1_id = above[0][0] if above else None
            top3_ids = [pid for pid, _ in above[:3]]
            any_ids = [pid for pid, _ in above]

            if top1_id == exp: r1 += 1
            if exp in top3_ids: r3 += 1
            if exp in any_ids: ra += 1

            for pid, _ in above[:3]:
                total_acts += 1
                if pid == exp: correct_acts += 1

        precision = correct_acts / total_acts if total_acts > 0 else 0
        f1 = 2 * (r3/total * precision) / (r3/total + precision) if (r3/total + precision) > 0 else 0

        r1p = r1/total
        r3p = r3/total
        rap = ra/total

        print(f"  {thresh:>10.2f} | {r1p:>7.1%} | {r3p:>7.1%} | {rap:>7.1%} | {f1:>7.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n  >>> Mejor threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # ====================================================================
    # TEST 2: Barrido de DIMENSIONES MRL
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 2: DIMENSIONES MRL (Matryoshka)")
    print(f"  Instruction: especifica | Threshold: {best_thresh:.2f}")
    print(f"{'='*80}")

    print(f"\n  {'Dims':>6s} | {'R@1':>8s} | {'R@3':>8s} | {'R@any':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for dim in [-1, 1024, 512, 256, 128, 64]:
        # Re-encode everything with this dim
        facet_embs = model.encode(all_texts, is_query=False, dim=dim if dim != -1 else -1, batch_size=16)
        q_embs = model.encode(query_texts, is_query=True,
                              instruction=INSTRUCTIONS["especifica"],
                              dim=dim if dim != -1 else -1, batch_size=16)

        r1 = r3 = ra = 0
        for qi, (q_emb, exp) in enumerate(zip(q_embs, expected)):
            peer_scores = {}
            for p in peers:
                best = -1
                for fi, f in enumerate(p.facets):
                    f_emb = facet_embs[facet_map.index((peers.index(p), fi))] if (peers.index(p), fi) in facet_map else None
                    if f_emb is None:
                        for idx2, (pi2, fi2) in enumerate(facet_map):
                            if peers[pi2].id == p.id and fi2 == fi:
                                f_emb = facet_embs[idx2]
                                break
                    if f_emb is None:
                        continue
                    score = float(np.dot(q_emb, f_emb))
                    if score > best:
                        best = score
                peer_scores[p.id] = best

            sorted_p = sorted(peer_scores.items(), key=lambda x: -x[1])
            above = [(pid, sc) for pid, sc in sorted_p if sc >= best_thresh]

            if above and above[0][0] == exp: r1 += 1
            if exp in [pid for pid, _ in above[:3]]: r3 += 1
            if exp in [pid for pid, _ in above]: ra += 1

        dim_label = f"{dim}d" if dim != -1 else "full"
        print(f"  {dim_label:>6s} | {r1/len(test_phrases):>7.1%} | {r3/len(test_phrases):>7.1%} | {ra/len(test_phrases):>7.1%}")

    # ====================================================================
    # TEST 3: Barrido de INSTRUCTIONS
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"  TEST 3: INSTRUCTIONS")
    print(f"  Threshold: {best_thresh:.2f} | Dim: 1024")
    print(f"{'='*80}")

    # Re-encode facets at 1024d
    facet_embs_1024 = model.encode(all_texts, is_query=False, dim=1024, batch_size=16)

    print(f"\n  {'Instruction':>18s} | {'R@1':>8s} | {'R@3':>8s} | {'R@any':>8s}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for inst_name, inst_text in INSTRUCTIONS.items():
        q_embs = model.encode(query_texts, is_query=True, instruction=inst_text, dim=1024, batch_size=16)

        r1 = r3 = ra = 0
        for qi, (q_emb, exp) in enumerate(zip(q_embs, expected)):
            peer_scores = {}
            for p in peers:
                best = -1
                for fi, f in enumerate(p.facets):
                    for idx2, (pi2, fi2) in enumerate(facet_map):
                        if peers[pi2].id == p.id and fi2 == fi:
                            score = float(np.dot(q_emb, facet_embs_1024[idx2]))
                            if score > best:
                                best = score
                            break
                peer_scores[p.id] = best

            sorted_p = sorted(peer_scores.items(), key=lambda x: -x[1])
            above = [(pid, sc) for pid, sc in sorted_p if sc >= best_thresh]

            if above and above[0][0] == exp: r1 += 1
            if exp in [pid for pid, _ in above[:3]]: r3 += 1
            if exp in [pid for pid, _ in above]: ra += 1

        print(f"  {inst_name:>18s} | {r1/len(test_phrases):>7.1%} | {r3/len(test_phrases):>7.1%} | {ra/len(test_phrases):>7.1%}")

    # ====================================================================
    # TEST 4: Combinacion ganadora — reporte final
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"  RESUMEN C9")
    print(f"{'='*80}")
    print(f"  Modelo: Qwen3-Embedding-8B")
    print(f"  Mejor threshold: {best_thresh:.2f} (F1={best_f1:.3f})")
    print(f"  C7 baseline (MiniLM): R@1=83.3% R@3=93.1% R@any=95.1%")
    print(f"  C8 default (Qwen3):  R@1=63.7% R@3=88.2% R@any=100%")
    print(f"  Ver arriba para mejor combinacion de threshold/dims/instruction")


if __name__ == "__main__":
    main()
