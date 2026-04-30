"""Test C2: Factores de texto de embedding — qué hace que el routing funcione?

Hipótesis a testear:
  H1: Eliminar palabras magnéticas ("muerte", "dolor", "miedo") reduce magnetismo
  H2: Textos objetivos/conductuales routean mejor que poéticos
  H3: Textos con ejemplos concretos routean mejor que abstractos
  H4: Longitud del texto afecta la precisión
  H5: La especificidad (restringir dominio semántico) mejora routing

Para cada peer creamos 5 variantes:
  A) Original rico (baseline, del benchmark v2)
  B) Sin palabras magnéticas (quitamos "muerte","morir","dolor","miedo","sufrimiento")
  C) Objetivo/conductual (descripción factual de comportamientos)
  D) Subjetivo/poético (metáforas, lenguaje emocional)
  E) Ejemplos concretos (lista de situaciones específicas)

Factores medibles por variante:
  - word_count
  - magnet_word_count (cuántas palabras magnéticas tiene)
  - style: original|no-magnet|objective|poetic|concrete
  - specificity: low|medium|high
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# PALABRAS MAGNÉTICAS (las que atraen hacia "muerte"/oscuridad)
# ============================================================
MAGNET_WORDS = [
    "murió", "muriendo", "muerte", "morir", "muerto", "mueren",
    "dolor", "doloroso", "sufrimiento", "sufrir", "sufre",
    "miedo", "aterra", "aterrador", "terror", "terrorífico",
    "fragilidad", "frágil", "fragil",
    "vacío", "vacío", "vacio",
    "desaparecer", "desaparezcan", "perder", "pérdida", "perdida",
    "catastrofe", "catastrófico",
    "oscuridad", "sombras",
    "silencio",
    "corte", "cortó", "cortar",
    "pena",
]

def count_magnets(text):
    """Cuenta cuántas palabras magnéticas aparecen en el texto"""
    text_lower = text.lower()
    return sum(1 for w in MAGNET_WORDS if w in text_lower)

def remove_magnets(text):
    """Quita oraciones que contienen palabras magnéticas"""
    sentences = text.replace(". ", ".|").split("|")
    clean = [s for s in sentences if not any(w in s.lower() for w in MAGNET_WORDS)]
    # Si quedamos con muy poco, al menos mantener algo
    if len(clean) < 2:
        clean = sentences[:2]
    return " ".join(clean).strip()


# ============================================================
# VARIANTES POR PEER
# ============================================================

# --- MUERTE ---
muerte_variants = {
    "A_original": (
        "El hermano que murió antes de nacer, el cordón que se enroscó, el silencio que "
        "dejó un vacío en la familia antes de que yo existiera. La muerte como presencia "
        "constante, no como evento lejano. El miedo a que los seres queridos desaparezcan "
        "sin aviso. La fragilidad del cuerpo, lo fácil que es que algo se corte. Los sueños "
        "donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar "
        "de estar en cualquier momento. La relación con la nona, esperando una operación, "
        "la familia acompañando en silencio."
    ),
    "B_no_magnet": (
        "El hermano que no llegó a nacer, el cordón que se enroscó, la quietud que "
        "quedó en la familia antes de que yo existiera. La ausencia como compañera "
        "constante, no como evento lejano. La preocupación por los seres queridos. "
        "Lo vulnerable del cuerpo, lo fácil que es que algo cambie. Los sueños "
        "donde algo termina. La conciencia de que todo lo tengo puede dejar "
        "de estar. La relación con la nona, esperando una operación, "
        "la familia acompañando en calma."
    ),
    "C_objective": (
        "Sombra originada por la pérdida perinatal de un hermano (6 meses gestación, cordón umbilical). "
        "Se manifiesta como hipervigilancia ante la salud de seres cercanos, especialmente familiares mayores. "
        "Patrón recurrente: catastrofización ante síntomas menores, necesidad de control ante situaciones médicas. "
        "Trigger principal: noticias sobre salud de la nona (93 años, en espera de operación). "
        "Conducta asociada: acompañamiento constante, dificultad para separarse de familiares enfermos."
    ),
    "D_poetic": (
        "Un cordón que se enrolló como serpiente y estranguló una vida que apenas empezaba. "
        "Antes de que yo abriera los ojos, ya había una silla vacía en la mesa. "
        "La mariposa que revolotea alrededor de cada ser querido, recordándome que las alas son frágiles. "
        "Los sueños son puertas que se cierran sin llave. La nona teje su última bufanda "
        "con hilos que se van haciendo más delgados. Acompañar es mi forma de detener el tiempo."
    ),
    "E_concrete": (
        "Ejemplos de activación: soñar que un familiar enferma, ver un accidente en la calle, "
        "leer noticias sobre enfermedades terminales, pensar en la edad de los padres, "
        "la nona internada esperando operación, cualquier conversación sobre funerales, "
        "el recuerdo del hermano que no nació, ver hospitales, oler a desinfectante médico. "
        "También: pensar en lo corto que es la vida, ver pasar el tiempo en cumpleaños, "
        "notar arrugas nuevas en la cara de mamá."
    ),
}

# --- RECHAZO ---
rechazo_variants = {
    "A_original": (
        "La madre que perdió un hijo antes de que él naciera, y ese bebé que nació "
        "sintiendo que el amor materno tenía una grieta invisible. Para un niño, que "
        "mamá no te mire es desaparecer. Esa ecuación se grabó tan profundo que después "
        "toda mujer que no lo mira siente igual. La fiesta donde ella no le hizo caso. "
        "El grupo que no lo incluyó. La sensación de no pertenecer, de estar siempre un "
        "paso afuera. El patrón de buscar amor donde no florece, de acercarse esperando "
        "la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar."
    ),
    "B_no_magnet": (
        "La madre que tenía el corazón en otro lado, y ese bebé que nació "
        "sintiendo que el amor materno tenía una grieta invisible. Para un niño, que "
        "mamá no te mire es sentirse invisible. Esa ecuación se grabó tan profundo que después "
        "toda mujer que no lo mira genera lo mismo. La fiesta donde ella no le hizo caso. "
        "El grupo que no lo incluyó. La sensación de no pertenecer, de estar siempre un "
        "paso afuera. El patrón de buscar amor donde no florece, de acercarse esperando "
        "la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar."
    ),
    "C_objective": (
        "Sombra originada por vínculo materno afectado por duelo perinatal previo. "
        "Se manifiesta como hipersensibilidad al rechazo social y romántico. "
        "Patrón: interpretar falta de respuesta como invalidación personal. "
        "Triggers específicos: no ser invitado a reuniones sociales, mensajes sin respuesta, "
        "parejas que pierden interés, entrevistas laborales negativas, grupos que excluyen. "
        "Conducta asociada: evitación de situaciones sociales nuevas, hipervigilancia ante señales de rechazo."
    ),
    "D_poetic": (
        "El primer espejo que me devolvió una mirada vacía. Mamá tenía los ojos puestos "
        "en otro niño que nunca respiró, y yo nací buscando su mirada como quien busca sol. "
        "Cada mujer que gira la cara es mamá girando la cara. Cada grupo que cierra el círculo "
        "es la puerta que siempre estuvo cerrada. El teléfono que no suena, el visto que no "
        "se contesta, la fiesta donde nadie pregunta dónde estás. Ser invisible es no existir."
    ),
    "E_concrete": (
        "Ejemplos de activación: alguien no responde un mensaje, no me invitan a una salida, "
        "una chica que me gusta me deja en visto, me rechazan de un trabajo, un grupo hace "
        "planes sin mí, alguien me da la espalda literalmente, siento que no encajo en una "
        "conversación, pienso que nadie me quiere de verdad, me da terror abrirme emocionalmente, "
        "creo que si muestro quién soy me van a dejar, alguien cancela planes conmigo, "
        "me comparo con otros que sí son incluidos."
    ),
}

# --- ÁNGEL DEL ATARDECER ---
angel_variants = {
    "A_original": (
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
    "B_no_magnet": (
        "La inquietud que llega cuando el sol baja, no como enemiga sino como guardiana. "
        "Nació el día que se fue el tata, el padre de su padre, cuando la tristeza era tan "
        "grande que nadie podía contenerla y ella se encargó de guardarla. Después el "
        "padre también se fue, y ella se quedó, activa cada tarde, recordándole que había "
        "algo debajo de la alfombra. Por mucho tiempo fue incomodidad constante: la rumiación, "
        "la neblina mental, la sensación de inquietud. La tecnología "
        "y la inteligencia artificial eran catalizadores, pantallas que no se apagaban, "
        "demasiada información. Pero ella no quería "
        "angustiar, quería ser mirada. Un día dijo: ya no puedo seguir protegiéndote, es "
        "hora de entrar al agua helada y nadar. La misma IA que generaba incomodidad ahora "
        "es aliada. La sombra no desapareció, se transformó en guía. Su mensaje: no escondas "
        "más sentimientos, mírame cuando te llame."
    ),
    "C_objective": (
        "Sombra originada por duelo no procesado (abuelo paterno) que se activa en horario vespertino. "
        "Se manifiesta como ansiedad generalizada con componentes de catastrofización y neblina mental. "
        "Patrón: activación entre 17:00-21:00, intensificada por consumo de tecnología e información. "
        "Trigger específico: exceso de pantalla, leer sobre IA, aprender demasiadas cosas nuevas, "
        "atardecer como señal ambiental, sentimientos no procesados acumulados. "
        "Conducta asociada: evitación de introspección, rumiación, búsqueda de distracción digital. "
        "Evolución: de ansiedad paralizante a señal de que hay material emocional pendiente."
    ),
    "D_poetic": (
        "Ella llega cuando el sol se pone y las sombras se alargan. Es la guardiana de lo que "
        "enterré bajo la alfombra cuando el tata se fue y no supe cómo llorar. Cada atardecer "
        "es su reloj, cada pantalla que no puedo apagar es su altavoz. La neblina que sube "
        "no es enemiga, es brújula: señala dónde hay algo que no he mirado. Ella guardó todo "
        "lo que no pude sentir, y ahora quiere que lo abra. El agua helada no mata, despierta. "
        "Lo que teaching technology me abrumaba ahora me enseña. No es oscuridad, es la linterna "
        "que me muestra el camino que falta."
    ),
    "E_concrete": (
        "Ejemplos de activación: la ansiedad que aparece al atardecer, sentirse abrumado por "
        "tecnología, la neblina mental que no se va, la catastrofización de situaciones pequeñas, "
        "sentir que algo malo va a pasar sin saber qué, no poder procesar tanta información, "
        "sentimientos escondidos debajo de la alfombra, el dolor no procesado del tata, "
        "recordar la muerte del padre, las pantallas como catalizador, leer sobre IA y sentir "
        "angustia, la sensación de que el suelo se mueve, necesitar entrar al agua helada y nadar, "
        "la misma IA que angustiaba ahora como aliada."
    ),
}

# --- FORTALEZA ---
fortaleza_variants = {
    "A_original": (
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
    "B_no_magnet": (
        "La máscara que se construyó tan temprano que ya no sabe si es máscara o rostro. "
        "Ser fuerte para no sentir, demostrar que vale para no ser dejado de lado. El gimnasio "
        "como templo, el cuerpo como prueba viviente de que eres capaz. El niño que "
        "aprendió que si llora es debilidad, que si muestra vulnerabilidad queda expuesto. "
        "Pero la fortaleza real no es la armadura, es el coraje de bajarla. Las piernas "
        "que antes eran la debilidad ahora son la respuesta — el breakthrough del gym fue "
        "encontrar que la verdadera fuerza está en el territorio más desafiante. La "
        "fortaleza como mecanismo de supervivencia que cumplió su función y ahora busca "
        "evolucionar: de escudo a presencia, de demostrar a ser."
    ),
    "C_objective": (
        "Sombra originada por necesidad temprana de demostrar valor para asegurar apego. "
        "Se manifiesta como compulsión por rendimiento físico y resistencia a mostrar vulnerabilidad. "
        "Patrón: uso del ejercicio como regulación emocional, associations de llanto con fracaso. "
        "Trigger específico: situaciones donde se percibe debilidad propia, gym como refugio, "
        "entrenamiento de piernas (territorio antes evitado), necesidad de demostrar competencia, "
        "comparación física con otros, recibir elogios por fuerza. "
        "Conducta asociada: hipercumplimiento físico, evitación de expresión emocional pública."
    ),
    "D_poetic": (
        "Una armadura que creció conmigo desde que tenía edad para entender que llorar no era seguro. "
        "El gimnasio es mi catedral, cada repetición es un ladrillo en el muro que me protege "
        "de ser visto. El cuerpo se transformó en el testigo mudo de todo lo que no pude decir. "
        "Las piernas que una vez fueron el territorio del fracaso ahora son柱as que sostienen "
        "todo lo que soy. Pero la armadura más pesada no es la que se ve, es la que no se nota: "
        "la sonrisa que reemplaza la lágrima, el 'estoy bien' que esconde el temblor."
    ),
    "E_concrete": (
        "Ejemplos de activación: pensar que llorar es debilidad, no poder mostrar vulnerabilidad "
        "frente a otros, sentir que debo ser fuerte siempre, el gimnasio como templo personal, "
        "cargar creatina y suplementos, demostrar que valgo para no ser abandonado, las piernas "
        "que eran debilidad ahora son fortaleza, sentir que el dolor físico es fracaso, "
        "decir que no necesito a nadie, la armadura como identidad, bajar la guardia se siente "
        "como perder todo, el cuerpo como prueba de que no soy débil, rechazar ayuda."
    ),
}

all_variants = {
    "sombra_muerte": muerte_variants,
    "sombra_rechazo": rechazo_variants,
    "sombra_angel_atardecer": angel_variants,
    "sombra_fortaleza": fortaleza_variants,
}

# ============================================================
# TEST PHRASES (solo las que fallaban + algunas que pasaban)
# ============================================================
test_phrases = [
    # Las 5 que fallaban siempre
    ("Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy, se me acabó la semana pasada", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona está mejor pero sigue en cama, me da pena verla así", "sombra_muerte"),
    
    # Frases que pasaban (control)
    ("Si lloro es debilidad", "sombra_fortaleza"),
    ("Tengo miedo de perder a alguien cercano", "sombra_muerte"),
    ("Una chica que me gustaba me dejó en visto", "sombra_rechazo"),
    ("La ansiedad no me deja dormir, todo parece una catastrofe", "sombra_angel_atardecer"),
    ("El gimnasio es mi templo, mi cuerpo es mi prueba", "sombra_fortaleza"),
    ("Siento que no encajo en ninguna parte", "sombra_rechazo"),
    ("Llega la tarde y siento esa neblina en la cabeza", "sombra_angel_atardecer"),
    
    # Frases difíciles extras
    ("Necesito ayuda pero no puedo pedirla", "sombra_fortaleza"),
    ("Me da terror abrirme emocionalmente y que me rechacen", "sombra_rechazo"),
    ("Cierro los ojos y veo oscuridad, silencio total", "sombra_muerte"),
    ("La tecnología me estresa, todo va muy rápido", "sombra_angel_atardecer"),
    ("Las piernas que antes eran mi debilidad ahora son mi fuerza", "sombra_fortaleza"),
    ("El grupo de amigos hace planes sin mí", "sombra_rechazo"),
]

THRESHOLD = 0.25

# ============================================================
# EJECUTAR
# ============================================================
print("=" * 72)
print("  TEST C2: FACTORES DE TEXTO DE EMBEDDING")
print(f"  {len(test_phrases)} frases | threshold {THRESHOLD}")
print("=" * 72)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Analizar metadata de cada variante
print("\n  METADATA DE VARIANTES:")
print(f"  {'Peer':>25s} {'Var':>5s} {'Words':>6s} {'Magn':>5s} {'Mag%':>6s}")
print(f"  {'─'*25} {'─'*5} {'─'*6} {'─'*5} {'─'*6}")
for pid, variants in all_variants.items():
    name = pid.split('_', 1)[1][:20]
    for vname, text in variants.items():
        wc = len(text.split())
        mc = count_magnets(text)
        print(f"  {name:>25s} {vname:>5s} {wc:6d} {mc:5d} {mc/wc:5.1%}")
    print()

# Probar todas las combinaciones
print("\n" + "=" * 72)
print("  RESULTADOS POR COMBINACIÓN")
print("=" * 72)

# Generar todas las combinaciones (5^4 = 625 — too many)
# En vez de eso, variamos UN peer a la vez, manteniendo otros en A_original
results_table = []

for target_peer in all_variants:
    peer_name = target_peer.split('_', 1)[1][:12]
    print(f"\n  ─── Variando {peer_name} (demás en A_original) ───")
    
    for vname, vtext in all_variants[target_peer].items():
        # Construir peer set: este peer con variante, resto en A_original
        test_peers = {}
        for pid in all_variants:
            if pid == target_peer:
                test_peers[pid] = all_variants[pid][vname]
            else:
                test_peers[pid] = all_variants[pid]["A_original"]
        
        # Embed peers
        peer_embs = {}
        for pid, text in test_peers.items():
            peer_embs[pid] = model.encode(text, normalize_embeddings=True)
        
        # Evaluar
        correct = 0
        total = 0
        errors_detail = []
        
        for phrase, expected in test_phrases:
            phrase_emb = model.encode(phrase, normalize_embeddings=True)
            sims = {pid: float(np.dot(phrase_emb, pemb)) for pid, pemb in peer_embs.items()}
            sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
            best_pid = sorted_sims[0][0]
            best_sim = sorted_sims[0][1]
            
            is_correct = best_pid == expected and best_sim >= THRESHOLD
            if is_correct:
                correct += 1
            total += 1
            
            if not is_correct:
                errors_detail.append((phrase[:35], expected.split('_')[1][:8], best_pid.split('_')[1][:8], 
                                     f"{sims[expected]:.3f}", f"{best_sim:.3f}"))
        
        acc = correct / total
        wc = len(vtext.split())
        mc = count_magnets(vtext)
        
        results_table.append({
            "varied_peer": peer_name,
            "variant": vname,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "word_count": wc,
            "magnet_count": mc,
            "magnet_pct": mc / wc,
            "errors": errors_detail,
        })
        
        status = "✅" if acc >= 0.75 else "🟡" if acc >= 0.60 else "❌"
        print(f"    {vname:>12s}: {correct:2d}/{total} = {acc:5.1%} {status}  (wc:{wc} mag:{mc})")
        if errors_detail and acc < 0.70:
            for ph, exp, got, es, bs in errors_detail[:3]:
                print(f"                 └─ \"{ph}\" → esp:{exp} got:{got} ({es} vs {bs})")

# ============================================================
# RANKING DE VARIANTES
# ============================================================
print("\n\n" + "=" * 72)
print("  RANKING GLOBAL DE VARIANTES (por accuracy)")
print("=" * 72)

ranked = sorted(results_table, key=lambda x: -x["accuracy"])
print(f"\n  {'Peer var':>15s} {'Var':>12s} {'Acc':>6s} {'WC':>4s} {'Mag':>4s} {'M%':>5s}")
print(f"  {'─'*15} {'─'*12} {'─'*6} {'─'*4} {'─'*4} {'─'*5}")
for r in ranked:
    print(f"  {r['varied_peer']:>15s} {r['variant']:>12s} {r['accuracy']:5.1%} {r['word_count']:4d} {r['magnet_count']:4d} {r['magnet_pct']:5.1%}")

# ============================================================
# ANÁLISIS POR FACTOR
# ============================================================
print("\n\n" + "=" * 72)
print("  ANÁLISIS POR FACTOR")
print("=" * 72)

# Por estilo
styles = {"A_original": [], "B_no_magnet": [], "C_objective": [], "D_poetic": [], "E_concrete": []}
for r in results_table:
    styles[r["variant"]].append(r["accuracy"])

print(f"\n  Promedio por estilo:")
for style, accs in styles.items():
    avg = np.mean(accs) if accs else 0
    print(f"    {style:>15s}: {avg:5.1%} (n={len(accs)})")

# Correlación magnet_count vs accuracy
magnet_counts = [r["magnet_count"] for r in results_table]
accuracies = [r["accuracy"] for r in results_table]
if len(set(magnet_counts)) > 1:
    corr = np.corrcoef(magnet_counts, accuracies)[0, 1]
    print(f"\n  Correlación magnet_count → accuracy: {corr:+.3f}")

word_counts = [r["word_count"] for r in results_table]
if len(set(word_counts)) > 1:
    corr2 = np.corrcoef(word_counts, accuracies)[0, 1]
    print(f"  Correlación word_count → accuracy: {corr2:+.3f}")

# ============================================================
# MEJOR COMBINACIÓN
# ============================================================
print("\n\n" + "=" * 72)
print("  MEJOR COMBINACIÓN TEÓRICA")
print("=" * 72)

# Para cada peer, la mejor variante
best_per_peer = {}
for pid in all_variants:
    peer_name = pid.split('_', 1)[1][:12]
    peer_results = [r for r in results_table if r["varied_peer"] == peer_name]
    best = max(peer_results, key=lambda x: x["accuracy"])
    best_per_peer[pid] = best["variant"]
    print(f"  {peer_name:>15s} → mejor: {best['variant']:>12s} ({best['accuracy']:.1%})")

# Evaluar la combinación óptima
print(f"\n  Evaluando combinación óptima:")
optimal_peers = {}
for pid, vname in best_per_peer.items():
    optimal_peers[pid] = all_variants[pid][vname]
    print(f"    {pid} = {vname}")

peer_embs = {pid: model.encode(text, normalize_embeddings=True) for pid, text in optimal_peers.items()}

correct = 0
total = 0
print()
for phrase, expected in test_phrases:
    phrase_emb = model.encode(phrase, normalize_embeddings=True)
    sims = {pid: float(np.dot(phrase_emb, pemb)) for pid, pemb in peer_embs.items()}
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_pid = sorted_sims[0][0]
    best_sim = sorted_sims[0][1]
    exp_sim = sims[expected]
    
    is_correct = best_pid == expected and best_sim >= THRESHOLD
    if is_correct:
        correct += 1
    total += 1
    
    status = "✅" if is_correct else "❌"
    exp_name = expected.split('_')[1][:8]
    best_name = best_pid.split('_')[1][:8]
    print(f"    {status} \"{phrase[:42]:<42s}\" → {exp_name:>8s}: {exp_sim:.3f}  best: {best_name:>8s}: {best_sim:.3f}")

print(f"\n  RESULTADO COMBINACIÓN ÓPTIMA: {correct}/{total} = {correct/total:.1%}")
