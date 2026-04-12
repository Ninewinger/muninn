"""Test C4: Combinacion optima (principles + examples) + variaciones"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.25

test_phrases = [
    # Las 5 problematicas
    ("Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy, se me acabo la semana pasada", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona esta mejor pero sigue en cama, me da pena verla asi", "sombra_muerte"),
    # Control que pasaban
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
]

# Frases extras nunca vistas para test mas duro
extra_phrases = [
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
    ("Pienso en que pasaria si mi papá se enferma", "sombra_muerte"),
    ("La computadora me dio panico, demasiadas ventanas", "sombra_angel_atardecer"),
    ("Deje de ir al gym tres dias y me senti fracasado", "sombra_fortaleza"),
    ("Una amiga me dijo que la deje en visto y me senti mal", "sombra_rechazo"),
    ("Vi una foto de mi abuelo joven y me puse triste", "sombra_muerte"),
    ("Me cuesta concentrarme, la mente va muy rapido", "sombra_angel_atardecer"),
    ("Un nino lloraba en la calle y no supe que hacer", "sombra_fortaleza"),
]

all_phrases = test_phrases + extra_phrases

PEER_IDS = ["sombra_muerte", "sombra_rechazo", "sombra_angel_atardecer", "sombra_fortaleza"]
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def evaluate(peer_texts, label="", phrases=None):
    if phrases is None:
        phrases = all_phrases
    peer_embs = {pid: model.encode(peer_texts[pid], normalize_embeddings=True) for pid in PEER_IDS}
    correct = 0
    total = len(phrases)
    details = []
    for phrase, expected in phrases:
        phrase_emb = model.encode(phrase, normalize_embeddings=True)
        sims = {pid: float(np.dot(phrase_emb, peer_embs[pid])) for pid in PEER_IDS}
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_pid, best_sim = sorted_sims[0]
        exp_sim = sims[expected]
        ok = best_pid == expected and best_sim >= THRESHOLD
        if ok:
            correct += 1
        details.append({
            "phrase": phrase, "expected": expected, "best": best_pid,
            "exp_sim": exp_sim, "best_sim": best_sim, "ok": ok,
            "sims": sims
        })
    acc = correct / total
    return {"acc": acc, "correct": correct, "total": total, "details": details, "label": label}


# ====================================================================
# COMBINACION OPTIMA: principles + examples por peer
# ====================================================================
optimal = {
    "sombra_muerte": (
        "Premisa: todo lo que existe puede dejar de existir sin aviso. La vulnerabilidad del cuerpo "
        "es un hecho. El duelo perinatal del hermano establecio el patron original: la vida puede "
        "cortarse antes de empezar. La hipervigilancia ante la salud es mecanismo de supervivencia. "
        "La conciencia de finitud es constante. Los suenos terminales son proyecciones del patron. "
        "El acompanamiento es defensa contra la perdida anticipada. "
        "Se activa con: suenar que un familiar enferma, ver un accidente, leer sobre enfermedades, "
        "pensar en la edad de los padres, la nona internada esperando operacion, conversaciones sobre "
        "funerales, ver hospitales, pensar en lo corta que es la vida, ver arrugas en la cara de "
        "mama, notar que los abuelos envejecen, cumpleanos como cuenta regresiva, cualquier noticia "
        "de salud, recordar al hermano que no nacio."
    ),
    "sombra_rechazo": (
        "Premisa: si me rechazan, dejo de importar. El rechazo es existencial, no solo social. "
        "El vinculo materno marco el patron: amor condicional a ser visto. La hipersensibilidad "
        "al rechazo romantico es transferencia del rechazo materno original. La no-respuesta es "
        "una respuesta que confirma la narrativa de no pertenencia. Buscar amor donde no florece "
        "es patron recurrente. La exclusion grupal confirma la hipotesis central. "
        "Se activa con: alguien no responde un mensaje, no me invitan a una salida, una chica "
        "que me gusta me deja en visto, me rechazan de un trabajo, un grupo hace planes sin mi, "
        "alguien me da la espalda, no encajo en una conversacion, pienso que nadie me quiere, "
        "me da terror abrirme emocionalmente, creo que si muestro quien soy me van a dejar, "
        "alguien cancela planes, me comparo con otros que si son incluidos, una fiesta donde "
        "nadie me hizo caso."
    ),
    "sombra_angel_atardecer": (
        "Premisa: la inquietud vespertina es brujula, no enemiga. Las pantallas intensifican porque "
        "agregan informacion sin procesar. La neblina mental es senal de material emocional pendiente. "
        "La catastrofizacion es el intento de dar forma a lo amorfo. El duelo no procesado del tata "
        "es el origen. La evolucion es de paralis a senal navegable. El mensaje: no escondas mas "
        "sentimientos, mirame cuando te llame. "
        "Se activa con: ansiedad al atardecer, sentirse abrumado por tecnologia, neblina mental "
        "que no se va, catastrofizar situaciones pequenas, sentir que algo malo va a pasar, "
        "no poder procesar tanta informacion, sentimientos escondidos bajo la alfombra, "
        "recordar duelos no procesados, pantallas como catalizador, leer sobre IA y sentir "
        "angustia, la sensacion de que el suelo se mueve, demasiado tiempo frente a pantallas, "
        "aprender demasiadas cosas nuevas sin pausa."
    ),
    "sombra_fortaleza": (
        "Premisa: el valor personal se demuestra a traves de la capacidad fisica y la resistencia "
        "al dolor emocional. La vulnerabilidad expuesta equivale a perdida de proteccion. El "
        "rendimiento corporal es evidencia de competencia. Llorar es admitir derrota. La verdadera "
        "fortaleza no es la armadura, es el coraje de bajarla. El gym es regulacion emocional "
        "sustitutiva. Las piernas simbolizan el territorio conquistado. La armadura busca "
        "evolucionar de escudo a presencia. "
        "Se activa con: pensar que llorar es debilidad, no poder mostrar vulnerabilidad, sentir "
        "que debo ser fuerte siempre, el gym como templo personal, cargar creatina y suplementos, "
        "demostrar que valgo para no ser abandonado, las piernas como fortaleza personal, sentir "
        "dolor como fracaso, decir que no necesito a nadie, bajar la guardia se siente como "
        "perder todo, rechazar ayuda, el cuerpo como prueba de competencia, alguien me dice que "
        "soy fuerte y me siento vacio."
    ),
}

# ====================================================================
# VARIACIONES ALREDEDOR DEL OPTIMO
# ====================================================================

# V2: principles + examples + 1ra persona sutil
optimal_v2 = {
    "sombra_muerte": (
        "Mi premisa: todo lo que existe puede dejar de existir sin aviso. La vulnerabilidad "
        "del cuerpo es un hecho que vivo cada dia. El duelo perinatal de mi hermano establecio "
        "mi patron original. Mi hipervigilancia ante la salud es mi mecanismo de supervivencia. "
        "Mi conciencia de finitud es constante. Me activo cuando: sueno que un familiar enferma, "
        "veo un accidente, leo sobre enfermedades, pienso en la edad de mis padres, la nona "
        "internada, conversaciones sobre funerales, veo hospitales, pienso en lo corta que es "
        "la vida, veo arrugas en la cara de mama, noto que los abuelos envejecen, cumpleanos "
        "como cuenta regresiva, cualquier noticia de salud."
    ),
    "sombra_rechazo": (
        "Mi premisa: si me rechazan, dejo de importar. El rechazo es existencial para mi. "
        "Mi vinculo materno marco el patron: amor condicional a ser visto. Mi hipersensibilidad "
        "al rechazo romantico es transferencia del rechazo materno original. Para mi la "
        "no-respuesta es una respuesta que confirma que no pertenezco. Buscar amor donde no "
        "florece es mi patron recurrente. Me activo cuando: alguien no responde un mensaje, "
        "no me invitan a una salida, una chica me deja en visto, me rechazan de un trabajo, "
        "un grupo hace planes sin mi, alguien me da la espalda, no encajo, pienso que nadie "
        "me quiere, me da terror abrirme emocionalmente, alguien cancela planes, me comparo "
        "con otros que si son incluidos."
    ),
    "sombra_angel_atardecer": (
        "Mi premisa: mi inquietud vespertina es brujula, no enemiga. Las pantallas me intensifican "
        "porque agregan informacion sin procesar. Mi neblina mental es senal de material pendiente. "
        "Mi catastrofizacion es mi intento de dar forma a lo amorfo. El duelo no procesado del tata "
        "es mi origen. Mi mensaje: no escondas mas sentimientos. Me activo cuando: llega la "
        "ansiedad al atardecer, me siento abrumado por tecnologia, la neblina mental no se va, "
        "catastrofizo situaciones pequenas, siento que algo malo va a pasar, no puedo procesar "
        "tanta informacion, hay sentimientos escondidos bajo la alfombra, las pantallas son "
        "catalizador, leo sobre IA y siento angustia, aprendo demasiadas cosas sin pausa."
    ),
    "sombra_fortaleza": (
        "Mi premisa: mi valor personal se demuestra a traves de mi capacidad fisica y mi "
        "resistencia al dolor emocional. Mi vulnerabilidad expuesta equivale a perder proteccion. "
        "Mi rendimiento corporal es mi evidencia de competencia. Llorar es admitir derrota para mi. "
        "Mi verdadera fortaleza no es mi armadura, es el coraje de bajarla. Mi gym es mi "
        "regulacion emocional. Mis piernas simbolizan mi territorio conquistado. Me activo cuando: "
        "pienso que llorar es debilidad, no puedo mostrar vulnerabilidad, siento que debo ser "
        "fuerte siempre, el gym es mi templo, cargo creatina y suplementos, demuestro que valgo, "
        "mis piernas son mi fortaleza, siento dolor como fracaso, digo que no necesito a nadie, "
        "bajar la guardia se siente como perder todo, rechazo ayuda."
    ),
}

# V3: examples primero (reverse order) — hipótesis: lo primero tiene mas peso en el embedding
optimal_v3 = {
    "sombra_muerte": (
        "Se activa con: suenar que un familiar enferma, ver un accidente, leer sobre enfermedades, "
        "pensar en la edad de los padres, la nona internada esperando operacion, conversaciones sobre "
        "funerales, ver hospitales, pensar en lo corta que es la vida, ver arrugas en la cara de "
        "mama, notar que los abuelos envejecen, cumpleanos como cuenta regresiva, cualquier noticia "
        "de salud, recordar al hermano que no nacio. "
        "Premisa: todo lo que existe puede dejar de existir sin aviso. La vulnerabilidad del cuerpo "
        "es un hecho. El duelo perinatal del hermano establecio el patron original. La hipervigilancia "
        "ante la salud es mecanismo de supervivencia. La conciencia de finitud es constante."
    ),
    "sombra_rechazo": (
        "Se activa con: alguien no responde un mensaje, no me invitan a una salida, una chica "
        "que me gusta me deja en visto, me rechazan de un trabajo, un grupo hace planes sin mi, "
        "alguien me da la espalda, no encajo en una conversacion, pienso que nadie me quiere, "
        "me da terror abrirme emocionalmente, creo que si muestro quien soy me van a dejar, "
        "alguien cancela planes, me comparo con otros que si son incluidos, una fiesta donde "
        "nadie me hizo caso. "
        "Premisa: si me rechazan, dejo de importar. El rechazo es existencial. El vinculo materno "
        "marco el patron. La no-respuesta confirma la narrativa de no pertenencia."
    ),
    "sombra_angel_atardecer": (
        "Se activa con: ansiedad al atardecer, sentirse abrumado por tecnologia, neblina mental "
        "que no se va, catastrofizar situaciones pequenas, sentir que algo malo va a pasar, "
        "no poder procesar tanta informacion, sentimientos escondidos bajo la alfombra, "
        "pantallas como catalizador, leer sobre IA y sentir angustia, aprender demasiadas cosas "
        "nuevas sin pausa. "
        "Premisa: la inquietud vespertina es brujula, no enemiga. La neblina mental es senal "
        "de material pendiente. El duelo no procesado del tata es el origen. El mensaje: "
        "no escondas mas sentimientos, mirame cuando te llame."
    ),
    "sombra_fortaleza": (
        "Se activa con: pensar que llorar es debilidad, no poder mostrar vulnerabilidad, sentir "
        "que debo ser fuerte siempre, el gym como templo personal, cargar creatina y suplementos, "
        "demostrar que valgo para no ser abandonado, las piernas como fortaleza personal, sentir "
        "dolor como fracaso, decir que no necesito a nadie, bajar la guardia se siente como "
        "perder todo, rechazar ayuda, el cuerpo como prueba de competencia. "
        "Premisa: el valor personal se demuestra a traves de la capacidad fisica. La vulnerabilidad "
        "expuesta equivale a perdida de proteccion. Llorar es admitir derrota. La verdadera "
        "fortaleza es el coraje de bajar la armadura."
    ),
}

# V4: Solo examples sin principles (testear si principles aportan algo real)
optimal_v4 = {
    "sombra_muerte": (
        "Se activa con: suenar que un familiar enferma, ver un accidente, leer sobre enfermedades, "
        "pensar en la edad de los padres, la nona internada esperando operacion, conversaciones sobre "
        "funerales, ver hospitales, pensar en lo corta que es la vida, ver arrugas en la cara de "
        "mama, notar que los abuelos envejecen, cumpleanos como cuenta regresiva, cualquier noticia "
        "de salud, recordar al hermano que no nacio, pensar en la mortalidad de mis padres, "
        "el cuerpo fragil, el cordon umbilical del hermano, la cuenta regresiva."
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
        "que debo ser fuerte siempre, el gym como templo personal, cargar creatina y suplementos, "
        "demostrar que valgo para no ser abandonado, las piernas como fortaleza personal, sentir "
        "dolor como fracaso, decir que no necesito a nadie, bajar la guardia se siente como "
        "perder todo, rechazar ayuda, el cuerpo como prueba de competencia, la armadura como "
        "identidad, el coraje de bajarla."
    ),
}

# V5: Principles + examples + nombre del peer al inicio (contexto explicito)
optimal_v5 = {
    "sombra_muerte": (
        "SOMBRA MUERTE. Perdida, finitud, vulnerabilidad del cuerpo. "
        "Premisa: todo lo que existe puede dejar de existir sin aviso. El duelo perinatal del "
        "hermano establecio el patron original. La hipervigilancia ante la salud es mecanismo "
        "de supervivencia. Los suenos terminales son proyecciones del patron. "
        "Se activa con: suenar que un familiar enferma, ver un accidente, leer sobre enfermedades, "
        "pensar en la edad de los padres, la nona internada, conversaciones sobre funerales, "
        "ver hospitales, ver arrugas en la cara de mama, cumpleanos como cuenta regresiva, "
        "recordar al hermano que no nacio."
    ),
    "sombra_rechazo": (
        "SOMBRA RECHAZO. Exclusion, no pertenencia, no ser visto. "
        "Premisa: si me rechazan, dejo de importar. El rechazo es existencial. El vinculo materno "
        "marco el patron: amor condicional a ser visto. La no-respuesta confirma no pertenencia. "
        "Se activa con: alguien no responde un mensaje, no me invitan a una salida, una chica "
        "me deja en visto, me rechazan de un trabajo, un grupo hace planes sin mi, alguien me "
        "da la espalda, no encajo, pienso que nadie me quiere, me da terror abrirme emocionalmente, "
        "alguien cancela planes, me comparo con otros que si son incluidos."
    ),
    "sombra_angel_atardecer": (
        "SOMBRA ANGEL DEL ATARDECER. Inquietud vespertina, neblina mental, brujula emocional. "
        "Premisa: la inquietud vespertina es brujula, no enemiga. Las pantallas intensifican. "
        "La neblina mental es senal de material pendiente. El duelo del tata es el origen. "
        "Se activa con: ansiedad al atardecer, sentirse abrumado por tecnologia, neblina mental, "
        "catastrofizar situaciones, sentir que algo malo va a pasar, no poder procesar informacion, "
        "sentimientos bajo la alfombra, pantallas como catalizador, leer sobre IA y sentir angustia, "
        "aprender demasiadas cosas sin pausa."
    ),
    "sombra_fortaleza": (
        "SOMBRA FORTALEZA. Armadura, gym, rendimiento corporal, vulnerabilidad como exposicion. "
        "Premisa: el valor personal se demuestra con capacidad fisica y resistencia al dolor. "
        "La vulnerabilidad expuesta es perdida de proteccion. Llorar es admitir derrota. "
        "Se activa con: pensar que llorar es debilidad, no poder mostrar vulnerabilidad, sentir "
        "que debo ser fuerte siempre, el gym como templo, cargar creatina y suplementos, demostrar "
        "que valgo, las piernas como fortaleza, sentir dolor como fracaso, decir que no necesito "
        "a nadie, bajar la guardia como perder todo, rechazar ayuda."
    ),
}

# ====================================================================
# EJECUTAR TODOS
# ====================================================================
print("=" * 72)
print("  TEST C4: COMBINACION OPTIMA")
print(f"  {len(test_phrases)} frases base + {len(extra_phrases)} extras = {len(all_phrases)} total")
print("=" * 72)

variants = [
    ("V1: principles + examples", optimal),
    ("V2: principles + examples + 1ra pers", optimal_v2),
    ("V3: examples + principles (reversed)", optimal_v3),
    ("V4: solo examples (no principles)", optimal_v4),
    ("V5: label + principles + examples", optimal_v5),
]

# Test con frases originales primero
print("\n  --- FRASES ORIGINALES (18) ---")
for name, peers in variants:
    r = evaluate(peers, name, test_phrases)
    status = "OK" if r["acc"] >= 0.75 else "~" if r["acc"] >= 0.60 else "X"
    print(f"  [{status}] {r['label']:40s} {r['correct']:2d}/{r['total']} = {r['acc']:.1%}")
    errors = [d for d in r["details"] if not d["ok"]]
    for d in errors[:4]:
        print(f"       X \"{d['phrase'][:38]}\" -> esp:{d['expected'].split('_')[1][:6]} "
              f"got:{d['best'].split('_')[1][:6]} ({d['exp_sim']:.3f} vs {d['best_sim']:.3f})")

# Test con todas las frases
print("\n  --- TODAS LAS FRASES (38) ---")
all_results = []
for name, peers in variants:
    r = evaluate(peers, name, all_phrases)
    all_results.append(r)
    status = "OK" if r["acc"] >= 0.75 else "~" if r["acc"] >= 0.60 else "X"
    print(f"  [{status}] {r['label']:40s} {r['correct']:2d}/{r['total']} = {r['acc']:.1%}")

# Mejor variante - detalle completo
best = max(all_results, key=lambda x: x["acc"])
print(f"\n\n{'='*72}")
print(f"  MEJOR VARIANTE: {best['label']} ({best['acc']:.1%})")
print(f"{'='*72}")

print(f"\n  --- DETALLE POR PEER ---")
for pid in PEER_IDS:
    name = pid.split("_")[1][:12]
    peer_results = [d for d in best["details"] if d["expected"] == pid]
    correct = sum(1 for d in peer_results if d["ok"])
    total = len(peer_results)
    print(f"    {name:>15s}: {correct:2d}/{total} ({correct/total:.0%})")

print(f"\n  --- TODOS LOS ERRORES ---")
for d in best["details"]:
    if not d["ok"]:
        exp_name = d["expected"].split("_")[1][:8]
        best_name = d["best"].split("_")[1][:8]
        # Show all sim scores
        sims_str = "  ".join(f"{pid.split('_')[1][:4]}:{sim:.3f}" for pid, sim in 
                            sorted(d["sims"].items(), key=lambda x: -x[1]))
        print(f"    X \"{d['phrase'][:40]}\" -> esp:{exp_name} got:{best_name}")
        print(f"      {sims_str}")

# Comparacion directa: original vs optimo
print(f"\n\n{'='*72}")
print(f"  COMPARACION: ORIGINAL vs OPTIMO")
print(f"{'='*72}")

original = {
    "sombra_muerte": (
        "El hermano que murio antes de nacer, el cordon que se enrolllo, el silencio que "
        "dejo un vacio en la familia antes de que yo existiera. La muerte como presencia "
        "constante, no como evento lejano. El miedo a que los seres queridos desaparezcan "
        "sin aviso. La fragilidad del cuerpo, lo facil que es que algo se corte. Los suenos "
        "donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar "
        "de estar en cualquier momento. La relacion con la nona, esperando una operacion, "
        "la familia acompanando en silencio."
    ),
    "sombra_rechazo": (
        "La madre que perdio un hijo antes de que el naciera, y ese bebe que nacio "
        "sintiendo que el amor materno tenia una grieta invisible. Para un nino, que "
        "mama no te mire es desaparecer. Esa ecuacion se grabo tan profundo que despues "
        "toda mujer que no lo mira siente igual. La fiesta donde ella no le hizo caso. "
        "El grupo que no lo incluyo. La sensacion de no pertenecer, de estar siempre un "
        "paso afuera. El patron de buscar amor donde no florece, de acercarse esperando "
        "la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar."
    ),
    "sombra_angel_atardecer": (
        "La ansiedad que llega cuando el sol baja, no como enemiga sino como guardiana. "
        "Nacio el dia que murio el tata, el padre de su padre, cuando el dolor era tan "
        "grande que nadie podia entenderlo y ella se encargo de esconderlo. Despues el "
        "padre tambien se fue, y ella se quedo, activa cada tarde, recordandole que habia "
        "algo debajo de la alfombra. Por mucho tiempo fue sufrimiento puro: la catastrofizacion, "
        "la neblina mental, la sensacion de que algo malo estaba por ocurrir. La tecnologia "
        "y la inteligencia artificial eran catalizadores, pantallas que no se apagaban, "
        "demasiada informacion, el suelo moviendose bajo los pies."
    ),
    "sombra_fortaleza": (
        "La mascara que se construyo tan temprano que ya no sabe si es mascara o rostro. "
        "Ser fuerte para no sentir, demostrar que vale para no ser abandonado. El gimnasio "
        "como templo, el cuerpo como prueba viviente de que no eres debil. El nino que "
        "aprendio que si llora es debilidad, que si muestra vulnerabilidad pierde proteccion. "
        "Pero la fortaleza real no es la armadura, es el coraje de bajarla. Las piernas "
        "que antes eran la debilidad ahora son la respuesta. La fortaleza como mecanismo "
        "de supervivencia que cumplio su funcion y ahora busca evolucionar: de escudo a "
        "presencia, de demostrar a ser."
    ),
}

r_orig = evaluate(original, "ORIGINAL (benchmark v2)", all_phrases)
r_best = evaluate(optimal, "OPTIMAL (principles+examples)", all_phrases)

print(f"\n  Original: {r_orig['correct']:2d}/{r_orig['total']} = {r_orig['acc']:.1%}")
print(f"  Optimal:  {r_best['correct']:2d}/{r_best['total']} = {r_best['acc']:.1%}")
print(f"  Mejora:   {(r_best['acc'] - r_orig['acc']):+.1%}")

# Frases que cambiaron de estado
print(f"\n  FRASES QUE MEJORARON:")
for d_o, d_b in zip(r_orig["details"], r_best["details"]):
    if not d_o["ok"] and d_b["ok"]:
        print(f"    + \"{d_b['phrase'][:45]}\" -> {d_b['expected'].split('_')[1][:8]} "
              f"({d_o['exp_sim']:.3f}->{d_b['exp_sim']:.3f})")

print(f"\n  FRASES QUE EMPEORARON:")
for d_o, d_b in zip(r_orig["details"], r_best["details"]):
    if d_o["ok"] and not d_b["ok"]:
        print(f"    - \"{d_b['phrase'][:45]}\" -> esp:{d_b['expected'].split('_')[1][:8]} "
              f"got:{d_b['best'].split('_')[1][:8]} ({d_o['exp_sim']:.3f}->{d_b['exp_sim']:.3f})")
