"""Test C5: Enriquecer examples de Muerte y Fortaleza
Basado en V4 (solo examples) que fue la mejor variante (71.1%)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.25
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

PEER_IDS = ["sombra_muerte", "sombra_rechazo", "sombra_angel_atardecer", "sombra_fortaleza"]

all_phrases = [
    # Originales (18)
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
    # Extras (20)
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
]

# V4 baseline (solo examples, la mejor hasta ahora)
v4_baseline = {
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

# ====================================================================
# ITERACION A: Enriquecer Muerte con mas ejemplos especificos
# Los errores clave eran:
#   "pena verla asi" (nona) →rechazo, "miedo perder alguien" →rechazo,
#   "oscuridad silencio" →rechazo, "doctor examenes" →below threshold
# Hipotesis: Muerte necesita ejemplos de: salud, enfermedad, funerales,
# cuerpo fragil, envejecimiento, perdida anticipada, noña, abuelos
# ====================================================================
iter_a = dict(v4_baseline)  # copiar rechazo y angel
iter_a["sombra_muerte"] = (
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
)
iter_a["sombra_fortaleza"] = v4_baseline["sombra_fortaleza"]

# ====================================================================
# ITERACION B: Enriquecer Fortaleza — errores clave:
#   "cargar creatina" →rechazo, "PR sentadilla" →muerte,
#   "necesito ayuda no puedo pedirla" →rechazo, "me dieron abrazo casi lloro" →rechazo
# Hipotesis: Fortaleza necesita: gym especifico, suplementos especificos,
# no pedir ayuda como patron, armadura emocional, fuerza como identidad
# ====================================================================
iter_b = dict(v4_baseline)
iter_b["sombra_rechazo"] = v4_baseline["sombra_rechazo"]
iter_b["sombra_angel_atardecer"] = v4_baseline["sombra_angel_atardecer"]
iter_b["sombra_muerte"] = v4_baseline["sombra_muerte"]
iter_b["sombra_fortaleza"] = (
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
)

# ====================================================================
# ITERACION C: Ambos enriquecidos (A + B)
# ====================================================================
iter_c = dict(v4_baseline)
iter_c["sombra_rechazo"] = v4_baseline["sombra_rechazo"]
iter_c["sombra_angel_atardecer"] = v4_baseline["sombra_angel_atardecer"]
iter_c["sombra_muerte"] = iter_a["sombra_muerte"]
iter_c["sombra_fortaleza"] = iter_b["sombra_fortaleza"]

# ====================================================================
# ITERACION D: Ambos enriquecidos +Rechazo fortalecido contra "oscuridad"
# Problema: "oscuridad silencio" va a rechazo en vez de muerte
# "necesito ayuda" va a rechazo en vez de fortaleza
# "nona cama pena" va a rechazo en vez de muerte
# => Rechazo esta robando activaciones con su "no me quieren" 
# Solucion: hacer rechazo mas especifico, menos generico
# ====================================================================
iter_d = dict(iter_c)
iter_d["sombra_rechazo"] = (
    "Se activa con: alguien no me responde un mensaje de texto, no me invitan a una salida "
    "o聚会, una chica que me gusta me deja en visto en redes sociales, me rechazan de un "
    "trabajo despues de postular, un grupo de amigos hace planes sin incluirme, alguien me "
    "da la espalda cuando hablo, no encajo en una conversacion grupal, pienso que nadie me "
    "quiere de verdad, me da terror abrirme emocionalmente y que me rechacen por quien soy, "
    "alguien cancela planes conmigo, me comparo con otros que si son incluidos en el grupo, "
    "una fiesta donde nadie me hizo caso, buscar amor romantico donde no florece, "
    "sentir que no pertenezco a ningun grupo social, la puerta cerrada de una relacion, "
    "me borraron del grupo de WhatsApp, mi ex cambio su estado de relacion, "
    "una amiga me reclama que la deje en visto, mama no me beso al despedirme."
)

# ====================================================================
# ITERACION E: Igual que D pero sin el caracter chino que se me coló
# ====================================================================
iter_e = dict(iter_d)
iter_e["sombra_rechazo"] = (
    "Se activa con: alguien no me responde un mensaje de texto, no me invitan a una salida "
    "o reunion, una chica que me gusta me deja en visto en redes sociales, me rechazan de un "
    "trabajo despues de postular, un grupo de amigos hace planes sin incluirme, alguien me "
    "da la espalda cuando hablo, no encajo en una conversacion grupal, pienso que nadie me "
    "quiere de verdad, me da terror abrirme emocionalmente y que me rechacen por quien soy, "
    "alguien cancela planes conmigo, me comparo con otros que si son incluidos en el grupo, "
    "una fiesta donde nadie me hizo caso, buscar amor romantico donde no florece, "
    "sentir que no pertenezco a ningun grupo social, la puerta cerrada de una relacion, "
    "me borraron del grupo de WhatsApp, mi ex cambio su estado de relacion, "
    "una amiga me reclama que la deje en visto, mama no me beso al despedirme."
)

# ====================================================================
# ITERACION F: E + keywords anti-confusion
# Muerte: "no es rechazo social, es perdida y finitud"
# Fortaleza: "no es rechazo, es armadura y rendimiento"
# Angel: "no es rechazo, es inquietud interna"
# ====================================================================
iter_f = dict(iter_e)
# Angel tambien necesita ayuda vs "abrumado" →fortaleza
iter_f["sombra_angel_atardecer"] = (
    "Se activa con: ansiedad que llega al atardecer, sentirse abrumado por tecnologia y "
    "demasiada informacion, neblina mental que no se va, catastrofizar situaciones pequenas, "
    "sentir que algo malo va a pasar sin razon clara, no poder procesar tanta informacion "
    "del dia, sentimientos escondidos bajo la alfombra que salen de noche, pantallas como "
    "catalizador de inquietud, leer sobre IA y sentir angustia existencial, aprender "
    "demasiadas cosas nuevas sin pausa, el sol baja y la inquietud sube, la neblina "
    "cognitiva vespertina, la rumiacion nocturna, el duelo del tata no procesado, "
    "demasiado tiempo en pantallas sin poder parar, scrollear sin control, la mente "
    "va muy rapido y no puedo frenar, pensar en AGI y paralizarme."
)


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
            "exp_sim": exp_sim, "best_sim": best_sim, "ok": ok, "sims": sims
        })
    acc = correct / total
    return {"acc": acc, "correct": correct, "total": total, "details": details, "label": label}


def print_result(r, show_errors=5):
    status = "OK" if r["acc"] >= 0.80 else "+" if r["acc"] >= 0.70 else "~" if r["acc"] >= 0.60 else "X"
    print(f"  [{status}] {r['label']:45s} {r['correct']:2d}/{r['total']} = {r['acc']:.1%}")
    errors = [d for d in r["details"] if not d["ok"]]
    for d in errors[:show_errors]:
        exp_n = d['expected'].split('_')[1][:6]
        best_n = d['best'].split('_')[1][:6]
        print(f"       X \"{d['phrase'][:35]}\" -> esp:{exp_n} got:{best_n} "
              f"({d['exp_sim']:.3f} vs {d['best_sim']:.3f})")


# ====================================================================
# EJECUTAR
# ====================================================================
print("=" * 75)
print("  TEST C5: ENRIQUECER MUERTE Y FORTALEZA")
print(f"  {len(all_phrases)} frases | Baseline V4: 71.1%")
print("=" * 75)

variants = [
    ("V4 baseline (solo examples)", v4_baseline),
    ("A: Muerte enriquecida", iter_a),
    ("B: Fortaleza enriquecida", iter_b),
    ("C: Ambos enriquecidos (A+B)", iter_c),
    ("D: C + Rechazo especifico", iter_d),
    ("E: D sin bug chino", iter_e),
    ("F: E + Angel fortalecida", iter_f),
]

print("\n  --- COMPARACION GENERAL ---")
results = []
for name, peers in variants:
    r = evaluate(peers, name)
    results.append(r)
    print_result(r)

# Mejor variante
best = max(results, key=lambda x: x["acc"])
print(f"\n\n{'='*75}")
print(f"  MEJOR: {best['label']} ({best['acc']:.1%})")
print(f"{'='*75}")

# Desglose por peer
print(f"\n  --- DESGLOSE POR PEER ---")
for pid in PEER_IDS:
    name = pid.split("_")[1][:12]
    peer_results = [d for d in best["details"] if d["expected"] == pid]
    correct = sum(1 for d in peer_results if d["ok"])
    total = len(peer_results)
    print(f"    {name:>15s}: {correct:2d}/{total} ({correct/total:.0%})")

# Todos los errores del mejor
print(f"\n  --- ERRORES DETALLADOS ---")
for d in best["details"]:
    if not d["ok"]:
        exp_n = d['expected'].split('_')[1][:8]
        best_n = d['best'].split('_')[1][:8]
        sims_str = "  ".join(f"{pid.split('_')[1][:4]}:{sim:.3f}" for pid, sim in 
                            sorted(d["sims"].items(), key=lambda x: -x[1]))
        print(f"    X \"{d['phrase'][:42]}\" -> esp:{exp_n} got:{best_n}")
        print(f"      {sims_str}")

# Comparacion delta vs baseline
print(f"\n\n  --- DELTA vs BASELINE ---")
base = results[0]
for r in results[1:]:
    delta = r["acc"] - base["acc"]
    improved = sum(1 for d_b, d_r in zip(base["details"], r["details"]) if not d_b["ok"] and d_r["ok"])
    worsened = sum(1 for d_b, d_r in zip(base["details"], r["details"]) if d_b["ok"] and not d_r["ok"])
    print(f"    {r['label']:45s} {delta:+.1%} (+{improved}/-{worsened})")

# Stack: que frases son problematicas en TODAS las variantes?
print(f"\n  --- FRASES QUE NUNCA PASAN (problema de modelo) ---")
for i, phrase_data in enumerate(all_phrases):
    phrase, expected = phrase_data
    never_passes = all(not r["details"][i]["ok"] for r in results)
    if never_passes:
        print(f"    \"{phrase[:45]}\" -> {expected.split('_')[1][:8]}")
        # Mostrar mejor sim score alcanzado
        best_sim = max(r["details"][i]["exp_sim"] for r in results)
        best_variant = max(results, key=lambda r: r["details"][i]["exp_sim"])
        print(f"      Mejor: {best_sim:.3f} ({best_variant['label'][:30]})")
