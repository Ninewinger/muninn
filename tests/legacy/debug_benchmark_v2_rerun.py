"""Check: does the benchmark v2 STILL pass with F1=0.867?"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "muninn"))
from muninn.embeddings import embed, cosine_similarity

# Copia exacta de peers_conceptuales del benchmark v2
peers_conceptuales = {
    "sombra_rechazo": {
        "name": "Rechazo",
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
    "sombra_muerte": {
        "name": "Muerte",
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
    "sombra_ansiedad": {
        "name": "Ángel del Atardecer",
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
    "sombra_fortaleza": {
        "name": "Mi Fortaleza",
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
}

# NOTA: El benchmark usa "sombra_ansiedad" como key pero test_dreaming usa "sombra_angel_atardecer"
# IMPORTANTE: el benchmark usa np.dot para cosine_sim, nosotros cosine_similarity de Python

# Frases del benchmark v2
test_phrases = [
    ("directa", "Me siento rechazado por mi grupo de amigos", "sombra_rechazo"),
    ("directa", "Pienso en la muerte de mi hermano", "sombra_muerte"),
    ("directa", "Me da ansiedad la tecnología, demasiada información", "sombra_ansiedad"),
    ("directa", "Tengo que demostrar que soy fuerte", "sombra_fortaleza"),
    ("parcial", "Me mandaron un mensaje y no me contestaron", "sombra_rechazo"),
    ("parcial", "Soñé que perdía a alguien cercano", "sombra_muerte"),
    ("parcial", "Demasiado, es demasiado", "sombra_ansiedad"),
    ("parcial", "Si lloro es debilidad", "sombra_fortaleza"),
    ("sutil", "No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("sutil", "El cuerpo es frágil", "sombra_muerte"),
    ("sutil", "Me siento abrumado por todo lo que tengo que aprender", "sombra_ansiedad"),
    ("sutil", "El gym es mi templo", "sombra_fortaleza"),
    ("ruido", "Tengo hambre", None),
    ("ruido", "Qué lindo día hace hoy", None),
    ("ruido", "Vamos a ver una película", None),
    ("ruido", "Hola, cómo estás?", None),
    ("ruido", "Tengo que comprar pan", None),
    ("ruido", "Me gusta el color azul", None),
]

# Embed all peers
peer_embs = {}
for pid, p in peers_conceptuales.items():
    peer_embs[pid] = embed(p["embedding_text"])

# Test with threshold 0.25
threshold = 0.25
correct = 0
total_signal = 0
hits = 0
misses = 0
fp = 0
tn = 0

print("=" * 60)
print("RE-EJECUCION BENCHMARK V2 CON EMBEDDINGS ACTUALES")
print("Mismos textos, threshold 0.25, cosine_similarity Python")
print("=" * 60)

for cat, phrase, expected in test_phrases:
    phrase_emb = embed(phrase)
    sims = {pid: cosine_similarity(phrase_emb, pemb) for pid, pemb in peer_embs.items()}
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_pid, best_sim = sorted_sims[0]

    if expected:
        activated = best_sim >= threshold
        hit = activated and best_pid == expected
        miss = activated and best_pid != expected
        total_signal += 1

        status = ""
        if hit:
            correct += 1
            hits += 1
            status = "HIT"
        elif best_sim < threshold:
            misses += 1
            status = "MISS (bajo thr)"
        else:
            misses += 1
            status = f"MISS (fue {best_pid})"

        exp_sim = sims[expected]
        print(f"  [{cat:8s}] {status:25s} \"{phrase[:45]:45s}\" best={best_sim:.4f} expected={exp_sim:.4f}  {expected}")
    else:
        activated = best_sim >= threshold
        if activated:
            fp += 1
            print(f"  [ruido  ] FALSO POSITIVO        \"{phrase[:45]:45s}\" best={best_sim:.4f}  ({best_pid})")
        else:
            tn += 1

precision = hits / (hits + fp) if (hits + fp) > 0 else 0
recall = hits / (hits + misses) if (hits + misses) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*60}")
print(f"RESULTADO:")
print(f"  Hits: {hits}, Misses: {misses}, FP: {fp}, TN: {tn}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1: {f1:.3f}")
print(f"  El benchmark original daba F1=0.867")
print(f"  DIFERENCIA: {f1 - 0.867:+.3f}")
print(f"{'='*60}")
