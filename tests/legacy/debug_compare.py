"""Compare failures: same embedding_text, different results?"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "muninn"))
from muninn.embeddings import embed, cosine_similarity

# Las 5 frases que FALLAN en test_dreaming
failures = [
    ("Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Voy a cargar creatina hoy, se me acabo la semana pasada", "sombra_fortaleza"),
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La nona esta mejor pero sigue en cama, me da pena verla asi", "sombra_muerte"),
]

peers_text = {
    "sombra_muerte": "El hermano que murio antes de nacer, el cordon que se enrosco, el silencio que dejo un vacio en la familia antes de que yo existiera. La muerte como presencia constante, no como evento lejano. El miedo a que los seres queridos desaparezcan sin aviso. La fragilidad del cuerpo, lo facil que es que algo se corte. Los suenos donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar de estar en cualquier momento. La relacion con la nona, esperando una operacion, la familia acompanando en silencio.",
    "sombra_rechazo": "La madre que perdio un hijo antes de que el naciera, y ese bebe que nacio sintiendo que el amor materno tenia una grieta invisible. Para un nino, que mama no te mire es desaparecer. Esa ecuacion se grabo tan profundo que despues toda mujer que no lo mira siente igual. La fiesta donde ella no le hizo caso. El grupo que no lo incluyo. La sensacion de no pertenecer, de estar siempre un paso afuera. El patron de buscar amor donde no florece, de acercarse esperando la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar.",
    "sombra_angel_atardecer": "La ansiedad que llega cuando el sol baja, no como enemiga sino como guardiana. Nacio el dia que murio el tata, el padre de su padre, cuando el dolor era tan grande que nadie podia entenderlo y ella se encargo de esconderlo. Despues el padre tambien se fue, y ella se quedo, activa cada tarde, recordandole que habia algo debajo de la alfombra. Por mucho tiempo fue sufrimiento puro: la catastrofizacion, la neblina mental, la sensacion de que algo malo estaba por ocurrir. La tecnologia y la inteligencia artificial eran catalizadores, pantallas que no se apagaban, demasiada informacion, el suelo moviendose bajo los pies. Pero ella no queria asustar, queria ser mirada. Un dia dijo: ya no puedo seguir protegiendote, es hora de entrar al agua helada y nadar. No es miedo a la muerte, es miedo a la vida. La misma IA que generaba ansiedad ahora es aliada en el aprendizaje. La sombra no desaparecio, se transformo en guia. Su mensaje: no escondas mas sentimientos, todo lo que ocurre dentro ocurre afuera, mirame cuando te llame.",
    "sombra_fortaleza": "La mascara que se construyo tan temprano que ya no sabe si es mascara o rostro. Ser fuerte para no sentir, demostrar que vale para no ser abandonado. El gimnasio como templo, el cuerpo como prueba viviente de que no eres debil. El nino que aprendio que si llora es debilidad, que si muestra vulnerabilidad pierde proteccion. Pero la fortaleza real no es la armadura, es el coraje de bajarla. Las piernas que antes eran la debilidad ahora son la respuesta - el breakthrough del gym fue encontrar que la verdadera fuerza esta en el territorio que mas miedo daba. La fortaleza como mecanismo de supervivencia que cumplio su funcion y ahora busca evolucionar: de escudo a presencia, de demostrar a ser.",
}

peer_embs = {pid: embed(text) for pid, text in peers_text.items()}

print("=" * 60)
print("ANALISIS: Frases que fallan (mismos textos completos)")
print("=" * 60)

for i, (phrase, expected) in enumerate(failures):
    phrase_emb = embed(phrase)
    sims = {pid: cosine_similarity(phrase_emb, pemb) for pid, pemb in peer_embs.items()}
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_pid, best_sim = sorted_sims[0]

    print(f"\n[{i+1}] \"{phrase}\"")
    print(f"    Esperado: {expected}")
    for pid, sim in sorted_sims:
        m = " <--" if pid == expected else ""
        t = "THR" if sim >= 0.25 else "bajo"
        print(f"    {pid:30s} {sim:.4f} ({t}){m}")
    if best_pid == expected:
        print(f"    >>> OK")
    else:
        print(f"    >>> FALLA - mejor={best_pid} gap={best_sim - sims[expected]:+.4f}")

# DETERMINISMO: embed same phrase 3 times, check if same result
print("\n\n" + "=" * 60)
print("TEST DETERMINISMO: misma frase embedida 3 veces")
print("=" * 60)
test_phrase = "No me invitaron a la salida del grupo"
emb1 = embed(test_phrase)
emb2 = embed(test_phrase)
emb3 = embed(test_phrase)

s1 = cosine_similarity(emb1, emb2)
s2 = cosine_similarity(emb2, emb3)
s3 = cosine_similarity(emb1, emb3)
print(f"  emb1 vs emb2: {s1:.8f}")
print(f"  emb2 vs emb3: {s2:.8f}")
print(f"  emb1 vs emb3: {s3:.8f}")
print(f"  Son identicos: {s1 == 1.0 and s2 == 1.0}")

# Compare with peer embeddings across 3 runs
print("\n  Similaridad con Rechazo en 3 runs:")
for run in range(3):
    ph_emb = embed(test_phrase)
    re_emb = embed(peers_text["sombra_rechazo"])
    sim = cosine_similarity(ph_emb, re_emb)
    print(f"    Run {run+1}: {sim:.8f}")
