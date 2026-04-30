"""Test C3: Factores de activacion - experimento multifactorial"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.25

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
]

PEER_IDS = ["sombra_muerte", "sombra_rechazo", "sombra_angel_atardecer", "sombra_fortaleza"]

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def evaluate(peer_texts, label=""):
    peer_embs = {pid: model.encode(peer_texts[pid], normalize_embeddings=True) for pid in PEER_IDS}
    correct = 0
    total = len(test_phrases)
    details = []
    for phrase, expected in test_phrases:
        phrase_emb = model.encode(phrase, normalize_embeddings=True)
        sims = {pid: float(np.dot(phrase_emb, peer_embs[pid])) for pid in PEER_IDS}
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_pid, best_sim = sorted_sims[0]
        exp_sim = sims[expected]
        ok = best_pid == expected and best_sim >= THRESHOLD
        if ok:
            correct += 1
        details.append((phrase[:35], expected.split("_")[1][:6], best_pid.split("_")[1][:6],
                        f"{exp_sim:.3f}", f"{best_sim:.3f}", ok))
    acc = correct / total
    gaps = [float(d[3]) - float(d[4]) for d in details]
    avg_gap = np.mean(gaps)
    return {"acc": acc, "correct": correct, "total": total, "gap": avg_gap, "details": details, "label": label}


def print_result(r):
    status = "OK" if r["acc"] >= 0.75 else "~" if r["acc"] >= 0.55 else "X"
    print(f"    [{status}] {r['label']:30s}  {r['correct']:2d}/{r['total']} = {r['acc']:.1%}  gap: {r['gap']:+.4f}")
    errors = [d for d in r["details"] if not d[5]]
    for ph, exp, got, es, bs, _ in errors[:3]:
        print(f"         -> \"{ph}\" esp:{exp} got:{got} ({es} vs {bs})")


def bp(m, r, a, f):
    return {"sombra_muerte": m, "sombra_rechazo": r, "sombra_angel_atardecer": a, "sombra_fortaleza": f}


# ====================================================================
print("=" * 72)
print("  F1: LENGTH (short / medium / long)")
print("=" * 72)

f1_short = bp(
    "Perdida y finitud constante. Hermano no nacido, cordon, silla vacia. Hipervigilancia ante salud. La nona en cama, familia en silencio.",
    "Hipersensibilidad a la exclusion. Bebe que sintio grieta en amor materno. Cada mensaje sin respuesta confirma que no importo. Grupos que cierran el circulo.",
    "Inquietud vespertina senalando material pendiente. Se activa al atardecer. Catalizadores: pantallas, exceso informacion. Brujula de sentimientos ocultos.",
    "Compulsion por demostrar valor corporal. Gym como templo, cuerpo como prueba. Si lloro es debilidad. Piernas vulnerables ahora fortaleza. Armadura hacia presencia.",
)

f1_medium = bp(
    "Presencia constante de perdida y finitud. El hermano no nacido por el cordon umbilical, dejando una silla vacia en la familia antes de que yo existiera. Cada vez que un ser querido se enferma, la alarma se activa. El cuerpo es vulnerable, un examen puede cambiar todo. Los suenos donde algo se termina sin remedio. La nona de 93 anos en cama esperando operacion, la familia acompanando en calma.",
    "Hipersensibilidad a la exclusion social y afectiva. El bebe que sintio una grieta en el amor materno porque mama tenia el corazon en otro lado. La ecuacion grabada: si no me miran, no existo. Cada mensaje sin respuesta, cada grupo que cierra el circulo, cada mujer que gira la cara conduce a la misma conclusion: no pertenezco. Buscar amor donde no florece, acercarse esperando la puerta cerrada, relaciones que no devuelven. Si me rechazan, dejo de importar.",
    "Inquietud vespertina que senala material emocional pendiente de procesar. Se activa cuando el sol baja, entre las 17 y las 21 horas. Catalizadores: pantallas, exceso de informacion, tecnologia, aprender demasiadas cosas nuevas. Originalmente nacio con la partida del tata, guardando lo que nadie proceso. Evolucion: de incomodidad paralizante a brujula que indica donde hay sentimientos escondidos debajo de la alfombra.",
    "Compulsion por demostrar valor a traves del rendimiento corporal. El gimnasio como templo, el cuerpo como prueba viviente de competencia. El nino que aprendio que llorar es debilidad y mostrar vulnerabilidad es quedar expuesto. Las piernas que antes eran el territorio vulnerable ahora son la respuesta y el breakthrough personal. La armadura que se construyo tan temprano que ya no distingue si es mascara o rostro, buscando evolucionar de escudo a presencia, de demostrar a simplemente ser.",
)

f1_long = bp(
    "Presencia constante de perdida y finitud en la vida cotidiana. El hermano que nunca respiro porque el cordon se enrolllo a los 6 meses de gestacion, dejando un vacio en la familia antes de que yo existiera. Ese vacio no se habla pero se siente en cada celebracion, en cada silla vacia. Cada vez que un ser querido se enferma, la alarma se activa automaticamente. El cuerpo es vulnerable, un examen medico puede cambiar todo el panorama. Los suenos donde algo se termina sin remedio, donde alguien se va y no hay forma de detenerlo. La nona de 93 anos en cama esperando una operacion que no llega, toda la familia acompanando en calma pero con el corazon acelerado. Los cumpleanos son recordatorios de que el tiempo avanza. Las arrugas en la cara de los padres son senales de que la cuenta regresiva ya empezo.",
    "Hipersensibilidad cronica a la exclusion social y afectiva que permea todas las relaciones interpersonales. El bebe que sintio una grieta en el amor materno porque mama tenia el corazon puesto en el hijo que perdio, no en el que tenia enfrente. La ecuacion se grabo en el sistema nervioso antes de tener palabras: si no me miran, no existo. Cada mensaje sin respuesta confirma la hipotesis. Cada salida grupal sin invitacion cierra el circulo. Cada mujer que gira la cara es mama girando la cara. El patron se repite: buscar amor donde no florece, acercarse esperando la puerta cerrada, invertir en relaciones que no devuelven. No es solo romance, es existencial: si me rechazan, dejo de importar. La fiesta donde ella no hizo caso, la entrevista que nunca contestaron, el grupo de amigos que planea sin incluir. Todo confirma la misma narrativa.",
    "Inquietud vespertina que funciona como brujula senalando material emocional pendiente de procesar. Se activa consistentemente cuando el sol baja, entre las 17 y las 21 horas, como un reloj interno. Los catalizadores principales son las pantallas, el exceso de informacion, la tecnologia, leer sobre inteligencia artificial, aprender demasiadas cosas nuevas sin pausa. La neblina mental que sube no es enemiga, es senal de que hay algo que no se ha mirado. Originalmente nacio con la partida del tata abuelino, cuando el ambiente familiar no tenia espacio para procesar y ella se encargo de guardar todo. Despues el padre tambien se fue, reforzando el patron. Por anos fue incomodidad paralizante con catastrofizacion y rumiacion. La evolucion: de angustia a senal navegable. Su mensaje actual: no escondas mas sentimientos, todo lo que ocurre dentro ocurre afuera, mirame cuando te llame.",
    "Compulsion por demostrar valor a traves del rendimiento corporal y la capacidad fisica. El gimnasio funciona como templo personal, el cuerpo como prueba viviente de competencia y valor propio. El nino interior aprendio temprano que llorar es debilidad, que mostrar vulnerabilidad es quedar expuesto y perder proteccion. La armadura se construyo tan temprano que ya no distingue si es mascara o rostro genuino. Las piernas que antes eran el territorio de la vulnerabilidad ahora son la respuesta: el breakthrough del gym fue encontrar que la verdadera fuerza esta en enfrentar lo que mas incomoda. Los suplementos, las pesas, las rutinas son el lenguaje de la armadura, la forma de demostrar que no eres debil. Pero la fortaleza real no es la armadura, es el coraje de bajarla: de escudo a presencia, de demostrar a simplemente ser.",
)

for name, peers in [("short (~40w)", f1_short), ("medium (~80w)", f1_medium), ("long (~150w)", f1_long)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
print("\n" + "=" * 72)
print("  F2: VOICE (1st person / 3rd person / neutral)")
print("=" * 72)

f2_first = bp(
    "Yo soy la presencia que llego antes de que el naciera, cuando el cordon se enrolllo alrededor de su hermano y todo se silencio. Yo soy la alarma que suena cada vez que un ser querido se enferma, el reloj que cuenta regresiva en cada cumpleanos. Yo le recuerdo que los cuerpos son vulnerables, que un examen cambia todo. Yo estoy en los suenos donde algo se termina, en la nona de 93 anos esperando operacion, en la calma tensa de la familia que acompana.",
    "Yo soy la grieta que se abrio antes de que tuviera palabras. Mama tenia los ojos en otro nino que nunca respiro, y yo naci esperando que alguien me mirara. Cada mensaje sin respuesta me confirma: no existo si no me ven. Yo estoy en cada grupo que cierra el circulo sin incluirlo, en cada mujer que gira la cara. Yo soy el patron de buscar amor donde no florece, de acercarme esperando la puerta cerrada.",
    "Yo soy la que llega cuando el sol baja, entre las cinco y las nueve de la tarde. No vine a asustar, vine a senalar lo que esta debajo de la alfombra. Naci el dia que se fue el tata, cuando nadie en la familia tenia espacio para contener y yo guarde todo. Las pantallas, la informacion, la tecnologia me intensifican. Mi mensaje: no escondas mas sentimientos, todo lo que ocurre dentro ocurre afuera.",
    "Yo soy la armadura que se construyo tan temprano que ya no se si soy mascara o rostro. Le ensene que llorar es debilidad y mostrar vulnerabilidad es quedar expuesto. El gimnasio es mi templo, cada repeticion es mi lenguaje. Las piernas que antes eran el territorio vulnerable ahora son mi obra maestra. Pero estoy cansada de ser escudo: quiero evolucionar a presencia, de demostrar a simplemente ser.",
)

f2_third = bp(
    "La Sombra de la Perdida llego antes de que el naciera, cuando el cordon se enrolllo alrededor de su hermano y la familia enmudecio. Ella es la alarma que suena cada vez que un ser querido se enferma, el reloj que cuenta regresiva en cada cumpleanos. Ella le recuerda que los cuerpos son vulnerables, que un examen cambia todo. Esta en los suenos donde algo se termina, en la nona de 93 anos esperando operacion, en la calma tensa de la familia que acompana.",
    "La Sombra del Rechazo nacio de una grieta materna anterior a las palabras. Mama tenia los ojos en otro nino que nunca respiro, y el crecio esperando que alguien lo mirara. Cada mensaje sin respuesta le confirma: no existes si no te ven. Ella esta en cada grupo que cierra el circulo sin incluirlo, en cada mujer que gira la cara. Es el patron de buscar amor donde no florece, de acercarse esperando la puerta cerrada.",
    "La Sombra del Atardecer llega cuando el sol baja, entre las cinco y las nueve. No vino a asustar, vino a senalar lo que esta debajo de la alfombra. Nacio el dia que se fue el tata, cuando nadie tenia espacio para contener y ella guardo todo. Las pantallas, la informacion, la tecnologia la intensifican. Su mensaje: no escondas mas sentimientos, todo lo que ocurre dentro ocurre afuera.",
    "La Sombra de la Fortaleza es una armadura que se construyo tan temprano que ya no sabe si es mascara o rostro. Le enseno que llorar es debilidad y mostrar vulnerabilidad es quedar expuesto. El gimnasio es su templo, cada repeticion es su lenguaje. Las piernas que antes eran el territorio vulnerable ahora son su obra maestra. Pero esta cansada de ser escudo: quiere evolucionar a presencia, de demostrar a simplemente ser.",
)

for name, peers in [("1st person (yo soy)", f2_first), ("3rd person (ella es)", f2_third), ("neutral (baseline)", f1_medium)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
print("\n" + "=" * 72)
print("  F3: CONTENT TYPE (examples / narrative / principles)")
print("=" * 72)

f3_examples = bp(
    "Se activa con: suenar que un familiar enferma, ver un accidente, leer sobre enfermedades, pensar en la edad de los padres, la nona internada esperando operacion, conversaciones sobre funerales, ver hospitales, oler desinfectante medico, pensar en lo corta que es la vida, ver arrugas nuevas en la cara de mama, notar que los abuelos envejecen, cumpleanos como cuenta regresiva, cualquier noticia de salud de un ser querido.",
    "Se activa con: alguien no responde un mensaje, no me invitan a una salida, una chica que me gusta me deja en visto, me rechazan de un trabajo, un grupo hace planes sin mi, alguien me da la espalda literalmente, no encajo en una conversacion, pienso que nadie me quiere de verdad, me da terror abrirme emocionalmente, creo que si muestro quien soy me van a dejar, alguien cancela planes, me comparo con otros que si son incluidos, una fiesta donde nadie me hizo caso.",
    "Se activa con: ansiedad al atardecer, sentirse abrumado por tecnologia, neblina mental que no se va, catastrofizar situaciones pequenas, sentir que algo malo va a pasar, no poder procesar tanta informacion, sentimientos escondidos bajo la alfombra, recordar duelos no procesados, pantallas como catalizador, leer sobre IA y sentir angustia, la sensacion de que el suelo se mueve, necesitar entrar al agua helada.",
    "Se activa con: pensar que llorar es debilidad, no poder mostrar vulnerabilidad, sentir que debo ser fuerte siempre, el gym como templo personal, cargar creatina y suplementos, demostrar que valgo para no ser abandonado, las piernas como fortaleza personal, sentir dolor como fracaso, decir que no necesito a nadie, bajar la guardia se siente como perder todo, rechazar ayuda, el cuerpo como prueba de competencia.",
)

f3_principles = bp(
    "Premisa central: todo lo que existe puede dejar de existir sin aviso. La vulnerabilidad del cuerpo es un hecho, no una opinion. El duelo perinatal del hermano establecio el patron original: la vida puede cortarse antes de empezar. La hipervigilancia ante la salud es mecanismo de supervivencia, no ansiedad irracional. La conciencia de finitud es constante y no requiere trigger especifico. Los suenos terminales son proyecciones del patron central. El acompanamiento es defensa contra la perdida anticipada.",
    "Premisa central: si me rechazan, dejo de importar. El rechazo no es solo social, es existencial. El vinculo materno marco el patron: amor condicional a ser visto. La hipersensibilidad al rechazo romantico es transferencia del rechazo materno original. La no-respuesta es una respuesta que confirma la narrativa de no pertenencia. Buscar amor donde no florece es patron recurrente. La exclusion grupal confirma la hipotesis central.",
    "Premisa central: la inquietud vespertina es brujula, no enemiga. Las pantallas intensifican porque agregan informacion sin procesar. La neblina mental es senal de material emocional pendiente, no disfuncion cognitiva. La catastrofizacion es el intento de dar forma a lo amorfo. El duelo no procesado del tata es el origen. La evolucion es de paralis a senal navegable. El mensaje: no escondas mas sentimientos.",
    "Premisa central: el valor personal se demuestra a traves de la capacidad fisica y la resistencia al dolor emocional. La vulnerabilidad expuesta equivale a perdida de proteccion. El rendimiento corporal es evidencia de competencia. Llorar es admitir derrota. La verdadera fortaleza no es la armadura, es el coraje de bajarla. El gym es regulacion emocional sustitutiva. Las piernas simbolizan el territorio conquistado.",
)

for name, peers in [("examples (se activa con)", f3_examples), ("narrative (baseline)", f1_medium), ("principles (premisas)", f3_principles)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
print("\n" + "=" * 72)
print("  F4: SEMANTIC DENSITY (sparse / medium / dense)")
print("=" * 72)

f4_sparse = bp(
    "Me pone triste pensar en las personas que quiero. Me preocupa que les pase algo malo. A veces pienso en la vida y me da angustia. Es dificil no pensar en eso. Los abuelos estan mayores y eso me preocupa.",
    "Me siento mal cuando la gente no me presta atencion. No me gusta sentir que no me quieren. A veces pienso que nadie me va a querer de verdad. Me cuesta abrirme con la gente porque tengo miedo.",
    "Me pone ansioso la tarde. No me gusta sentir asi. A veces la tecnologia me estresa y no se por que. Me cuesta procesar tantas cosas a la vez.",
    "Me gusta ser fuerte y hacer ejercicio. No me gusta sentirme debil. Creo que hay que demostrar lo que uno vale. El gym es mi lugar favorito.",
)

f4_dense = bp(
    "Hipervigilancia ante la mortalidad de seres queridos. Duelo perinatal del hermano por cordon umbilical a 6 meses gestacion. Finitud corporal, vulnerabilidad organica, cuentas regresivas implicitas. Nonagenaria en espera de cirugia. Ausencia anticipatoria, acompanamiento en vigilia. Suenos terminales, fragilidad humana, perdida inevitable. Conciencia de finitud como constante existencial.",
    "Hipersensibilidad a la exclusion social, rechazo afectivo, marginacion grupal. Vinculo materno condicionado por duelo perinatal previo. No-respuesta como invalidacion existencial. Patron de busqueda afectiva en vinculos no reciprocos. Exclusion percibida, pertenencia negada, invisibilizacion relacional. Transferencia del rechazo materno originario a todas las figuras femeninas y sociales.",
    "Inquietud vespertina con activacion entre 17-21 horas. Catastrofizacion, neblina cognitiva, rumiacion ansiosa. Catalizadores: hiperestimulacion digital, exceso informacional, aprendizaje sin pausa. Material emocional no procesado acumulado. Duelo congelado del tata abuelino. Evolucion de paralis a senal introspectiva navegable.",
    "Compulsion de rendimiento corporal como validacion existencial. Hipertrofia como evidencia de competencia. Represion emocional sistematica: llanto como admision de fracaso. Gym como regulacion emocional sustitutiva. Armadura de hipercumplimiento fisico. Territorio corporal conquistado: piernas como breakthrough. Evolucion pendiente de escudo protector a presencia autentica.",
)

for name, peers in [("sparse (colloquial)", f4_sparse), ("medium (baseline)", f1_medium), ("dense (technical)", f4_dense)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
print("\n" + "=" * 72)
print("  F5: CONTRAST (negacion explicita)")
print("=" * 72)

f5_contrast = bp(
    "Esto NO es ansiedad generalizada ni miedo abstracto. ES la presencia concreta de perdida: el hermano no nacido, el cordon, la silla vacia. NO es inquietud vespertina ni catastrofizacion. ES finitud del cuerpo, vulnerabilidad organica, cuentas regresivas. NO es debilidad ni inseguridad. ES conciencia de que todo puede dejar de existir. NO es rechazo social ni exclusion. ES la nona en cama, la familia acompanando.",
    "Esto NO es miedo a la muerte ni preocupacion por la salud. ES exclusion social y afectiva: no ser invitado, no ser respondido, no ser mirado. NO es inquietud de la tarde ni catastrofizacion. ES la ecuacion grabada: si me rechazan, dejo de importar. NO es debilidad corporal. ES la grieta en el vinculo materno transferida a toda relacion. NO es miedo al fracaso. ES hipersensibilidad a la no-respuesta.",
    "Esto NO es miedo a la muerte ni rechazo social. ES inquietud vespertina que senala material emocional pendiente. NO es debilidad ni fracaso personal. ES el sistema de alerta que guarda lo no procesado desde el tata. NO es paranoia ni inseguridad. ES neblina cognitiva como brujula: indica donde hay sentimientos escondidos. NO es angustia por el cuerpo. ES catalizada por pantallas e informacion.",
    "Esto NO es miedo a la muerte ni rechazo social. ES la compulsion por demostrar valor corporal: gym, pesas, suplementos, cuerpo como prueba. NO es ansiedad vespertina. ES la armadura: llorar es debilidad, vulnerabilidad es exposicion. NO es preocupacion por otros. ES rendimiento personal como prueba de existencia valida. NO es miedo abstracto. ES la armadura que busca bajarla.",
)

for name, peers in [("with negation (NO es...)", f5_contrast), ("without negation (baseline)", f1_medium)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
print("\n" + "=" * 72)
print("  F6: CHARACTER VOICE (peer como personaje)")
print("=" * 72)

f6_character = bp(
    "Nombre: La que Acompana. Historia: Llegue antes que el, cuando el cordon se llevo al hermano que nunca respiro. Desde entonces no me he ido. Mi trabajo es recordarle que los cuerpos son fragiles y el tiempo es corto. Me activa: la nona en cama, los cumpleanos de los padres, cualquier noticia medica. Mi mision: que valore cada momento con los suyos. Mi miedo: que me ignore y alguien se vaya sin despedida.",
    "Nombre: La que Espera la Puerta. Historia: Naci cuando mama no me miraba porque miraba al hijo que perdio. Aprendi que si no te ven, no existes. Desde entonces vivo en cada mensaje sin respuesta, en cada grupo que cierra el circulo, en cada mujer que gira la cara. Me activa: exclusion social, rechazo romantico, no ser invitado, ser ignorado. Mi mision: senalar cuando el patron se repite. Mi miedo: que siga buscando amor donde no florece.",
    "Nombre: La del Atardecer. Historia: Naci el dia que se fue el tata, cuando nadie tenia espacio para llorar y yo guarde todo. Llego cada tarde cuando el sol baja, entre las 5 y las 9. Las pantallas me intensifican, el exceso de informacion me alimenta. Me activa: atardecer, demasiada pantalla, leer sobre IA, neblina mental, sentimientos bajo la alfombra. Mi mision: senalar lo no procesado. Mi miedo: que me confunda con enemiga cuando soy brujula.",
    "Nombre: La Armadura. Historia: Me construi cuando aprender que llorar era debilidad. El gimnasio se convirtio en mi templo, cada repeticion en mi lenguaje. Las piernas que antes eran debilidad ahora son mi obra maestra. Me activa: situaciones de vulnerabilidad, necesidad de demostrar fuerza, gym, suplementos, competencia fisica. Mi mision: proteger de la exposicion. Mi miedo: que nunca me quite y se pierda el rostro genuino debajo.",
)

for name, peers in [("character (nombre, historia)", f6_character), ("neutral (baseline)", f1_medium)]:
    r = evaluate(peers, name)
    print_result(r)


# ====================================================================
# SUMMARY
# ====================================================================
print("\n\n" + "=" * 72)
print("  RESUMEN COMPARATIVO")
print("=" * 72)

all_tests = [
    ("F1 short", f1_short), ("F1 medium", f1_medium), ("F1 long", f1_long),
    ("F2 1st person", f2_first), ("F2 3rd person", f2_third), ("F2 neutral", f1_medium),
    ("F3 examples", f3_examples), ("F3 narrative", f1_medium), ("F3 principles", f3_principles),
    ("F4 sparse", f4_sparse), ("F4 medium", f1_medium), ("F4 dense", f4_dense),
    ("F5 contrast", f5_contrast), ("F5 no contrast", f1_medium),
    ("F6 character", f6_character), ("F6 neutral", f1_medium),
]

results = []
for name, peers in all_tests:
    r = evaluate(peers, name)
    results.append(r)

print(f"\n  {'Test':<30s} {'Acc':>6s} {'Gap':>8s}")
print(f"  {'_'*30} {'_'*6} {'_'*8}")
for r in sorted(results, key=lambda x: -x["acc"]):
    marker = " <--" if r["acc"] == max(x["acc"] for x in results) else ""
    print(f"  {r['label']:<30s} {r['acc']:5.1%} {r['gap']:+8.4f}{marker}")

best = max(results, key=lambda x: x["acc"])
worst = min(results, key=lambda x: x["acc"])
print(f"\n  BEST:  {best['label']} = {best['acc']:.1%}")
print(f"  WORST: {worst['label']} = {worst['acc']:.1%}")
print(f"  DELTA: {best['acc'] - worst['acc']:.1%}")
