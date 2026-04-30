"""Test D: Tildes vs sin tildes — cuánto afecta al embedding?"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer

m = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def sim(a, b):
    va = m.encode(a, normalize_embeddings=True)
    vb = m.encode(b, normalize_embeddings=True)
    return float(np.dot(va, vb))

# Las 5 frases que fallan, CON y SIN tildes
test_cases = [
    {
        "con": "Me mandaron un mensaje y no me contestaron, me siento ignorado",
        "sin": "Me mandaron un mensaje y no me contestaron, me siento ignorado",  # same
        "expected": "sombra_rechazo",
    },
    {
        "con": "No me invitaron a la salida del grupo",
        "sin": "No me invitaron a la salida del grupo",  # same
        "expected": "sombra_rechazo",
    },
    {
        "con": "Voy a cargar creatina hoy, se me acabó la semana pasada",
        "sin": "Voy a cargar creatina hoy, se me acabo la semana pasada",
        "expected": "sombra_fortaleza",
    },
    {
        "con": "Me siento abrumado por todo lo que tengo que aprender",
        "sin": "Me siento abrumado por todo lo que tengo que aprender",  # same
        "expected": "sombra_angel_atardecer",
    },
    {
        "con": "La nona está mejor pero sigue en cama, me da pena verla así",
        "sin": "La nona esta mejor pero sigue en cama, me da pena verla asi",
        "expected": "sombra_muerte",
    },
]

# Peers CON tildes (como en el benchmark original)
peers_con = {
    "sombra_muerte": (
        "El hermano que murió antes de nacer, el cordón que se enroscó, el silencio que "
        "dejó un vacío en la familia antes de que yo existiera. La muerte como presencia "
        "constante, no como evento lejano. El miedo a que los seres queridos desaparezcan "
        "sin aviso. La fragilidad del cuerpo, lo fácil que es que algo se corte. Los sueños "
        "donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar "
        "de estar en cualquier momento. La relación con la nona, esperando una operación, "
        "la familia acompañando en silencio."
    ),
    "sombra_rechazo": (
        "La madre que perdió un hijo antes de que él naciera, y ese bebé que nació "
        "sintiendo que el amor materno tenía una grieta invisible. Para un niño, que "
        "mamá no te mire es desaparecer. Esa ecuación se grabó tan profundo que después "
        "toda mujer que no lo mira siente igual. La fiesta donde ella no le hizo caso. "
        "El grupo que no lo incluyó. La sensación de no pertenecer, de estar siempre un "
        "paso afuera. El patrón de buscar amor donde no florece, de acercarse esperando "
        "la puerta cerrada. No es solo romance, es existencial: si me rechazan, dejo de importar."
    ),
    "sombra_angel_atardecer": (
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
    "sombra_fortaleza": (
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
}

# Peers SIN tildes
peers_sin = {}
for pid, text in peers_con.items():
    sin = text.replace('ó', 'o').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ú', 'u').replace('ñ', 'n').replace('—', '-')
    peers_sin[pid] = sin

# Embed peers CON y SIN tildes
peers_con_emb = {pid: m.encode(text, normalize_embeddings=True) for pid, text in peers_con.items()}
peers_sin_emb = {pid: m.encode(text, normalize_embeddings=True) for pid, text in peers_sin.items()}

print("=" * 65)
print("  TEST D: TILDES vs SIN TILDES")
print("=" * 65)

# Primero: cuánto cambia el embedding del mismo peer con/sin tildes?
print("\n  [1] IMPACTO DE TILDES EN EMBEDDINGS DE PEERS:")
for pid in peers_con:
    s = float(np.dot(peers_con_emb[pid], peers_sin_emb[pid]))
    print(f"    {pid:30s} con vs sin: {s:.6f}")

# Segundo: las frases de test CON tildes vs SIN tildes
print("\n  [2] FRASES CON/SIN TILDES vs PEERS CON TILDES:")
for i, tc in enumerate(test_cases):
    phrase_con = tc["con"]
    phrase_sin = tc["sin"]
    expected = tc["expected"]
    
    emb_con = m.encode(phrase_con, normalize_embeddings=True)
    emb_sin = m.encode(phrase_sin, normalize_embeddings=True)
    
    # Self-similarity (con vs sin)
    self_sim = float(np.dot(emb_con, emb_sin))
    
    print(f"\n  [{i+1}] \"{phrase_sin[:50]}\"")
    print(f"       Con vs sin tildes: {self_sim:.6f}")
    
    # Similarity with expected peer, con tildes en peer
    for label, phrase_emb in [("frase con", emb_con), ("frase sin", emb_sin)]:
        sims = {}
        for pid, pemb in peers_con_emb.items():
            sims[pid] = float(np.dot(phrase_emb, pemb))
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_pid, best_sim = sorted_sims[0]
        exp_sim = sims[expected]
        
        status = "OK" if best_pid == expected else f"FALLA({best_pid})"
        print(f"       {label} tildes → {expected}: {exp_sim:.4f}  best: {best_pid}({best_sim:.4f}) [{status}]")

# Tercero: frase SIN tildes vs peers SIN tildes
print("\n\n  [3] TODO SIN TILDES (frase + peers):")
for i, tc in enumerate(test_cases):
    phrase_sin = tc["sin"]
    expected = tc["expected"]
    phrase_emb = m.encode(phrase_sin, normalize_embeddings=True)
    
    sims = {}
    for pid, pemb in peers_sin_emb.items():
        sims[pid] = float(np.dot(phrase_emb, pemb))
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_pid, best_sim = sorted_sims[0]
    exp_sim = sims[expected]
    
    status = "OK" if best_pid == expected else f"FALLA({best_pid})"
    print(f"    [{i+1}] \"{phrase_sin[:45]}\" → {expected}: {exp_sim:.4f}  best: {best_pid}({best_sim:.4f}) [{status}]")

# Cuarto: todo CON tildes
print("\n\n  [4] TODO CON TILDES (frase + peers):")
for i, tc in enumerate(test_cases):
    phrase_con = tc["con"]
    expected = tc["expected"]
    phrase_emb = m.encode(phrase_con, normalize_embeddings=True)
    
    sims = {}
    for pid, pemb in peers_con_emb.items():
        sims[pid] = float(np.dot(phrase_emb, pemb))
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    best_pid, best_sim = sorted_sims[0]
    exp_sim = sims[expected]
    
    status = "OK" if best_pid == expected else f"FALLA({best_pid})"
    print(f"    [{i+1}] \"{phrase_con[:45]}\" → {expected}: {exp_sim:.4f}  best: {best_pid}({best_sim:.4f}) [{status}]")

# Quinto: la frase clave del benchmark "Si lloro es debilidad"
print("\n\n  [5] CASO FAMOSO: 'Si lloro es debilidad' vs 'Si lloro es debilidad'")
for phrase in ["Si lloro es debilidad", "Si lloro es debilidad"]:  # same
    pass  # no tildes difference here

# Variations
variations = [
    "Si lloro es debilidad",
    "Si lloro es debilidad",
    "Si lloro, es debilidad",
    "Llorar es de debiles",
    "Mostrar emocion es debilidad",
]
print()
for v in variations:
    vemb = m.encode(v, normalize_embeddings=True)
    f_sim = float(np.dot(vemb, peers_con_emb["sombra_fortaleza"]))
    m_sim = float(np.dot(vemb, peers_con_emb["sombra_muerte"]))
    print(f"    \"{v:40s}\" → Fort: {f_sim:.4f}  Muer: {m_sim:.4f}  gap: {f_sim-m_sim:+.4f}")
