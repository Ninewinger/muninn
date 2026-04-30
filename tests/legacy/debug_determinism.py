import sys
sys.stdout.reconfigure(encoding='utf-8')

from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Model loaded OK')

# Determinism
v1 = m.encode('No me invitaron a la salida del grupo', normalize_embeddings=True)
v2 = m.encode('No me invitaron a la salida del grupo', normalize_embeddings=True)
print('Determinism:', np.allclose(v1, v2))
print('Max diff:', np.max(np.abs(v1 - v2)))

# Versions
import sentence_transformers
print('ST version:', sentence_transformers.__version__)
import torch
print('Torch version:', torch.__version__)

# The KEY test: does the benchmark v2 phrase "Si lloro es debilidad" still map to Fortaleza?
peer_text = (
    "La mascara que se construyo tan temprano que ya no sabe si es mascara o rostro. "
    "Ser fuerte para no sentir, demostrar que vale para no ser abandonado. El gimnasio "
    "como templo, el cuerpo como prueba viviente de que no eres debil. El nino que "
    "aprendio que si llora es debilidad, que si muestra vulnerabilidad pierde proteccion. "
    "Pero la fortaleza real no es la armadura, es el coraje de bajarla. Las piernas "
    "que antes eran la debilidad ahora son la respuesta - el breakthrough del gym fue "
    "encontrar que la verdadera fuerza esta en el territorio que mas miedo daba. La "
    "fortaleza como mecanismo de supervivencia que cumplio su funcion y ahora busca "
    "evolucionar: de escudo a presencia, de demostrar a ser."
)
muerte_text = (
    "El hermano que murio antes de nacer, el cordon que se enrosco, el silencio que "
    "dejo un vacio en la familia antes de que yo existiera. La muerte como presencia "
    "constante, no como evento lejano. El miedo a que los seres queridos desaparezcan "
    "sin aviso. La fragilidad del cuerpo, lo facil que es que algo se corte. Los suenos "
    "donde algo termina sin remedio. La conciencia de que todo lo que tengo puede dejar "
    "de estar en cualquier momento. La relacion con la nona, esperando una operacion, "
    "la familia acompanando en silencio."
)

phrase = "Si lloro es debilidad"
p_emb = m.encode(phrase, normalize_embeddings=True)
f_emb = m.encode(peer_text, normalize_embeddings=True)
m_emb = m.encode(muerte_text, normalize_embeddings=True)

sim_f = float(np.dot(p_emb, f_emb))
sim_m = float(np.dot(p_emb, m_emb))

print(f'\n\"Si lloro es debilidad\":')
print(f'  Fortaleza: {sim_f:.4f}')
print(f'  Muerte:    {sim_m:.4f}')
print(f'  Benchmark original: Fortaleza=0.58, Muerte=0.17')
print(f'  Delta Fortaleza: {sim_f - 0.58:+.4f}')
print(f'  Delta Muerte:    {sim_m - 0.17:+.4f}')

# Test with and without accents
phrase_no_acc = "Si lloro es debilidad"
phrase_acc = "Si lloro es debilidad"  # same
e1 = m.encode(phrase_no_acc, normalize_embeddings=True)
e2 = m.encode(phrase_acc, normalize_embeddings=True)
print(f'\nAcentos check: {np.allclose(e1, e2)}')
