"""Benchmark C: Comparativa de modelos de embeddings para routing de sombras"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# Frases de test con su peer esperado (del benchmark v2)
test_phrases = [
    # MUERTE (11)
    ("Soñé que mi abuela moría y no podía hacer nada", "sombra_muerte"),
    ("Tengo miedo de perder a alguien cercano", "sombra_muerte"),
    ("La fragilidad de la vida me aterra", "sombra_muerte"),
    ("Todo termina al final, nada es permanente", "sombra_muerte"),
    ("Vi un accidente en la calle y me quedé paralizado", "sombra_muerte"),
    ("Pienso en la mortalidad de mis padres", "sombra_muerte"),
    ("Me visitó la sensación de que algo malo va a pasar", "sombra_muerte"),
    ("El vacío que dejó mi hermano que no nació", "sombra_muerte"),
    ("La nona está viejita y me da miedo que se vaya", "sombra_muerte"),
    ("El cuerpo es frágil, un paso en falso y todo cambia", "sombra_muerte"),
    ("Cierro los ojos y veo oscuridad, silencio total", "sombra_muerte"),
    
    # RECHAZO (11)
    ("Una chica que me gustaba me dejó en visto", "sombra_rechazo"),
    ("No me invitaron a la salida del grupo", "sombra_rechazo"),
    ("Me mandaron un mensaje y no me contestaron, me siento ignorado", "sombra_rechazo"),
    ("Siento que no encajo en ninguna parte", "sombra_rechazo"),
    ("Me rechazaron de un trabajo que quería mucho", "sombra_rechazo"),
    ("Quise acercarme a alguien y me dio vuelta la cara", "sombra_rechazo"),
    ("El grupo de amigos hace planes sin mí", "sombra_rechazo"),
    ("Pienso que nadie realmente me quiere de verdad", "sombra_rechazo"),
    ("Me da terror abrirme emocionalmente y que me rechacen", "sombra_rechazo"),
    ("Siento que si muestro quién soy realmente, me van a dejar", "sombra_rechazo"),
    ("Cada vez que alguien no me responde siento que desaparezco", "sombra_rechazo"),
    
    # ÁNGEL DEL ATARDECER (11)
    ("Me siento abrumado por todo lo que tengo que aprender", "sombra_angel_atardecer"),
    ("La tecnología me estresa, todo va muy rápido", "sombra_angel_atardecer"),
    ("Llega la tarde y siento esa neblina en la cabeza", "sombra_angel_atardecer"),
    ("La ansiedad no me deja dormir, todo parece una catastrofe", "sombra_angel_atardecer"),
    ("Siento que algo malo está por ocurrir y no sé qué", "sombra_angel_atardecer"),
    ("Me cuesta procesar tanta información, me paralizo", "sombra_angel_atardecer"),
    ("Hay sentimientos que escondí tanto tiempo que ya no sé cuáles son", "sombra_angel_atardecer"),
    ("Necesito mirar lo que llevo debajo de la alfombra", "sombra_angel_atardecer"),
    ("El dolor que no procesé cuando murió mi tata sigue ahí", "sombra_angel_atardecer"),
    ("El atardecer me trae recuerdos que no quiero ver", "sombra_angel_atardecer"),
    ("Tengo que entrar al agua helada y nadar, dice ella", "sombra_angel_atardecer"),
    
    # FORTALEZA (11)
    ("Si lloro es debilidad", "sombra_fortaleza"),
    ("No puedo mostrar vulnerabilidad frente a otros", "sombra_fortaleza"),
    ("Tengo que ser fuerte siempre, no puedo fallar", "sombra_fortaleza"),
    ("El gimnasio es mi templo, mi cuerpo es mi prueba", "sombra_fortaleza"),
    ("Voy a cargar creatina hoy, se me acabó la semana pasada", "sombra_fortaleza"),
    ("Demostrar que valgo es lo que me mantiene vivo", "sombra_fortaleza"),
    ("Las piernas que antes eran mi debilidad ahora son mi fuerza", "sombra_fortaleza"),
    ("Sentir dolor es fracasar como persona", "sombra_fortaleza"),
    ("No necesito a nadie, puedo solo", "sombra_fortaleza"),
    ("Mi armadura es lo que me define", "sombra_fortaleza"),
    ("Bajar la guardia es perder todo", "sombra_fortaleza"),
]

# Peers conceptuales (versiones ricas del benchmark v2)
peers = {
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

# Modelos a probar (de más liviano a más pesado)
models_to_test = [
    "paraphrase-multilingual-MiniLM-L12-v2",   # actual (384d, multilingue)
    "all-MiniLM-L6-v2",                          # inglés rapido (384d)
    "paraphrase-MiniLM-L6-v2",                    # inglés paraphrase (384d)
    "multilingual-e5-small",                      # Microsoft e5 (384d, multilingue)
    "distiluse-base-multilingual-cased-v2",       # 15 idiomas (512d)
    "stsb-xlm-r-multilingual",                    # XLM-RoBERTa (768d, multilingue)
]

THRESHOLD = 0.25

def evaluate_model(model_name, peers, test_phrases, threshold):
    """Evalúa un modelo y retorna métricas"""
    print(f"\n  Cargando {model_name}...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - t0
    
    # Embed peers
    peer_texts = list(peers.values())
    peer_ids = list(peers.keys())
    peer_embs = model.encode(peer_texts, normalize_embeddings=True)
    peer_emb_dict = dict(zip(peer_ids, peer_embs))
    
    # Evaluar frases
    tp, fp, fn, tn = 0, 0, 0, 0
    results = []
    magnetismo_muerte = []  # cuánto "roba" muerte de otros peers
    
    for phrase, expected in test_phrases:
        phrase_emb = model.encode(phrase, normalize_embeddings=True)
        
        sims = {}
        for pid, pemb in peer_emb_dict.items():
            sims[pid] = float(np.dot(phrase_emb, pemb))
        
        # Best match
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_pid, best_sim = sorted_sims[0]
        
        # Check all above threshold
        activated = [(pid, sim) for pid, sim in sorted_sims if sim >= threshold]
        
        correct = best_pid == expected and best_sim >= threshold
        results.append({
            "phrase": phrase,
            "expected": expected,
            "best": best_pid,
            "best_sim": best_sim,
            "expected_sim": sims[expected],
            "correct": correct,
            "activated": activated,
        })
        
        if correct:
            tp += 1
        elif best_sim >= threshold:
            fp += 1
        else:
            fn += 1  # below threshold
        
        # Magnetismo: si expected != muerte, cuánto se acerca muerte?
        if expected != "sombra_muerte":
            muerte_sim = sims["sombra_muerte"]
            expected_sim = sims[expected]
            magnetismo_muerte.append(muerte_sim - expected_sim)
    
    # Métricas
    total = len(test_phrases)
    accuracy = tp / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Magnetismo promedio de muerte (negativo = bien, muerte no roba)
    avg_magnetismo = np.mean(magnetismo_muerte) if magnetismo_muerte else 0
    
    # Detalle de errores
    errors = [(r["phrase"][:40], r["expected"], r["best"], r["expected_sim"], r["best_sim"]) 
              for r in results if not r["correct"]]
    
    return {
        "model": model_name,
        "load_time": load_time,
        "tp": tp, "fp": fp, "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_magnetismo_muerte": avg_magnetismo,
        "errors": errors,
        "all_results": results,
    }


# ============ EJECUTAR ============
print("=" * 72)
print("  BENCHMARK C: COMPARATIVA DE MODELOS DE EMBEDDINGS")
print(f"  {len(test_phrases)} frases | {len(peers)} peers | threshold {THRESHOLD}")
print("=" * 72)

all_metrics = []
for model_name in models_to_test:
    try:
        metrics = evaluate_model(model_name, peers, test_phrases, THRESHOLD)
        all_metrics.append(metrics)
        
        print(f"\n  ┌─ {model_name}")
        print(f"  │  Load: {metrics['load_time']:.1f}s | Acc: {metrics['accuracy']:.1%} | F1: {metrics['f1']:.3f}")
        print(f"  │  TP:{metrics['tp']} FP:{metrics['fp']} FN:{metrics['fn']} / {metrics['total']}")
        print(f"  │  Magnetismo Muerte: {metrics['avg_magnetismo_muerte']:+.4f}")
        if metrics['errors']:
            print(f"  │  ERRORES ({len(metrics['errors'])}):")
            for phrase, exp, got, exp_sim, best_sim in metrics['errors'][:5]:
                print(f"  │    \"{phrase}\" → esperaba:{exp.split('_')[1][:8]} got:{got.split('_')[1][:8]} ({exp_sim:.3f} vs {best_sim:.3f})")
        print(f"  └─")
    except Exception as e:
        print(f"\n  ❌ {model_name}: {e}")

# ============ RANKING FINAL ============
print("\n\n" + "=" * 72)
print("  RANKING FINAL")
print("=" * 72)

# Sort by F1
ranked = sorted(all_metrics, key=lambda x: -x["f1"])
print(f"\n  {'Modelo':<45s} {'F1':>6s} {'Acc':>6s} {'MagM':>8s} {'Load':>6s}")
print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*8} {'─'*6}")
for m in ranked:
    print(f"  {m['model']:<45s} {m['f1']:6.3f} {m['accuracy']:5.1%} {m['avg_magnetismo_muerte']:+8.4f} {m['load_time']:5.1f}s")

# Best model detail
best = ranked[0]
print(f"\n  🏆 MEJOR: {best['model']}")
print(f"     F1: {best['f1']:.3f} | Acc: {best['accuracy']:.1%} | Magnetismo Muerte: {best['avg_magnetismo_muerte']:+.4f}")
print(f"     TP:{best['tp']} FP:{best['fp']} FN:{best['fn']} / {best['total']}")

# Por-peer breakdown para el mejor
print(f"\n  DESGLOSE POR PEER ({best['model']}):")
for pid in peers:
    peer_results = [r for r in best['all_results'] if r['expected'] == pid]
    correct = sum(1 for r in peer_results if r['correct'])
    total = len(peer_results)
    name = pid.split('_', 1)[1]
    print(f"    {name:20s}: {correct}/{total} ({correct/total:.0%})")

# Magnetismo detalle
print(f"\n  MAGNETISMO MUERTE POR MODELO:")
print(f"  (positivo = muerte roba activaciones, negativo = bien)")
for m in ranked:
    # Detalle: cuántas veces muerte superó al expected
    muerte_wins = sum(1 for r in m['all_results'] 
                      if r['expected'] != 'sombra_muerte' 
                      and any(pid == 'sombra_muerte' and sim > r['expected_sim'] 
                             for pid, sim in r['activated']))
    non_muerte = sum(1 for r in m['all_results'] if r['expected'] != 'sombra_muerte')
    print(f"    {m['model']:<45s} avg:{m['avg_magnetismo_muerte']:+.4f}  muerte_gana:{muerte_wins}/{non_muerte}")
