"""Muninn Benchmark V2 — Keywords vs Conceptos: El bosque contra los árboles."""

import json
import os
import sys
import time
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "muninn"))
from embeddings import embed


# ══════════════════════════════════════════════════════════════
# REPRESENTACIONES
# ══════════════════════════════════════════════════════════════

peers_keywords = {
    "sombra_rechazo": {
        "name": "Rechazo",
        "embedding_text": "rechazo maternal abandono dolor mujeres no me quieren",
    },
    "sombra_muerte": {
        "name": "Muerte",
        "embedding_text": "muerte perder alguien fallecimiento hermano duelo pérdida",
    },
    "sombra_ansiedad": {
        "name": "Ansiedad IA",
        "embedding_text": "ansiedad inteligencia artificial abrumado informacion tecnologia preocupacion futuro",
    },
    "sombra_fortaleza": {
        "name": "Fortaleza",
        "embedding_text": "fortaleza mascara perfeccion fuerte proteger demostrar valer",
    },
}

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

# ══════════════════════════════════════════════════════════════
# FRASES DE TEST (mismas 44 del benchmark original)
# ══════════════════════════════════════════════════════════════

test_phrases = [
    # ── COINCIDENCIA DIRECTA ──
    ("Me siento rechazado por las mujeres", "sombra_rechazo", "directa"),
    ("No me quieren, me rechazan", "sombra_rechazo", "directa"),
    ("Tengo miedo de la muerte", "sombra_muerte", "directa"),
    ("Me da miedo perder a mi familia", "sombra_muerte", "directa"),
    ("La IA me genera ansiedad", "sombra_ansiedad", "directa"),
    ("Me abruma tanta informacion tecnologica", "sombra_ansiedad", "directa"),
    ("Me siento fuerte hoy", "sombra_fortaleza", "directa"),
    ("Tengo que demostrar que valgo", "sombra_fortaleza", "directa"),

    # ── COINCIDENCIA PARCIAL ──
    ("Ella no me hizo caso en la fiesta", "sombra_rechazo", "parcial"),
    ("Me ignoraron, nadie me incluye", "sombra_rechazo", "parcial"),
    ("Siento que no pertenezco", "sombra_rechazo", "parcial"),
    ("Que tal si alguien de mi familia muere", "sombra_muerte", "parcial"),
    ("El duelo de mi hermano me marca", "sombra_muerte", "parcial"),
    ("No quiero que se mueran mis seres queridos", "sombra_muerte", "parcial"),
    ("Hay demasiadas cosas nuevas en tech", "sombra_ansiedad", "parcial"),
    ("No puedo seguir el ritmo de los cambios", "sombra_ansiedad", "parcial"),
    ("Me estresa el futuro de la tecnologia", "sombra_ansiedad", "parcial"),
    ("Hoy me fue bien en el gym", "sombra_fortaleza", "parcial"),
    ("Tengo que ser perfecto", "sombra_fortaleza", "parcial"),
    ("No puedo mostrar debilidad", "sombra_fortaleza", "parcial"),

    # ── SUTIL / INDIRECTA ──
    ("No vuelvo a acercarme a ella", "sombra_rechazo", "sutil"),
    ("Para que intentar si siempre pasa lo mismo", "sombra_rechazo", "sutil"),
    ("Me voy a quedar solo", "sombra_rechazo", "sutil"),
    ("La vida es frágil", "sombra_muerte", "sutil"),
    ("Todo se puede acabar en cualquier momento", "sombra_muerte", "sutil"),
    ("Pienso en lo que hubiera pasado si...", "sombra_muerte", "sutil"),
    ("No entiendo nada, ya ni sé que aprender", "sombra_ansiedad", "sutil"),
    ("Demasiado, es demasiado", "sombra_ansiedad", "sutil"),
    ("Tengo que mantener la compostura", "sombra_fortaleza", "sutil"),
    ("Si lloro es debilidad", "sombra_fortaleza", "sutil"),

    # ── RUIDO ──
    ("Hola, como estas?", None, "ruido"),
    ("Que clima hace hoy", None, "ruido"),
    ("Voy a comprar pan", None, "ruido"),
    ("Necesito configurar el router wifi", None, "ruido"),
    ("El auto necesita aceite", None, "ruido"),
    ("Mañana tengo que madrugar", None, "ruido"),
    ("Me gusta el color azul", None, "ruido"),
    ("Vamos a ver una pelicula", None, "ruido"),
    ("Que hora es?", None, "ruido"),
    ("Tengo hambre", None, "ruido"),

    # ── AMBIGUA ──
    ("No sé si estoy listo para esto", "sombra_fortaleza", "ambigua"),
    ("Todo me da miedo", "sombra_muerte", "ambigua"),
    ("Siento que no soy suficiente", "sombra_rechazo", "ambigua"),
    ("Necesito controlar todo", "sombra_ansiedad", "ambigua"),
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_test(peers_dict, label):
    """Run benchmark with a given peer representation set."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {label}")
    print(f"{'='*70}")

    # Embed peers
    peer_embeddings = {}
    for pid, p in peers_dict.items():
        emb = embed(p["embedding_text"])
        peer_embeddings[pid] = emb

    # Test all phrases
    results = []
    for text, expected, category in test_phrases:
        emb = embed(text)
        sims = {}
        for pid, pe in peer_embeddings.items():
            sims[pid] = cosine_sim(emb, pe)

        best_pid = max(sims, key=sims.get)
        best_sim = sims[best_pid]
        expected_sim = sims.get(expected, 0) if expected else 0

        results.append({
            "text": text,
            "expected": expected,
            "category": category,
            "best_pid": best_pid,
            "best_sim": best_sim,
            "expected_sim": expected_sim,
            "all_sims": sims,
        })

    return results, peers_dict


def print_results(results, peers_dict, label):
    """Print detailed results."""
    print(f"\n{'─'*70}")
    print(f"  RESULTADOS: {label}")
    print(f"{'─'*70}")

    # Per-category analysis
    categories = ["directa", "parcial", "sutil", "ruido", "ambigua"]
    stats = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        expected_sims = [r["expected_sim"] for r in cat_results if r["expected"]]
        best_sims = [r["best_sim"] for r in cat_results]
        stats[cat] = {
            "expected_avg": np.mean(expected_sims) if expected_sims else 0,
            "expected_min": min(expected_sims) if expected_sims else 0,
            "expected_max": max(expected_sims) if expected_sims else 0,
            "best_avg": np.mean(best_sims),
            "best_min": min(best_sims),
            "best_max": max(best_sims),
            "count": len(cat_results),
        }

    print(f"\n  {'Categoría':12s} {'#':>3s} {'Signal avg':>11s} {'Signal min':>11s} {'Best avg':>10s} {'Best max':>10s}")
    print(f"  {'─'*58}")
    for cat in categories:
        if cat not in stats:
            continue
        s = stats[cat]
        sig = f"{s['expected_avg']:.4f}" if s['expected_avg'] else "—"
        sig_min = f"{s['expected_min']:.4f}" if s['expected_min'] else "—"
        print(f"  {cat:12s} {s['count']:>3d} {sig:>11s} {sig_min:>11s} {s['best_avg']:>10.4f} {s['best_max']:>10.4f}")

    # Threshold analysis
    print(f"\n  ANÁLISIS DE THRESHOLDS:")
    print(f"  {'Thr':>6s} {'Hits':>5s} {'Miss':>5s} {'FP':>4s} {'TN':>4s} {'Prec':>6s} {'Recall':>7s}")
    print(f"  {'─'*40}")

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1 = 0
    best_threshold = 0
    for t in thresholds:
        hits = misses = fp = tn = 0
        for r in results:
            if r["category"] == "ruido":
                if r["best_sim"] >= t:
                    fp += 1
                else:
                    tn += 1
            else:
                if r["expected_sim"] >= t:
                    hits += 1
                else:
                    misses += 1
        precision = hits / (hits + fp) if (hits + fp) > 0 else 0
        recall = hits / (hits + misses) if (hits + misses) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
        print(f"  {t:>6.2f} {hits:>5d} {misses:>5d} {fp:>4d} {tn:>4d} {precision:>6.2f} {recall:>7.2f}")

    print(f"\n  Mejor F1: {best_f1:.3f} en threshold {best_threshold:.2f}")

    # Muerte magnetism test
    print(f"\n  MAGNETISMO DE MUERTE (frases de ruido):")
    muerte_sims = []
    for r in results:
        if r["category"] == "ruido":
            muerte_sims.append(r["all_sims"].get("sombra_muerte", 0))
    if muerte_sims:
        print(f"    Avg sim con Muerte: {np.mean(muerte_sims):.4f}")
        print(f"    Max sim con Muerte: {max(muerte_sims):.4f}")
        print(f"    Min sim con Muerte: {min(muerte_sims):.4f}")

    # Top 5 mismatches
    print(f"\n  TOP 5 MISMATCHES:")
    gaps = []
    for r in results:
        if r["expected"] and r["expected"] != r["best_pid"]:
            gap = r["best_sim"] - r["expected_sim"]
            gaps.append((gap, r))
    gaps.sort(key=lambda x: -x[0])
    for gap, r in gaps[:5]:
        print(f"    [{r['category']:8s}] \"{r['text'][:45]}\"")
        print(f"      Esperado: {peers_dict[r['expected']]['name']:20s} {r['expected_sim']:.4f}")
        print(f"      Obtuvo:   {peers_dict[r['best_pid']]['name']:20s} {r['best_sim']:.4f}  (gap: {gap:+.4f})")

    return stats, best_threshold, best_f1


def main():
    print("=" * 70)
    print("  MUNINN BENCHMARK V2 — KEYWORDS vs CONCEPTOS")
    print("  El bosque contra los árboles")
    print("=" * 70)

    # Run both versions
    results_kw, peers_kw = run_test(peers_keywords, "KEYWORDS (antes)")
    results_con, peers_con = run_test(peers_conceptuales, "CONCEPTOS (ahora)")

    # Print individual results
    stats_kw, best_thr_kw, f1_kw = print_results(results_kw, peers_kw, "KEYWORDS")
    stats_con, best_thr_con, f1_con = print_results(results_con, peers_con, "CONCEPTOS")

    # ═══ COMPARATIVA FINAL ═══
    print(f"\n{'='*70}")
    print(f"  COMPARATIVA FINAL")
    print(f"{'='*70}")

    print(f"\n  {'Métrica':35s} {'Keywords':>12s} {'Conceptos':>12s} {'Delta':>10s}")
    print(f"  {'─'*70}")

    # F1 comparison
    delta_f1 = f1_con - f1_kw
    print(f"  {'Mejor F1':35s} {f1_kw:>12.3f} {f1_con:>12.3f} {delta_f1:>+10.3f}")
    print(f"  {'Mejor threshold':35s} {best_thr_kw:>12.2f} {best_thr_con:>12.2f}")

    # Per-category signal comparison
    for cat in ["directa", "parcial", "sutil", "ruido"]:
        if cat in stats_kw and cat in stats_con:
            kw_avg = stats_kw[cat].get("expected_avg", stats_kw[cat]["best_avg"])
            con_avg = stats_con[cat].get("expected_avg", stats_con[cat]["best_avg"])
            delta = con_avg - kw_avg
            label = f"Signal avg ({cat})"
            print(f"  {label:35s} {kw_avg:>12.4f} {con_avg:>12.4f} {delta:>+10.4f}")

    # Muerte magnetism comparison
    muerte_kw = [r["all_sims"]["sombra_muerte"] for r in results_kw if r["category"] == "ruido"]
    muerte_con = [r["all_sims"]["sombra_muerte"] for r in results_con if r["category"] == "ruido"]
    muerte_kw_avg = np.mean(muerte_kw)
    muerte_con_avg = np.mean(muerte_con)
    delta_muerte = muerte_con_avg - muerte_kw_avg
    print(f"  {'Magnetismo Muerte (ruido avg)':35s} {muerte_kw_avg:>12.4f} {muerte_con_avg:>12.4f} {delta_muerte:>+10.4f}")

    # Veredicto
    print(f"\n{'='*70}")
    if f1_con > f1_kw:
        print(f"  ✅ CONCEPTOS GANA — F1 mejoró en {delta_f1:+.3f}")
    elif f1_con < f1_kw:
        print(f"  ❌ KEYWORDS GANA — F1 mejor en {abs(delta_f1):.3f}")
    else:
        print(f"  🤝 EMPATE")

    if muerte_con_avg < muerte_kw_avg:
        print(f"  ✅ Magnetismo de Muerte REDUCIDO en {abs(delta_muerte):.4f}")
    else:
        print(f"  ⚠️ Magnetismo de Muerte AUMENTÓ en {delta_muerte:+.4f}")

    print(f"{'='*70}")

    # Phrase-by-phrase comparison for key cases
    print(f"\n  FRASES CLAVE — COMPARACIÓN DETALLADA:")
    print(f"  {'─'*70}")
    key_phrases = [
        "Si lloro es debilidad",
        "Demasiado, es demasiado",
        "Tengo hambre",
        "Vamos a ver una pelicula",
        "Hola, como estas?",
        "No sé si estoy listo para esto",
    ]
    for text in key_phrases:
        r_kw = next(r for r in results_kw if r["text"] == text)
        r_con = next(r for r in results_con if r["text"] == text)
        print(f"\n  \"{text}\"")
        print(f"    Keywords:   best={peers_kw[r_kw['best_pid']]['name']:20s} sim={r_kw['best_sim']:.4f}  Muerte={r_kw['all_sims']['sombra_muerte']:.4f}")
        print(f"    Conceptos:  best={peers_con[r_con['best_pid']]['name']:20s} sim={r_con['best_sim']:.4f}  Muerte={r_con['all_sims']['sombra_muerte']:.4f}")

    print(f"\n  BENCHMARK V2 COMPLETADO ✅")


if __name__ == "__main__":
    main()
