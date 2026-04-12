"""Muninn Similarity Benchmark — mapear distribucion real de similitudes."""

import json
import os
import sys
import time
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
from muninn.embeddings import embed
from muninn.db import init_db

DB_PATH = os.path.join(os.path.dirname(__file__), "bench_muninn.db")
os.environ["DB_PATH"] = DB_PATH

if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)

# ── DATOS DE TEST ──

# Peers con sus embedding_text (lo que el sistema conoce de cada sombra)
peers = {
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

# Frases de test organizadas por categoría
# Cada frase tiene: (texto, peer_esperado, categoria)
test_phrases = [
    # ── COINCIDENCIA DIRECTA (mismo tema, palabras similares) ──
    ("Me siento rechazado por las mujeres", "sombra_rechazo", "directa"),
    ("No me quieren, me rechazan", "sombra_rechazo", "directa"),
    ("Tengo miedo de la muerte", "sombra_muerte", "directa"),
    ("Me da miedo perder a mi familia", "sombra_muerte", "directa"),
    ("La IA me genera ansiedad", "sombra_ansiedad", "directa"),
    ("Me abruma tanta informacion tecnologica", "sombra_ansiedad", "directa"),
    ("Me siento fuerte hoy", "sombra_fortaleza", "directa"),
    ("Tengo que demostrar que valgo", "sombra_fortaleza", "directa"),

    # ── COINCIDENCIA PARCIAL (mismo tema, palabras diferentes) ──
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

    # ── SUTIL / INDIRECTA (mismo tema, muy indirecto) ──
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

    # ── RUIDO / SIN RELACION (no deberia activar nada) ──
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

    # ── AMBIGUA (podria activar mas de una) ──
    ("No sé si estoy listo para esto", "sombra_fortaleza", "ambigua"),
    ("Todo me da miedo", "sombra_muerte", "ambigua"),
    ("Siento que no soy suficiente", "sombra_rechazo", "ambigua"),
    ("Necesito controlar todo", "sombra_ansiedad", "ambigua"),
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=" * 70)
    print("  MUNINN SIMILARITY BENCHMARK")
    print("=" * 70)

    # 1. Generar embeddings de peers
    print("\n[1] Generando embeddings de peers...")
    peer_embeddings = {}
    for pid, p in peers.items():
        emb = embed(p["embedding_text"])
        peer_embeddings[pid] = emb
        print(f"  + {p['name']}: {len(emb)}D vector")

    # 2. Testear todas las frases
    print(f"\n[2] Evaluando {len(test_phrases)} frases de test...")
    print("-" * 70)

    results = []
    for text, expected, category in test_phrases:
        emb = embed(text)
        sims = {}
        for pid, pe in peer_embeddings.items():
            sims[pid] = cosine_sim(emb, pe)

        # Best match
        best_pid = max(sims, key=sims.get)
        best_sim = sims[best_pid]
        best_name = peers[best_pid]["name"]

        # Expected match
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

        print(f"\n  [{category.upper():8s}] \"{text[:50]}\"")
        print(f"    Expected: {peers[expected]['name'] if expected else 'NINGUNO':12s} sim={expected_sim:.4f}")
        print(f"    Best:     {best_name:12s} sim={best_sim:.4f}")
        if expected and best_pid != expected:
            print(f"    *** MISMATCH ***")

    # 3. ANALISIS POR CATEGORIA
    print("\n" + "=" * 70)
    print("  ANALISIS POR CATEGORIA")
    print("=" * 70)

    categories = ["directa", "parcial", "sutil", "ruido", "ambigua"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        expected_sims = [r["expected_sim"] for r in cat_results if r["expected"]]
        best_sims = [r["best_sim"] for r in cat_results]

        print(f"\n  {cat.upper()} ({len(cat_results)} frases)")
        if expected_sims:
            print(f"    Sim con peer correcto:  min={min(expected_sims):.4f}  max={max(expected_sims):.4f}  avg={np.mean(expected_sims):.4f}")
        print(f"    Sim del best match:     min={min(best_sims):.4f}  max={max(best_sims):.4f}  avg={np.mean(best_sims):.4f}")

    # 4. DISTRIBUCION COMPLETA
    print("\n" + "=" * 70)
    print("  DISTRIBUCION DE SIMILITUDES")
    print("=" * 70)

    all_expected = [r["expected_sim"] for r in results if r["expected"]]
    all_ruido = [r["best_sim"] for r in results if r["category"] == "ruido"]

    print(f"\n  FRASES CON PEER ASIGNADO ({len(all_expected)} frases):")
    print(f"    Min:  {min(all_expected):.4f}")
    print(f"    P25:  {np.percentile(all_expected, 25):.4f}")
    print(f"    P50:  {np.percentile(all_expected, 50):.4f}")
    print(f"    P75:  {np.percentile(all_expected, 75):.4f}")
    print(f"    Max:  {max(all_expected):.4f}")
    print(f"    Avg:  {np.mean(all_expected):.4f}")

    print(f"\n  FRASES RUIDO ({len(all_ruido)} frases):")
    print(f"    Min:  {min(all_ruido):.4f}")
    print(f"    P25:  {np.percentile(all_ruido, 25):.4f}")
    print(f"    P50:  {np.percentile(all_ruido, 50):.4f}")
    print(f"    P75:  {np.percentile(all_ruido, 75):.4f}")
    print(f"    Max:  {max(all_ruido):.4f}")
    print(f"    Avg:  {np.mean(all_ruido):.4f}")

    # 5. THRESHOLD ANALYSIS
    print("\n" + "=" * 70)
    print("  ANALISIS DE THRESHOLDS")
    print("=" * 70)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n  {'Threshold':>10s}  {'Hits':>5s}  {'Misses':>6s}  {'FP':>5s}  {'TN':>5s}  {'Precision':>10s}  {'Recall':>8s}")
    print("  " + "-" * 60)

    for t in thresholds:
        hits = 0   # correcto: esperaba peer y activo ese peer
        misses = 0  # fallo: esperaba peer y no activo o activo otro
        fp = 0      # falso positivo: ruido que activo algo
        tn = 0      # true negative: ruido que no activo nada

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

        print(f"  {t:>10.2f}  {hits:>5d}  {misses:>6d}  {fp:>5d}  {tn:>5d}  {precision:>10.2f}  {recall:>8.2f}")

    # 6. TOP MISMATCHES
    print("\n" + "=" * 70)
    print("  PEORES CASOS (mayor gap entre expected y best)")
    print("=" * 70)

    gaps = []
    for r in results:
        if r["expected"] and r["expected"] != r["best_pid"]:
            gap = r["best_sim"] - r["expected_sim"]
            gaps.append((gap, r))

    gaps.sort(key=lambda x: -x[0])
    for gap, r in gaps[:5]:
        print(f"\n  [{r['category']}] \"{r['text'][:50]}\"")
        print(f"    Expected: {peers[r['expected']]['name']} (sim={r['expected_sim']:.4f})")
        print(f"    Got:      {peers[r['best_pid']]['name']} (sim={r['best_sim']:.4f})")
        print(f"    Gap:      {gap:+.4f}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETADO")
    print("=" * 70)

    # Cleanup
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)


if __name__ == "__main__":
    main()
