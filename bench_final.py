#!/usr/bin/env python3
"""Final benchmark: test API /api/v1/route with hybrid strategy."""

import json, requests, time

API = "http://127.0.0.1:8901"

TEST_QUERIES = [
    # SOMBRAS
    {"query": "sone que mi abuela se caia y no podia levantarla", "expected": ["sombra_muerte"], "cat": "sombras"},
    {"query": "me mandaron el curriculum para arriba, ni me contestaron", "expected": ["sombra_rechazo"], "cat": "sombras"},
    {"query": "siento que cuando algo me va bien, algo malo tiene que pasar", "expected": ["sombra_angel_atardecer"], "cat": "sombras"},
    {"query": "no se si tengo realmente la disciplina para esto o soy debil", "expected": ["sombra_fortaleza"], "cat": "sombras"},
    # TEMAS
    {"query": "hoy tocaba pecho y triceps pero me duele el hombro", "expected": ["gym_rutina"], "cat": "temas"},
    {"query": "estoy aprendiendo sobre decoradores en python, los closures me confunden", "expected": ["programacion"], "cat": "temas"},
    {"query": "como funciona un context manager? quiero entender el protocolo", "expected": ["programacion"], "cat": "temas"},
    {"query": "voy a salir con unos amigos al bar del centro", "expected": ["casual_social"], "cat": "temas"},
    # PROYECTOS
    {"query": "estoy pensando en la mecanica de niveles para el juego, deberia ser infinito", "expected": ["proyecto_juego"], "cat": "proyectos"},
    {"query": "tengo que armar el plan de negocios para Valle Alto", "expected": ["valle_alto"], "cat": "proyectos"},
    # SISTEMA
    {"query": "puedes actualizarme el skill de debugging?", "expected": ["peer_skills"], "cat": "sistema"},
]

correct = 0
total = len(TEST_QUERIES)
results = []

for tq in TEST_QUERIES:
    t0 = time.time()
    r = requests.post(f"{API}/api/v1/route", json={"text": tq["query"], "top_k": 3, "strategy": "hybrid"}, timeout=30)
    elapsed = time.time() - t0
    data = r.json()

    activations = data.get("activations", [])
    top_peers = [a.get("peer_id", "") for a in activations[:3]]

    # Check if ANY expected peer is in top results
    hit = any(exp in top_peers for exp in tq["expected"])
    if hit:
        correct += 1

    results.append({
        "query": tq["query"][:50],
        "expected": tq["expected"],
        "got": top_peers[:3],
        "hit": hit,
        "latency": round(elapsed, 2),
        "cat": tq["cat"],
    })

    status = "OK  " if hit else "MISS"
    print(f"  [{status}] {tq['query'][:45]:45s} expected={tq['expected']}, got={top_peers[:2]} ({elapsed:.2f}s)")

pct = correct / total * 100
print("")
print("=" * 60)
print(f"RESULTS: {correct}/{total} = {pct:.0f}% (target: >70%)")
print("=" * 60)

# Save results
with open("/mnt/d/github/muninn/benchmark_final.json", "w") as f:
    json.dump({"accuracy": pct, "correct": correct, "total": total, "results": results}, f, indent=2, ensure_ascii=False)
print("Saved to benchmark_final.json")
