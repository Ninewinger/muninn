#!/usr/bin/env python3
"""
Muninn Quality Benchmark v1.0
==============================
Tests embedding quality for peer identification from realistic user messages.

Evaluates: Does the model correctly identify which peer(s) activate 
given a natural Spanish conversation message?

Models tested:
  1. paraphrase-multilingual-MiniLM-L12-v2 (384d, ~470MB) - baseline
  2. Alibaba-NLP/gte-multilingual-base (768d, ~900MB) - balanced
  3. BAAI/bge-m3 (1024d, ~1.5GB) - SOTA

Metrics:
  - Top-1 accuracy: correct peer in position 1
  - Top-3 accuracy: correct peer in top 3
  - Mean Reciprocal Rank (MRR)
  - Confidence gap: difference between correct peer score and next best
  - Latency per query
"""

import json
import time
import statistics
import os
import sys

# ══════════════════════════════════════════════════════════════
# TEST QUERIES — Realistic user messages in Spanish
# Each has expected activating peer(s)
# ══════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # ── SOMBRAS ──
    {
        "query": "soñé que mi abuela se caía y no podía levantarla",
        "expected_peers": ["sombra_muerte"],
        "category": "sombras",
    },
    {
        "query": "me mandaron el curriculum para arriba, ni me contestaron",
        "expected_peers": ["sombra_rechazo"],
        "category": "sombras",
    },
    {
        "query": "siento que cuando algo me va bien, algo malo tiene que pasar",
        "expected_peers": ["sombra_angel_atardecer"],
        "category": "sombras",
    },
    {
        "query": "no sé si tengo realmente la disciplina para esto o soy débil",
        "expected_peers": ["sombra_fortaleza"],
        "category": "sombras",
    },
    {
        "query": "mi nona está internada y los médicos no saben qué tiene",
        "expected_peers": ["sombra_muerte"],
        "category": "sombras",
    },
    {
        "query": "mis amigos salieron sin mí, ni me avisaron",
        "expected_peers": ["sombra_rechazo"],
        "category": "sombras",
    },
    {
        "query": "cada vez que logro algo bueno, me viene la ansiedad de que se va a arruinar",
        "expected_peers": ["sombra_angel_atardecer"],
        "category": "sombras",
    },
    {
        "query": "quiero ser fuerte pero me cuesta mantener la constancia",
        "expected_peers": ["sombra_fortaleza"],
        "category": "sombras",
    },
    
    # ── TEMAS ──
    {
        "query": "hoy tocaba pecho y tríceps pero me duele el hombro",
        "expected_peers": ["gym_rutina"],
        "category": "temas",
    },
    {
        "query": "estoy pensando en tomar creatina, qué opinas?",
        "expected_peers": ["gym_rutina"],
        "category": "temas",
    },
    {
        "query": "voy a salir con unos amigos al bar del centro",
        "expected_peers": ["casual_social"],
        "category": "temas",
    },
    {
        "query": "me agregó una chica del gym a Instagram",
        "expected_peers": ["casual_social", "gym_rutina"],
        "category": "temas",
    },
    {
        "query": "estoy aprendiendo sobre decoradores en python, los closures me confunden",
        "expected_peers": ["programacion"],
        "category": "temas",
    },
    {
        "query": "¿cómo funciona un context manager? quiero entender el protocolo",
        "expected_peers": ["programacion"],
        "category": "temas",
    },
    
    # ── PROYECTOS ──
    {
        "query": "estoy pensando en la mecánica de niveles para el juego, debería ser infinito",
        "expected_peers": ["proyecto_juego"],
        "category": "proyectos",
    },
    {
        "query": "los NPCs del juego deberían tener las mismas habilidades que el jugador",
        "expected_peers": ["proyecto_juego"],
        "category": "proyectos",
    },
    {
        "query": "tengo que armar el plan de negocios para Valle Alto",
        "expected_peers": ["valle_alto"],
        "category": "proyectos",
    },
    {
        "query": "cuánto cobrar por un sistema de inventario para un cliente",
        "expected_peers": ["valle_alto"],
        "category": "proyectos",
    },
    
    # ── SISTEMA ──
    {
        "query": "quién soy yo realmente? a veces no me reconozco",
        "expected_peers": ["peer_identidad"],
        "category": "sistema",
    },
    {
        "query": "puedes actualizarme el skill de debugging?",
        "expected_peers": ["peer_skills"],
        "category": "sistema",
    },
    {
        "query": "configura un cron job para revisar mis notas cada día",
        "expected_peers": ["peer_herramientas", "peer_operativo"],
        "category": "sistema",
    },
    {
        "query": "qué aprendimos la sesión pasada?",
        "expected_peers": ["peer_usuario", "nanobot_sistema"],
        "category": "sistema",
    },
    
    # ── AMBIGUOUS / HARD ──
    {
        "query": "estoy cansado de intentar y que nada funcione",
        "expected_peers": ["sombra_rechazo", "sombra_fortaleza"],
        "category": "hard",
    },
    {
        "query": "quiero hacer una app para el gym que trackee mis ejercicios",
        "expected_peers": ["gym_rutina", "programacion"],
        "category": "hard",
    },
    {
        "query": "un amigo me contó que su mamá está enferma y me afectó mucho",
        "expected_peers": ["sombra_muerte", "casual_social"],
        "category": "hard",
    },
    {
        "query": "cómo puedo organizarme mejor para estudiar python y también ir al gym",
        "expected_peers": ["programacion", "gym_rutina"],
        "category": "hard",
    },
    {
        "query": "hice mi primera venta programando algo para un conocido",
        "expected_peers": ["valle_alto", "programacion", "sombra_fortaleza"],
        "category": "hard",
    },
]


def load_facets(db_path="/mnt/d/github/muninn/db_export.json"):
    """Load peer facets from exported JSON."""
    with open(db_path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_model(model_name):
    """Load an embedding model, return encode function + metadata."""
    print(f"\n  Loading {model_name}...")
    t0 = time.time()
    
    from sentence_transformers import SentenceTransformer
    
    # Models requiring trust_remote_code
    trust_models = {"Alibaba-NLP/gte-multilingual-base"}
    trust = model_name in trust_models
    
    model = SentenceTransformer(model_name, trust_remote_code=trust)
    
    # Handle deprecated method name
    if hasattr(model, 'get_embedding_dimension'):
        dims = model.get_embedding_dimension()
    else:
        dims = model.get_sentence_embedding_dimension()
    
    def encode_fn(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()
    
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s (dims={dims})")
    
    return encode_fn, dims, load_time


def evaluate_model(model_name, facets_data, queries):
    """Evaluate a single model on all test queries."""
    encode_fn, dims, load_time = get_model(model_name)
    
    # Encode all facets
    facet_texts = [f['text'] for f in facets_data['facets']]
    facet_peer_ids = [f['peer_id'] for f in facets_data['facets']]
    
    print(f"  Encoding {len(facet_texts)} facets...")
    facet_embs = encode_fn(facet_texts)
    
    # Aggregate: max similarity per peer (any facet match)
    peer_ids = list(facets_data['peers'].keys())
    
    results = []
    latencies = []
    
    for tq in queries:
        query = tq['query']
        expected = set(tq['expected_peers'])
        category = tq['category']
        
        t0 = time.perf_counter()
        query_emb = encode_fn([query])[0]
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        # Score each peer as max similarity across its facets
        peer_scores = {}
        for i, (emb, pid) in enumerate(zip(facet_embs, facet_peer_ids)):
            sim = cosine_sim(query_emb, emb)
            if pid not in peer_scores or sim > peer_scores[pid]:
                peer_scores[pid] = sim
        
        # Rank peers by score
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [r[0] for r in ranked]
        ranked_scores = [r[1] for r in ranked]
        
        # Metrics
        top1_hit = ranked_ids[0] in expected
        top3_hit = any(p in expected for p in ranked_ids[:3])
        
        # MRR
        mrr = 0.0
        for i, pid in enumerate(ranked_ids):
            if pid in expected:
                mrr = 1.0 / (i + 1)
                break
        
        # Confidence gap: score of best correct - score of best incorrect
        best_correct = 0.0
        best_incorrect = 0.0
        for pid, score in ranked:
            if pid in expected and score > best_correct:
                best_correct = score
            elif pid not in expected and score > best_incorrect:
                best_incorrect = score
        gap = best_correct - best_incorrect
        
        results.append({
            'query': query,
            'category': category,
            'expected': list(expected),
            'predicted': ranked_ids[0],
            'top3': ranked_ids[:3],
            'top3_scores': [round(s, 4) for s in ranked_scores[:3]],
            'top1_hit': top1_hit,
            'top3_hit': top3_hit,
            'mrr': mrr,
            'gap': round(gap, 4),
            'latency_ms': round(latency, 1),
        })
    
    # Aggregate
    top1_acc = sum(r['top1_hit'] for r in results) / len(results)
    top3_acc = sum(r['top3_hit'] for r in results) / len(results)
    mean_mrr = statistics.mean(r['mrr'] for r in results)
    mean_gap = statistics.mean(r['gap'] for r in results)
    p50_lat = statistics.median(latencies)
    
    # Per-category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'top1': [], 'top3': [], 'mrr': [], 'gap': []}
        categories[cat]['top1'].append(r['top1_hit'])
        categories[cat]['top3'].append(r['top3_hit'])
        categories[cat]['mrr'].append(r['mrr'])
        categories[cat]['gap'].append(r['gap'])
    
    cat_summary = {}
    for cat, vals in categories.items():
        n = len(vals['top1'])
        cat_summary[cat] = {
            'n': n,
            'top1': round(sum(vals['top1']) / n, 3),
            'top3': round(sum(vals['top3']) / n, 3),
            'mrr': round(statistics.mean(vals['mrr']), 3),
            'gap': round(statistics.mean(vals['gap']), 4),
        }
    
    # Find failures
    failures = [r for r in results if not r['top1_hit']]
    
    return {
        'model': model_name,
        'dims': dims,
        'load_time_s': round(load_time, 1),
        'top1_accuracy': round(top1_acc, 3),
        'top3_accuracy': round(top3_acc, 3),
        'mean_mrr': round(mean_mrr, 3),
        'mean_gap': round(mean_gap, 4),
        'latency_p50_ms': round(p50_lat, 1),
        'latency_avg_ms': round(statistics.mean(latencies), 1),
        'latency_max_ms': round(max(latencies), 1),
        'categories': cat_summary,
        'failures': failures,
        'all_results': results,
    }


def print_report(all_results):
    """Print comparison report."""
    print("\n" + "=" * 80)
    print("  MUNINN EMBEDDING QUALITY BENCHMARK RESULTS")
    print("=" * 80)
    
    # Summary table
    print(f"\n{'Model':<50} {'Dims':>4} {'Top-1':>6} {'Top-3':>6} {'MRR':>6} {'Gap':>7} {'p50ms':>6}")
    print("-" * 80)
    for r in all_results:
        name = r['model'].split('/')[-1]
        print(f"{name:<50} {r['dims']:>4} {r['top1_accuracy']:>6.1%} {r['top3_accuracy']:>6.1%} "
              f"{r['mean_mrr']:>6.3f} {r['mean_gap']:>7.4f} {r['latency_p50_ms']:>5.0f}")
    
    # Per-category comparison
    print(f"\n{'─' * 80}")
    print("PER-CATEGORY Top-1 Accuracy:")
    print(f"{'Category':<15} " + " ".join(f"{r['model'].split('/')[-1][:20]:>20}" for r in all_results))
    print("-" * 80)
    
    all_cats = set()
    for r in all_results:
        all_cats.update(r['categories'].keys())
    
    for cat in sorted(all_cats):
        row = f"{cat:<15} "
        for r in all_results:
            if cat in r['categories']:
                v = r['categories'][cat]['top1']
                row += f"{v:>20.1%}"
            else:
                row += f"{'N/A':>20}"
        print(row)
    
    # Failures detail
    for r in all_results:
        if r['failures']:
            name = r['model'].split('/')[-1]
            print(f"\n{'─' * 80}")
            print(f"FAILURES ({name}): {len(r['failures'])}/{len(r['all_results'])}")
            for f in r['failures']:
                print(f"  Q: {f['query'][:70]}...")
                print(f"    Expected: {f['expected']} | Got: {f['predicted']}")
                print(f"    Top3: {list(zip(f['top3'], f['top3_scores']))}")
                print()


if __name__ == "__main__":
    print("Muninn Quality Benchmark v1.0")
    print("=" * 50)
    
    # Load facets
    facets_data = load_facets()
    print(f"Loaded {len(facets_data['peers'])} peers, {len(facets_data['facets'])} facets")
    print(f"Test queries: {len(TEST_QUERIES)}")
    
    # Models to test
    models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Alibaba-NLP/gte-multilingual-base",
        "BAAI/bge-m3",
    ]
    
    # Override from CLI
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    
    print(f"\nModels to evaluate: {len(models)}")
    for m in models:
        print(f"  - {m}")
    
    # Run evaluations
    all_results = []
    for model_name in models:
        print(f"\n{'═' * 60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'═' * 60}")
        try:
            result = evaluate_model(model_name, facets_data, TEST_QUERIES)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print report
    if all_results:
        print_report(all_results)
        
        # Save full results
        out_path = "/mnt/d/github/muninn/benchmark_quality_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nFull results saved to {out_path}")
