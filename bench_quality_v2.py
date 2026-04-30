#!/usr/bin/env python3
"""
Muninn Quality Benchmark v2.0
==============================
Tests different STRATEGIES for matching queries to peers, not just models.

Strategies:
  A. Baseline (facet text only, max sim per peer)
  B. Query instruction prefix + facet text
  C. Peer composite: description + representation + tags + facets
  D. Combined: instruction prefix + composite peer representation
"""

import json
import time
import statistics
import os
import sys

# Re-use test queries from v1
sys.path.insert(0, os.path.dirname(__file__))
from bench_quality import TEST_QUERIES, cosine_sim, load_facets

QUERY_INSTRUCTION = "Identifica qué aspecto de la vida personal se activa con este mensaje: "


def encode_with_model(model, texts, normalize=True):
    """Encode texts using a SentenceTransformer model."""
    return model.encode(texts, normalize_embeddings=normalize).tolist()


def strategy_baseline(model, facets_data, queries):
    """Strategy A: Raw facet text matching (original approach)."""
    facet_texts = [f['text'] for f in facets_data['facets']]
    facet_peer_ids = [f['peer_id'] for f in facets_data['facets']]
    
    facet_embs = encode_with_model(model, facet_texts)
    
    results = []
    for tq in queries:
        query_emb = encode_with_model(model, [tq['query']])[0]
        
        peer_scores = {}
        for emb, pid in zip(facet_embs, facet_peer_ids):
            sim = cosine_sim(query_emb, emb)
            if pid not in peer_scores or sim > peer_scores[pid]:
                peer_scores[pid] = sim
        
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        results.append({
            'query': tq['query'],
            'expected': set(tq['expected_peers']),
            'ranked': ranked,
            'category': tq['category'],
        })
    return results


def strategy_instruction(model, facets_data, queries):
    """Strategy B: Add instruction prefix to queries."""
    facet_texts = [f['text'] for f in facets_data['facets']]
    facet_peer_ids = [f['peer_id'] for f in facets_data['facets']]
    
    facet_embs = encode_with_model(model, facet_texts)
    
    results = []
    for tq in queries:
        instructed_query = QUERY_INSTRUCTION + tq['query']
        query_emb = encode_with_model(model, [instructed_query])[0]
        
        peer_scores = {}
        for emb, pid in zip(facet_embs, facet_peer_ids):
            sim = cosine_sim(query_emb, emb)
            if pid not in peer_scores or sim > peer_scores[pid]:
                peer_scores[pid] = sim
        
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        results.append({
            'query': tq['query'],
            'expected': set(tq['expected_peers']),
            'ranked': ranked,
            'category': tq['category'],
        })
    return results


def strategy_composite(model, facets_data, queries):
    """Strategy C: Create composite peer representations."""
    peers = facets_data['peers']
    facets = facets_data['facets']
    
    # Build composite text per peer
    peer_texts = {}
    for pid, p in peers.items():
        parts = [f"{p['name']}. {p['description']}. {p['representation']}."]
        if p.get('tags'):
            tags = p['tags'].strip('[]').replace('"', '').replace("'", '')
            parts.append(f"Palabras clave: {tags}")
        # Add all facets
        peer_facets = [f['text'] for f in facets if f['peer_id'] == pid]
        if peer_facets:
            parts.append("Ejemplos: " + ". ".join(peer_facets))
        peer_texts[pid] = " ".join(parts)
    
    peer_ids = list(peer_texts.keys())
    peer_embs = encode_with_model(model, [peer_texts[pid] for pid in peer_ids])
    
    results = []
    for tq in queries:
        query_emb = encode_with_model(model, [tq['query']])[0]
        
        peer_scores = {}
        for emb, pid in zip(peer_embs, peer_ids):
            peer_scores[pid] = cosine_sim(query_emb, emb)
        
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        results.append({
            'query': tq['query'],
            'expected': set(tq['expected_peers']),
            'ranked': ranked,
            'category': tq['category'],
        })
    return results


def strategy_combined(model, facets_data, queries):
    """Strategy D: Instruction prefix + composite peer representation."""
    peers = facets_data['peers']
    facets = facets_data['facets']
    
    # Build composite text per peer  
    peer_texts = {}
    for pid, p in peers.items():
        parts = [f"{p['name']}. {p['description']}. {p['representation']}."]
        if p.get('tags'):
            tags = p['tags'].strip('[]').replace('"', '').replace("'", '')
            parts.append(f"Palabras clave: {tags}")
        peer_facets = [f['text'] for f in facets if f['peer_id'] == pid]
        if peer_facets:
            parts.append("Ejemplos: " + ". ".join(peer_facets))
        peer_texts[pid] = " ".join(parts)
    
    peer_ids = list(peer_texts.keys())
    peer_embs = encode_with_model(model, [peer_texts[pid] for pid in peer_ids])
    
    results = []
    for tq in queries:
        instructed_query = QUERY_INSTRUCTION + tq['query']
        query_emb = encode_with_model(model, [instructed_query])[0]
        
        peer_scores = {}
        for emb, pid in zip(peer_embs, peer_ids):
            peer_scores[pid] = cosine_sim(query_emb, emb)
        
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        results.append({
            'query': tq['query'],
            'expected': set(tq['expected_peers']),
            'ranked': ranked,
            'category': tq['category'],
        })
    return results


def strategy_facet_avg(model, facets_data, queries):
    """Strategy E: Average similarity across all facets per peer."""
    facet_texts = [f['text'] for f in facets_data['facets']]
    facet_peer_ids = [f['peer_id'] for f in facets_data['facets']]
    
    facet_embs = encode_with_model(model, facet_texts)
    
    # Group facets by peer
    from collections import defaultdict
    peer_facet_embs = defaultdict(list)
    for emb, pid in zip(facet_embs, facet_peer_ids):
        peer_facet_embs[pid].append(emb)
    
    results = []
    for tq in queries:
        query_emb = encode_with_model(model, [tq['query']])[0]
        
        peer_scores = {}
        for pid, embs in peer_facet_embs.items():
            scores = [cosine_sim(query_emb, e) for e in embs]
            peer_scores[pid] = sum(scores) / len(scores)  # Average instead of max
        
        ranked = sorted(peer_scores.items(), key=lambda x: x[1], reverse=True)
        results.append({
            'query': tq['query'],
            'expected': set(tq['expected_peers']),
            'ranked': ranked,
            'category': tq['category'],
        })
    return results


def compute_metrics(results, peer_set=None):
    """Compute accuracy metrics from results."""
    top1_hits = 0
    top3_hits = 0
    mrrs = []
    gaps = []
    
    for r in results:
        expected = r['expected']
        ranked = r['ranked']
        ranked_ids = [x[0] for x in ranked]
        ranked_scores = [x[1] for x in ranked]
        
        top1_hits += ranked_ids[0] in expected
        top3_hits += any(p in expected for p in ranked_ids[:3])
        
        mrr = 0.0
        for i, pid in enumerate(ranked_ids):
            if pid in expected:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        
        best_correct = 0.0
        best_incorrect = 0.0
        for pid, score in ranked:
            if pid in expected and score > best_correct:
                best_correct = score
            elif pid not in expected and score > best_incorrect:
                best_incorrect = score
        gaps.append(best_correct - best_incorrect)
    
    n = len(results)
    return {
        'top1': round(top1_hits / n, 3),
        'top3': round(top3_hits / n, 3),
        'mrr': round(statistics.mean(mrrs), 3),
        'gap': round(statistics.mean(gaps), 4),
    }


def main():
    print("Muninn Quality Benchmark v2.0 — Strategy Comparison")
    print("=" * 60)
    
    facets_data = load_facets()
    print(f"Peers: {len(facets_data['peers'])}, Facets: {len(facets_data['facets'])}")
    print(f"Test queries: {len(TEST_QUERIES)}")
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    print(f"\nLoading {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    
    strategies = [
        ("A: Baseline (facet max-sim)", strategy_baseline),
        ("B: Query instruction", strategy_instruction),
        ("C: Composite peer", strategy_composite),
        ("D: Instruction + Composite", strategy_combined),
        ("E: Facet avg-sim", strategy_facet_avg),
    ]
    
    all_results = {}
    print(f"\n{'Strategy':<35} {'Top-1':>6} {'Top-3':>6} {'MRR':>6} {'Gap':>7}")
    print("-" * 65)
    
    for name, strategy_fn in strategies:
        print(f"  Running {name}...", end="", flush=True)
        t0 = time.time()
        results = strategy_fn(model, facets_data, TEST_QUERIES)
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        all_results[name] = {'metrics': metrics, 'results': results}
        print(f" {elapsed:.1f}s")
        print(f"{name:<35} {metrics['top1']:>6.1%} {metrics['top3']:>6.1%} {metrics['mrr']:>6.3f} {metrics['gap']:>7.4f}")
    
    # Per-category comparison for best strategy vs baseline
    print(f"\n{'─' * 65}")
    print("PER-CATEGORY Top-1:")
    
    all_cats = set()
    for name in all_results:
        for r in all_results[name]['results']:
            all_cats.add(r['category'])
    
    print(f"{'Category':<15} " + " ".join(f"{n.split(':')[0]:>10}" for n in all_results))
    print("-" * 65)
    for cat in sorted(all_cats):
        row = f"{cat:<15}"
        for name, data in all_results.items():
            cat_results = [r for r in data['results'] if r['category'] == cat]
            if cat_results:
                n = len(cat_results)
                hits = sum(1 for r in cat_results if r['ranked'][0][0] in r['expected'])
                row += f"{hits/n:>10.1%}"
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    # Find worst failures across all strategies
    print(f"\n{'─' * 65}")
    print("CONSISTENT FAILURES (fail in ALL strategies):")
    
    consistent = []
    for i, tq in enumerate(TEST_QUERIES):
        fails = 0
        for name, data in all_results.items():
            if data['results'][i]['ranked'][0][0] not in data['results'][i]['expected']:
                fails += 1
        if fails == len(all_results):
            consistent.append(tq)
            best_ranked = all_results['A: Baseline (facet max-sim)']['results'][i]['ranked'][0]
            print(f"  [{tq['category']}] {tq['query'][:70]}...")
            print(f"    Expected: {tq['expected_peers']} | Got: {best_ranked[0]} ({best_ranked[1]:.3f})")
    
    print(f"\n  Total consistent failures: {len(consistent)}/{len(TEST_QUERIES)}")
    
    # Save
    out = {}
    for name, data in all_results.items():
        out[name] = {
            'metrics': data['metrics'],
            'per_query': [{
                'query': r['query'],
                'expected': list(r['expected']),
                'predicted': r['ranked'][0][0],
                'top3': [(p, round(s, 4)) for p, s in r['ranked'][:3]],
                'category': r['category'],
            } for r in data['results']]
        }
    
    with open('/mnt/d/github/muninn/benchmark_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to benchmark_v2_results.json")


if __name__ == "__main__":
    main()
