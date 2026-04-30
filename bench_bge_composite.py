#!/usr/bin/env python3
"""Quick BGE-M3 composite test."""
from sentence_transformers import SentenceTransformer
import json, time, statistics
from bench_quality import TEST_QUERIES, cosine_sim, load_facets

facets_data = load_facets()
print("Loading BAAI/bge-m3...")
model = SentenceTransformer('BAAI/bge-m3')

peers = facets_data['peers']
facets = facets_data['facets']
peer_texts = {}
for pid, p in peers.items():
    parts = [p['name'] + ". " + p['description'] + ". " + p['representation'] + "."]
    if p.get('tags'):
        tags = p['tags'].strip('[]').replace('"', '').replace("'", '')
        parts.append('Palabras clave: ' + tags)
    pf = [f['text'] for f in facets if f['peer_id'] == pid]
    if pf:
        parts.append('Ejemplos: ' + '. '.join(pf))
    peer_texts[pid] = ' '.join(parts)

peer_ids = list(peer_texts.keys())
peer_embs = model.encode([peer_texts[pid] for pid in peer_ids], normalize_embeddings=True).tolist()

top1 = 0; top3 = 0; mrrs = []; gaps = []
failures = []
for tq in TEST_QUERIES:
    qe = model.encode([tq['query']], normalize_embeddings=True).tolist()[0]
    expected = set(tq['expected_peers'])
    scores = {pid: cosine_sim(qe, emb) for pid, emb in zip(peer_ids, peer_embs)}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rids = [r[0] for r in ranked]
    
    hit1 = rids[0] in expected
    top1 += hit1
    top3 += any(p in expected for p in rids[:3])
    for i, pid in enumerate(rids):
        if pid in expected:
            mrrs.append(1.0/(i+1)); break
    else:
        mrrs.append(0)
    bc = max((s for p, s in ranked if p in expected), default=0)
    bi = max((s for p, s in ranked if p not in expected), default=0)
    gaps.append(bc - bi)
    
    if not hit1:
        failures.append(f"  {tq['query'][:60]}... -> {rids[0]} (expected {list(expected)})")

n = len(TEST_QUERIES)
print(f"\nBGE-M3 + Composite: Top-1={top1/n:.1%}  Top-3={top3/n:.1%}  MRR={statistics.mean(mrrs):.3f}  Gap={statistics.mean(gaps):.4f}")
print(f"\nFailures ({n - top1}/{n}):")
for f in failures:
    print(f)
