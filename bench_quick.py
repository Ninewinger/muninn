#!/usr/bin/env python3
"""Quick Muninn latency benchmark — only cached models."""

import sys
import time
import statistics
import os

sys.path.insert(0, "/mnt/d/github/muninn")

QUERIES = [
    "me duele mucho el brazo después de entrenar ayer",
    "tengo una entrevista de trabajo mañana y estoy nervioso",
    "quiero aprender más sobre decoradores en python",
    "soñé que volaba sobre la ciudad de noche",
    "no sé si tomar el suplemento de creatina o no",
    "mi amigo me prestó un libro de filosofía budista",
    "el proyecto del juego avanza bien con las mecánicas de exploración",
    "estoy pensando en mudarme a otra ciudad por trabajo",
    "vi un episodio de Adventure Time que me hizo pensar en la sombra",
    "quiero organizar mi vault de Obsidian mejor",
]

FAKE_DOCS = [
    "El usuario entrena pesas regularmente y le gusta el gym",
    "Prefiere aprendizaje práctico sobre teórico",
    "Le interesa la filosofía oriental y la psicología junguiana",
    "Está trabajando en un juego con mecánicas de exploración",
    "Tiene un proyecto de aprendizaje de programación activo",
]


def bench_backend(name, backend, n=10):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Dimensions: {backend.dimensions}")
    print(f"{'='*55}")

    # Warmup
    for q in QUERIES[:2]:
        backend.embed(q, is_query=True)
    
    # Store embeddings for pipeline
    stored = backend.embed_batch(FAKE_DOCS)

    # Single embed
    single_ms = []
    for _ in range(n):
        for q in QUERIES:
            t0 = time.perf_counter()
            qvec = backend.embed(q, is_query=True)
            single_ms.append((time.perf_counter() - t0) * 1000)

    # Full pipeline: embed + cosine search
    pipeline_ms = []
    for _ in range(n):
        for q in QUERIES:
            t0 = time.perf_counter()
            qvec = backend.embed(q, is_query=True)
            for svec in stored:
                dot = sum(a * b for a, b in zip(qvec, svec))
            pipeline_ms.append((time.perf_counter() - t0) * 1000)

    print(f"\n  Single embed:")
    print(f"    p50={statistics.median(single_ms):.1f}ms  avg={statistics.mean(single_ms):.1f}ms  max={max(single_ms):.1f}ms")
    print(f"\n  Full pipeline (embed + cosine x5):")
    print(f"    p50={statistics.median(pipeline_ms):.1f}ms  avg={statistics.mean(pipeline_ms):.1f}ms  max={max(pipeline_ms):.1f}ms")

    p50 = statistics.median(pipeline_ms)
    if p50 < 100:
        verdict = "✅ FAST"
    elif p50 < 500:
        verdict = "⚠️ MODERATE"
    else:
        verdict = "❌ SLOW"
    print(f"\n  Verdict: {verdict} (p50={p50:.1f}ms)")
    return {"name": name, "p50": p50, "avg": statistics.mean(pipeline_ms), "max": max(pipeline_ms)}


def bench_reranker(n=10):
    """Benchmark reranker separately."""
    print(f"\n{'='*55}")
    print(f"  bge-reranker-v2-m3 (cross-encoder)")
    print(f"{'='*55}")

    try:
        from sentence_transformers import CrossEncoder
        print("  Loading reranker...")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    except Exception as e:
        print(f"  SKIP: {e}")
        return None

    # Warmup
    reranker.predict([(QUERIES[0], FAKE_DOCS[0])])

    rerank_ms = []
    for _ in range(n):
        for q in QUERIES:
            pairs = [(q, doc) for doc in FAKE_DOCS]
            t0 = time.perf_counter()
            reranker.predict(pairs)
            rerank_ms.append((time.perf_counter() - t0) * 1000)

    print(f"\n  Rerank 5 pairs:")
    print(f"    p50={statistics.median(rerank_ms):.1f}ms  avg={statistics.mean(rerank_ms):.1f}ms  max={max(rerank_ms):.1f}ms")
    return {"name": "reranker", "p50": statistics.median(rerank_ms), "avg": statistics.mean(rerank_ms), "max": max(rerank_ms)}


if __name__ == "__main__":
    results = []

    # ── MiniLM (cached) ──
    print("Loading MiniLM-L12-v2...")
    from muninn.embeddings_v2 import SentenceTransformerBackend
    minilm = SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")
    results.append(bench_backend("MiniLM-L12-v2 (local CPU)", minilm, n=10))

    # ── Reranker (cached) ──
    r = bench_reranker(n=10)
    if r:
        results.append(r)

    # ── Qwen3 (may need download) ──
    try:
        print("\nLoading Qwen3-Embedding-8B (this may take a while if not cached)...")
        from muninn.embeddings_v2 import Qwen3Backend
        qwen3 = Qwen3Backend("Qwen/Qwen3-Embedding-8B", dimensions=1024)
        results.append(bench_backend("Qwen3-Embedding-8B (local GPU)", qwen3, n=5))
    except Exception as e:
        print(f"\n  SKIP Qwen3: {e}")

    # ── Summary ──
    if results:
        print(f"\n{'='*55}")
        print("  SUMMARY")
        print(f"{'='*55}")
        print(f"  {'Component':<40} {'p50':>8} {'avg':>8} {'max':>8}")
        print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8}")
        for r in sorted(results, key=lambda x: x["p50"]):
            print(f"  {r['name']:<40} {r['p50']:>7.1f}ms {r['avg']:>7.1f}ms {r['max']:>7.1f}ms")

        # Hermes recommendation
        embed_p50 = [r for r in results if "reranker" not in r["name"]]
        rerank_p50 = [r for r in results if "reranker" in r["name"]]
        total = embed_p50[0]["p50"] if embed_p50 else 0
        total += rerank_p50[0]["p50"] if rerank_p50 else 0
        print(f"\n  Estimated prefetch latency (embed + search + rerank): ~{total:.0f}ms")
        if total < 200:
            print("  → ✅ Great for real-time prefetch()")
        elif total < 800:
            print("  → ⚠️ Use queue_prefetch() (background) to avoid blocking")
        else:
            print("  → ❌ Consider online embeddings or lighter model")
