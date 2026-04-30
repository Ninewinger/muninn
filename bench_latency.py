#!/usr/bin/env python3
"""
Muninn Latency Benchmark — Local vs Online embedding backends.

Measures end-to-end latency for the critical prefetch path:
  embed(query) → cosine similarity → (optional rerank) → context injection

Usage:
  python bench_latency.py                     # All backends
  python bench_latency.py --backend local     # Only local models
  python bench_latency.py --backend cloud     # Only cloud APIs
  python bench_latency.py --queries 50        # More iterations
  python bench_latency.py --reranker          # Include reranker timing

Results guide the MUNINN_TIMEOUT and backend choice for the Hermes plugin.
"""

import argparse
import os
import sys
import time
import statistics
from pathlib import Path

# Add muninn to path
MUNNIN_ROOT = Path(__file__).parent
sys.path.insert(0, str(MUNNIN_ROOT.parent))

# ---------------------------------------------------------------------------
# Test queries (representative of real Hermes user messages)
# ---------------------------------------------------------------------------

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


def bench_embed_single(backend, query: str) -> float:
    """Time a single embedding call. Returns seconds."""
    start = time.perf_counter()
    backend.embed(query, is_query=True)
    return time.perf_counter() - start


def bench_embed_batch(backend, queries: list) -> float:
    """Time a batch embedding call. Returns seconds."""
    start = time.perf_counter()
    backend.embed_batch(queries, is_query=True)
    return time.perf_counter() - start


def bench_reranker(reranker, query: str, docs: list) -> float:
    """Time reranking. Returns seconds."""
    pairs = [(query, doc) for doc in docs]
    start = time.perf_counter()
    reranker.predict(pairs)
    return time.perf_counter() - start


def bench_full_pipeline(backend, query: str, reranker=None,
                        stored_embeddings: list = None,
                        stored_texts: list = None) -> dict:
    """Time the full prefetch pipeline: embed + search + (rerank).

    Returns dict with timing breakdown.
    """
    timings = {}

    # Step 1: Embed query
    t0 = time.perf_counter()
    query_vec = backend.embed(query, is_query=True)
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    # Step 2: Cosine similarity against stored embeddings
    if stored_embeddings:
        t0 = time.perf_counter()
        scores = []
        for stored in stored_embeddings:
            dot = sum(a * b for a, b in zip(query_vec, stored))
            scores.append(dot)
        timings["search_ms"] = (time.perf_counter() - t0) * 1000
    else:
        timings["search_ms"] = 0.0

    # Step 3: Rerank (optional)
    if reranker and stored_texts:
        top_pairs = [(query, stored_texts[i]) for i in
                     sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:5]]
        t0 = time.perf_counter()
        reranker.predict(top_pairs)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    else:
        timings["rerank_ms"] = 0.0

    timings["total_ms"] = sum([
        timings["embed_ms"],
        timings["search_ms"],
        timings["rerank_ms"],
    ])

    return timings


# ---------------------------------------------------------------------------
# Backend constructors
# ---------------------------------------------------------------------------

def get_local_minilm():
    """SentenceTransformer with MiniLM (fast, low quality)."""
    from muninn.embeddings_v2 import SentenceTransformerBackend
    print("  Loading MiniLM-L12-v2...")
    return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")


def get_local_qwen3():
    """Qwen3-Embedding via transformers (slow, high quality)."""
    from muninn.embeddings_v2 import Qwen3Backend
    print("  Loading Qwen3-Embedding-8B...")
    return Qwen3Backend("Qwen/Qwen3-Embedding-8B", dimensions=1024)


def get_cloud_openrouter():
    """OpenRouter API embedding (network dependent)."""
    from muninn.embeddings_v2 import OpenRouterBackend
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  SKIP: OPENROUTER_API_KEY not set")
        return None
    print("  Configuring OpenRouter backend...")
    return OpenRouterBackend(api_key=api_key)


def get_reranker():
    """Load cross-encoder reranker."""
    try:
        from sentence_transformers import CrossEncoder
        print("  Loading bge-reranker-v2-m3...")
        return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    except Exception as e:
        print(f"  SKIP reranker: {e}")
        return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_stats(times: list) -> str:
    """Format timing statistics."""
    if not times:
        return "  (no data)"
    ms = [t * 1000 for t in times]
    return (
        f"  p50={statistics.median(ms):.1f}ms  "
        f"p75={sorted(ms)[int(len(ms)*0.75)]:.1f}ms  "
        f"p99={sorted(ms)[int(len(ms)*0.99)]:.1f}ms  "
        f"avg={statistics.mean(ms):.1f}ms  "
        f"min={min(ms):.1f}ms  max={max(ms):.1f}ms  "
        f"(n={len(ms)})"
    )


def fmt_pipeline_stats(all_timings: list, key: str) -> str:
    """Format pipeline step statistics."""
    values = [t[key] for t in all_timings if key in t]
    if not values:
        return "  (no data)"
    return (
        f"  p50={statistics.median(values):.1f}ms  "
        f"avg={statistics.mean(values):.1f}ms  "
        f"max={max(values):.1f}ms"
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(backend_name: str, backend, queries: list,
                  reranker=None, n: int = 20):
    """Run all benchmarks for a single backend."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {backend_name}")
    print(f"  Dimensions: {backend.dimensions}")
    print(f"  Queries: {len(queries)} | Iterations: {n}")
    print(f"{'='*60}")

    # -- Generate some fake stored embeddings for pipeline test --
    print("\n  Generating stored embeddings for pipeline test...")
    fake_docs = [
        "El usuario entrena pesas regularmente y le gusta el gym",
        "Prefiere aprendizaje práctico sobre teórico",
        "Le interesa la filosofía oriental y la psicología junguiana",
        "Está trabajando en un juego con mecánicas de exploración",
        "Tiene un proyecto de aprendizaje de programación activo",
    ]
    stored_embeddings = backend.embed_batch(fake_docs)

    # ── Test 1: Single embed latency ──────────────────────────────────────
    print(f"\n  [1] Single embed latency (per query):")
    single_times = []
    # Warmup
    for q in queries[:2]:
        backend.embed(q, is_query=True)
    # Measure
    for _ in range(n):
        for q in queries:
            t = bench_embed_single(backend, q)
            single_times.append(t)
    print(fmt_stats(single_times))

    # ── Test 2: Batch embed latency ───────────────────────────────────────
    print(f"\n  [2] Batch embed latency ({len(queries)} queries):")
    batch_times = []
    for _ in range(n):
        t = bench_embed_batch(backend, queries)
        batch_times.append(t)
    print(fmt_stats(batch_times))

    # ── Test 3: Full pipeline latency ─────────────────────────────────────
    print(f"\n  [3] Full prefetch pipeline (embed + search{'+ rerank' if reranker else ''}):")
    pipeline_timings = []
    for _ in range(n):
        for q in queries:
            t = bench_full_pipeline(
                backend, q, reranker, stored_embeddings, fake_docs
            )
            pipeline_timings.append(t)

    print(f"\n  Embed step:")
    print(fmt_pipeline_stats(pipeline_timings, "embed_ms"))
    print(f"\n  Search step (cosine against 5 stored):")
    print(fmt_pipeline_stats(pipeline_timings, "search_ms"))
    if reranker:
        print(f"\n  Rerank step (top-5 cross-encoder):")
        print(fmt_pipeline_stats(pipeline_timings, "rerank_ms"))
    print(f"\n  TOTAL pipeline:")
    print(fmt_pipeline_stats(pipeline_timings, "total_ms"))

    # ── Summary ───────────────────────────────────────────────────────────
    totals = [t["total_ms"] for t in pipeline_timings]
    p50 = statistics.median(totals)

    print(f"\n  {'─'*50}")
    if p50 < 100:
        verdict = "✅ FAST — suitable for prefetch (under 100ms)"
    elif p50 < 500:
        verdict = "⚠️  MODERATE — acceptable with queue_prefetch background"
    elif p50 < 1500:
        verdict = "🟡 SLOW — must use background prefetch, consider online backend"
    else:
        verdict = "❌ TOO SLOW — use online backend or lighter model"
    print(f"  Verdict: {verdict}")
    print(f"  Recommended MUNINN_TIMEOUT: {int(p50 * 3 / 1000) + 2}s")
    print(f"  {'─'*50}")

    return {
        "backend": backend_name,
        "p50_ms": p50,
        "avg_ms": statistics.mean(totals),
        "max_ms": max(totals),
    }


def main():
    parser = argparse.ArgumentParser(description="Muninn embedding latency benchmark")
    parser.add_argument("--backend", choices=["local", "cloud", "all"], default="all",
                        help="Which backends to test")
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of iterations per benchmark")
    parser.add_argument("--reranker", action="store_true",
                        help="Include reranker in pipeline test")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Muninn Latency Benchmark                               ║")
    print("║  Measures embedding + routing latency for prefetch()    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = []

    # -- Local backends --
    if args.backend in ("local", "all"):
        try:
            backend = get_local_minilm()
            r = run_benchmark("MiniLM-L12-v2 (local CPU)", backend,
                              QUERIES, n=args.queries,
                              reranker=get_reranker() if args.reranker else None)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR loading MiniLM: {e}")

        try:
            backend = get_local_qwen3()
            r = run_benchmark("Qwen3-Embedding-8B (local GPU)", backend,
                              QUERIES, n=args.queries,
                              reranker=get_reranker() if args.reranker else None)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR loading Qwen3: {e}")

    # -- Cloud backends --
    if args.backend in ("cloud", "all"):
        backend = get_cloud_openrouter()
        if backend:
            r = run_benchmark("OpenRouter text-embedding-3-small (cloud)", backend,
                              QUERIES, n=args.queries)
            results.append(r)

    # -- Comparison table --
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"  {'Backend':<40} {'p50':>8} {'avg':>8} {'max':>8}")
        print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8}")
        for r in sorted(results, key=lambda x: x["p50_ms"]):
            print(f"  {r['backend']:<40} {r['p50_ms']:>7.1f}ms {r['avg_ms']:>7.1f}ms {r['max_ms']:>7.1f}ms")
        print()


if __name__ == "__main__":
    main()
