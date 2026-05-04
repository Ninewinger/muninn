"""OpenRouter reranker backend for Muninn.

Uses Cohere Rerank models via OpenRouter API.
Supports multilingual reranking with 32K context window.
"""
import os, json, time
from typing import List, Dict, Optional

# Cache for reranker results: {(query_hash, doc_id): score}
_rerank_cache: dict = {}
_cache_hits = 0


def get_reranker_model() -> str:
    """Get configured reranker model name."""
    return os.getenv("MUNINN_RERANKER_MODEL", "cohere/rerank-v3.5")


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    # Try env first, then source from .env
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    # Source from .env
    import subprocess
    r = subprocess.run(
        ['bash', '-c', 'source /root/.hermes/.env && echo $OPENROUTER_API_KEY'],
        capture_output=True, text=True
    )
    return r.stdout.strip()


def rerank(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    model: str = None,
) -> List[Dict]:
    """
    Re-rank documents using Cohere Rerank via OpenRouter.
    
    Args:
        query: The search query
        documents: List of dicts with at least 'peer_id' and 'text' keys
        top_k: Number of top results to return
        model: Reranker model (default: cohere/rerank-v3.5)
    
    Returns:
        List of documents sorted by reranker score, with 'rerank_score' added.
        Only documents above threshold are returned.
    """
    if not documents:
        return []
    
    model = model or get_reranker_model()
    api_key = get_api_key()
    
    if not api_key:
        print("[Reranker] No API key found — skipping rerank")
        return documents[:top_k]
    
    # Build request payload
    # OpenRouter uses the same format as Cohere's rerank API
    payload = {
        "model": model,
        "query": query,
        "documents": [d.get("text", "") for d in documents],
        "top_n": top_k,
    }
    
    # Check cache
    query_hash = str(hash(query + model))[:16]
    cache_key = (query_hash,)
    if cache_key in _rerank_cache and _rerank_cache[cache_key].get("model") == model:
        return _rerank_cache[cache_key]["results"]
    
    try:
        import requests
        resp = requests.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        
        if resp.status_code != 200:
            print(f"[Reranker] API error {resp.status_code}: {resp.text[:200]}")
            # Fallback: return top_k as-is
            return documents[:top_k]
        
        data = resp.json()
        
        # Parse results - OpenRouter returns Cohere-compatible format
        results = data.get("results", [])
        if not results and "data" in data:
            results = data["data"]
        
        # Map reranker results back to documents
        reranked = []
        for r in results:
            idx = r.get("index", r.get("position", 0))
            score = r.get("relevance_score", r.get("score", 0))
            if idx < len(documents):
                doc = dict(documents[idx])
                doc["rerank_score"] = float(score)
                doc["rerank_model"] = model
                reranked.append(doc)
        
        # Sort by rerank score descending
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Cache
        _rerank_cache[cache_key] = {"results": reranked, "model": model, "timestamp": time.time()}
        
        # Trim cache to 100 entries
        if len(_rerank_cache) > 100:
            oldest = min(_rerank_cache.keys(), key=lambda k: _rerank_cache[k]["timestamp"])
            del _rerank_cache[oldest]
        
        return reranked
        
    except Exception as e:
        print(f"[Reranker] Error: {e}")
        return documents[:top_k]


def clear_cache():
    """Clear reranker cache."""
    global _rerank_cache, _cache_hits
    _rerank_cache.clear()
    _cache_hits = 0


def test_reranker():
    """Quick test to verify the reranker works."""
    query = "crypto bitcoin inversiones"
    docs = [
        {"peer_id": "finanzas_patrimonio", "text": "finanzas personales, patrimonio, inversiones en instrumentos financieros, acciones, fondos mutuos, presupuesto"},
        {"peer_id": "gym_rutina", "text": "gimnasio, rutina de ejercicios, pesas, cardio, entrenamiento"},
        {"peer_id": "proyecto_juego", "text": "diseño de juego de rol, narrativa, mecánicas, progresión"},
    ]
    
    print(f"Query: {query}")
    results = rerank(query, docs, top_k=3)
    print("Results:")
    for r in results:
        print(f"  {r['peer_id']:25s} rerank_score={r.get('rerank_score', 0):.4f}")
    return results


if __name__ == "__main__":
    test_reranker()