"""Embedding utilities for Muninn."""

import os
from typing import List

# Default model configuration
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSIONS = 384


def get_embedding_model():
    """Get the sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )


def embed(text: str, model=None) -> List[float]:
    """Generate embedding for a single text."""
    if model is None:
        model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: List[str], model=None) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    if model is None:
        model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Assumes vectors are already normalized
    dot = sum(x * y for x, y in zip(a, b))
    return dot
