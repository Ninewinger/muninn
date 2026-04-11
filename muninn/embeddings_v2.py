"""Embedding utilities for Muninn v0.2 — Multi-backend support.

Supports:
  - sentence-transformers (local, MiniLM, etc.)
  - Qwen3-Embedding via transformers (local, GPU)
  - OpenRouter API (cloud)
  - Google Gemini API (cloud)
"""

import os
from typing import List, Optional
from abc import ABC, abstractmethod

# Default model configuration
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_DIMENSIONS = 1024
DEFAULT_INSTRUCTION = "Dado un mensaje de un usuario, identifica que dominio de su vida esta activando"

# Singleton model cache
_backend = None


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, text: str, is_query: bool = False, instruction: str = None) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], is_query: bool = False, instruction: str = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """Backend using sentence-transformers library (MiniLM, etc.)."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dims = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str, is_query: bool = False, instruction: str = None) -> List[float]:
        result = self._model.encode(text, normalize_embeddings=True)
        return result.tolist()

    def embed_batch(self, texts: List[str], is_query: bool = False, instruction: str = None) -> List[List[float]]:
        results = self._model.encode(texts, normalize_embeddings=True)
        return results.tolist()

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name


class Qwen3Backend(EmbeddingBackend):
    """Backend using Qwen3-Embedding via transformers (GPU/CPU)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B",
                 dimensions: int = 1024, instruction: str = None):
        import torch
        import torch.nn.functional as F
        from torch import Tensor
        from transformers import AutoTokenizer, AutoModel

        self._torch = torch
        self._F = F
        self._model_name = model_name
        self._dims = dimensions
        self._instruction = instruction or DEFAULT_INSTRUCTION

        print(f"  [Qwen3] Cargando {model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side='left'
        )
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float16
        )
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            print(f"  [Qwen3] GPU: {torch.cuda.get_device_name(0)}")
        self._model.eval()
        print(f"  [Qwen3] Listo ({dimensions}d)")

    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[self._torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _encode_raw(self, texts: List[str], is_query: bool = False,
                    instruction: str = None, dim: int = -1) -> list:
        """Encode texts, return numpy arrays."""
        import numpy as np

        if is_query:
            instr = instruction or self._instruction
            texts = [f'Instruct: {instr}\nQuery:{t}' for t in texts]

        inputs = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=8192, return_tensors='pt'
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            pooled = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            if dim != -1:
                pooled = pooled[:, :dim]
            pooled = self._F.normalize(pooled, p=2, dim=1)
            return pooled.cpu().float().numpy()

    def embed(self, text: str, is_query: bool = False, instruction: str = None) -> List[float]:
        result = self._encode_raw([text], is_query, instruction, dim=self._dims)
        return result[0].tolist()

    def embed_batch(self, texts: List[str], is_query: bool = False,
                    instruction: str = None, batch_size: int = 32) -> List[List[float]]:
        import numpy as np
        all_outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self._encode_raw(batch, is_query, instruction, dim=self._dims)
            all_outputs.append(result)
        combined = np.vstack(all_outputs)
        return combined.tolist()

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name


class OpenRouterBackend(EmbeddingBackend):
    """Backend using OpenRouter API (cloud)."""

    def __init__(self, model_name: str = "openai/text-embedding-3-small",
                 api_key: str = None, dimensions: int = 1536):
        self._model_name = model_name
        self._dims = dimensions
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    def embed(self, text: str, is_query: bool = False, instruction: str = None) -> List[float]:
        return self.embed_batch([text], is_query, instruction)[0]

    def embed_batch(self, texts: List[str], is_query: bool = False, instruction: str = None) -> List[List[float]]:
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_name,
                "input": texts,
            }
        )
        data = response.json()
        return [d["embedding"] for d in data["data"]]

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name


# ── Factory ────────────────────────────────────────────────

def get_backend(model_name: str = None, **kwargs) -> EmbeddingBackend:
    """Get or create the configured embedding backend."""
    global _backend
    model = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

    # Return cached if same model
    if _backend is not None and _backend.model_name == model:
        return _backend

    if "qwen3" in model.lower() or "Qwen3" in model:
        dims = kwargs.get("dimensions", DEFAULT_DIMENSIONS)
        _backend = Qwen3Backend(model, dimensions=dims)
    elif "openrouter" in model.lower() or model.startswith("openai/"):
        _backend = OpenRouterBackend(model)
    else:
        # Default: sentence-transformers
        _backend = SentenceTransformerBackend(model)

    return _backend


def embed(text: str, is_query: bool = False, instruction: str = None) -> List[float]:
    """Generate embedding for a single text using configured backend."""
    backend = get_backend()
    return backend.embed(text, is_query, instruction)


def embed_batch(texts: List[str], is_query: bool = False, instruction: str = None) -> List[List[float]]:
    """Generate embeddings for multiple texts using configured backend."""
    backend = get_backend()
    return backend.embed_batch(texts, is_query, instruction)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot
