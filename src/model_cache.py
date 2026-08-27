"""
Process-wide model cache.

The embedding model and cross-encoder reranker are expensive to load (~tens of
seconds each, cold). The pipeline previously loaded the embedding model 2-3
times per run (embedder + retriever) and the reranker once. These helpers
return cached singletons so each model is loaded at most once per process,
which persists across Streamlit reruns.

Loading the same weights once vs. many times is output-identical — this is a
pure speedup, not a behavior change.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def get_embedding_model(model_name: str):
    """Return a cached SentenceTransformer for `model_name`."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    print(f"[model_cache] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=None)
def get_reranker_model(model_name: str):
    """Return a cached CrossEncoder for `model_name`."""
    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    print(f"[model_cache] Loading reranker: {model_name}")
    return CrossEncoder(model_name)
