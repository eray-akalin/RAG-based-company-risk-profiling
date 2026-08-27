"""
Embedding Generator & FAISS Indexer

Generates embeddings using BAAI/bge-small-en-v1.5 and builds
a FAISS index for semantic evidence retrieval.
"""

from __future__ import annotations

import os
import json
import pickle
import numpy as np
from typing import Any
import faiss  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CHUNKS_DIR, EMBEDDINGS_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    get_chunks_dir, get_embeddings_dir,
)


def load_chunks(chunks_path: str = None) -> list[dict]:
    """Load all chunks from JSON file."""
    if chunks_path is None:
        chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list[dict], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for all chunks using the configured model.
    Uses BAAI/bge-small-en-v1.5 which requires the "Represent this sentence: " prefix
    for queries but not for documents.

    Args:
        chunks: List of chunk dicts with 'text' field
        batch_size: Number of chunks to embed at once

    Returns:
        numpy array of shape (n_chunks, embedding_dim)
    """
    from src.model_cache import get_embedding_model
    model = get_embedding_model(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")

    # bge models work best without prefix for documents
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """
    Build a FAISS index for inner product (cosine similarity since vectors are normalized).

    Args:
        embeddings: numpy array of shape (n, dim)

    Returns:
        FAISS index
    """
    dim = embeddings.shape[1]
    print(f"Building FAISS index: {embeddings.shape[0]} vectors × {dim} dimensions")

    # Use IndexFlatIP (Inner Product) since embeddings are L2-normalized
    # This is equivalent to cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index(
    index: Any,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> dict:
    """
    Save the FAISS index, chunk metadata mapping, and raw embeddings.

    Returns:
        Dict with file paths
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved: {index_path}")

    # Save chunk ID → metadata mapping (without full text to save space)
    id_to_meta = {}
    for i, chunk in enumerate(chunks):
        id_to_meta[i] = {
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "company_name": chunk["company_name"],
            "year": chunk["year"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],  # Keep text for retrieval display
        }

    meta_path = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(id_to_meta, f)
    print(f"Chunk metadata saved: {meta_path}")

    # Save raw embeddings (useful for debugging and evaluation)
    emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Embeddings saved: {emb_path}")

    # Save index metadata
    index_meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_vectors": len(chunks),
        "index_type": "IndexFlatIP",
        "normalized": True,
    }
    index_meta_path = os.path.join(EMBEDDINGS_DIR, "index_metadata.json")
    with open(index_meta_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    return {
        "index_path": index_path,
        "meta_path": meta_path,
        "emb_path": emb_path,
    }


def save_index_for_year(
    index: Any,
    chunks: list[dict],
    embeddings: np.ndarray,
    year: int,
) -> dict:
    """Save FAISS index and metadata to a year-specific directory."""
    year_emb_dir = get_embeddings_dir(year)
    os.makedirs(year_emb_dir, exist_ok=True)

    index_path = os.path.join(year_emb_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)

    id_to_meta = {}
    for i, chunk in enumerate(chunks):
        id_to_meta[i] = {
            "chunk_id": chunk["chunk_id"],
            "company": chunk["company"],
            "company_name": chunk["company_name"],
            "year": chunk["year"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        }

    meta_path = os.path.join(year_emb_dir, "chunk_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(id_to_meta, f)

    emb_path = os.path.join(year_emb_dir, "embeddings.npy")
    np.save(emb_path, embeddings)

    index_meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_vectors": len(chunks),
        "index_type": "IndexFlatIP",
        "normalized": True,
        "fiscal_year": year,
    }
    index_meta_path = os.path.join(year_emb_dir, "index_metadata.json")
    with open(index_meta_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"Saved year-specific index for FY{year}: {index.ntotal} vectors")
    return {"index_path": index_path, "meta_path": meta_path, "emb_path": emb_path}


def build_full_index() -> tuple:
    """
    Full pipeline: load chunks → generate embeddings → build index → save.

    Returns:
        (faiss_index, chunks, embeddings)
    """
    # Load chunks
    chunks = load_chunks()

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save everything
    paths = save_index(index, chunks, embeddings)

    print(f"\n{'='*60}")
    print("Embedding & Indexing Complete")
    print(f"{'='*60}")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Index type: FAISS IndexFlatIP (cosine similarity)")

    return index, chunks, embeddings


def build_index_for_year(year: int) -> tuple:
    """
    Build FAISS index for a specific fiscal year.
    Loads chunks from year-specific directory.
    """
    year_chunks_dir = get_chunks_dir(year)
    chunks_path = os.path.join(year_chunks_dir, "all_chunks.json")
    
    if not os.path.exists(chunks_path):
        print(f"No chunks found for FY{year} at {chunks_path}")
        return None, [], np.empty((0, EMBEDDING_DIMENSION))
    
    chunks = load_chunks(chunks_path)
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)
    save_index_for_year(index, chunks, embeddings, year)

    print(f"\n{'='*60}")
    print(f"Embedding & Indexing Complete (FY{year})")
    print(f"{'='*60}")
    print(f"Total vectors indexed: {index.ntotal}")

    return index, chunks, embeddings


if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Generator & FAISS Indexer")
    print("=" * 60)
    build_full_index()
