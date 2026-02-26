"""
Vector embeddings for semantic search.
Uses sentence-transformers (all-MiniLM-L6-v2) — runs locally, no API key needed.
Model loads lazily on first use (~200MB download, then cached).
"""
import json
import numpy as np
from pathlib import Path

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the embedding model (only downloaded once, cached by HuggingFace)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into vectors. Returns (N, dim) numpy array."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (dim,) numpy array."""
    model = _get_model()
    return model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all doc vectors.
    Assumes vectors are already L2-normalized (which sentence-transformers does)."""
    return doc_vecs @ query_vec


def build_chunk_embeddings(file_index: dict) -> dict:
    """Build embeddings for all chunks in the file index.
    Returns {rel_path: [(chunk_index, embedding), ...]}"""
    all_texts = []
    all_keys = []  # (rel_path, chunk_index)

    for rel_path, meta in file_index.items():
        for i, chunk in enumerate(meta.get("chunks", [])):
            # Embed a summary: file path + symbol + first few lines
            symbol = chunk.get("symbol") or ""
            text = f"{rel_path} {symbol}\n{chunk['content'][:500]}"
            all_texts.append(text)
            all_keys.append((rel_path, i))

    if not all_texts:
        return {}, np.array([])

    vectors = embed_texts(all_texts)
    return all_keys, vectors


def semantic_search(query: str, chunk_keys: list, chunk_vectors: np.ndarray,
                    top_k: int = 10) -> list[tuple[str, int, float]]:
    """Search chunks by semantic similarity.
    Returns [(rel_path, chunk_index, score), ...] sorted by score desc."""
    if len(chunk_vectors) == 0:
        return []

    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, chunk_vectors)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        rel_path, chunk_index = chunk_keys[idx]
        results.append((rel_path, chunk_index, float(scores[idx])))

    return results
