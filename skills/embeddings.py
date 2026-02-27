"""
Vector embeddings for semantic search.
Uses sentence-transformers (all-MiniLM-L6-v2) — runs locally, no API key needed.
Model loads lazily on first use (~200MB download, then cached).

Embeddings are persisted to SQLite at index time and loaded at query time,
so we never re-embed the entire index on every question.
"""
import numpy as np
from pathlib import Path

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


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


def build_and_save_embeddings(file_index: dict, project_root: str) -> int:
    """Build embeddings for all chunks and persist to SQLite.
    Called at index time. Returns number of chunks embedded."""
    from skills.storage import save_embeddings

    all_texts = []
    all_keys = []  # (rel_path, chunk_index)

    for rel_path, meta in file_index.items():
        for i, chunk in enumerate(meta.get("chunks", [])):
            symbol = chunk.get("symbol") or ""
            text = f"{rel_path} {symbol}\n{chunk['content'][:500]}"
            all_texts.append(text)
            all_keys.append((rel_path, i))

    if not all_texts:
        return 0

    vectors = embed_texts(all_texts)

    # Convert to list of (rel_path, chunk_index, vector_bytes) for storage
    embeddings_data = []
    for (rel_path, chunk_idx), vec in zip(all_keys, vectors):
        embeddings_data.append((rel_path, chunk_idx, vec.tobytes()))

    save_embeddings(project_root, embeddings_data)
    return len(embeddings_data)


def load_and_search(query: str, project_root: str = "", identifier: str = "",
                    top_k: int = 10) -> list[tuple[str, int, float]]:
    """Load persisted embeddings from SQLite and search by semantic similarity.
    Returns [(rel_path, chunk_index, score), ...] sorted by score desc."""
    from skills.storage import load_embeddings

    rows = load_embeddings(project_root=project_root, identifier=identifier)
    if not rows:
        return []

    # Reconstruct numpy arrays
    chunk_keys = []
    vectors = []
    for rel_path, chunk_idx, vec_bytes in rows:
        chunk_keys.append((rel_path, chunk_idx))
        vectors.append(np.frombuffer(vec_bytes, dtype=np.float32))

    chunk_vectors = np.stack(vectors)
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, chunk_vectors)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        rel_path, chunk_index = chunk_keys[idx]
        results.append((rel_path, chunk_index, float(scores[idx])))

    return results
