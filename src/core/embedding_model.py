"""
embedding_model.py
------------------
Wrapper around SentenceTransformers for generating semantic embeddings.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Multilingual support for English and many other languages
  - 384-dimensional embeddings
  - Strong performance on semantic similarity tasks
  - MIT licensed; safe for academic use
"""

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Singleton model loader ─────────────────────────────────────────────────────
# We load the model once and reuse it across calls to avoid repeated I/O.
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model_name() -> str:
    """Return the configured sentence-transformers model name."""
    return os.getenv("SEMANTIC_PLAGIARISM_MODEL", _DEFAULT_MODEL_NAME)


def _get_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (singleton pattern)."""
    global _model
    if _model is None:
        model_name = _get_model_name()
        print(f"[embedding_model] Loading model: {model_name} …")
        _model = SentenceTransformer(model_name)
        print("[embedding_model] Model loaded successfully.")
    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks:     List of text strings to embed.
        batch_size: Number of texts encoded per forward pass (tune for GPU/CPU).

    Returns:
        numpy array of shape (N, 384) where N = len(chunks).
    """
    if not chunks:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,   # Keep console clean in Streamlit
        normalize_embeddings=True, # L2-normalise → cosine sim = dot product
    )
    return embeddings


def embed_documents(chunked_docs: dict, batch_size: int = 64) -> dict:
    """
    Embed all chunks across multiple documents.

    Args:
        chunked_docs: Dict mapping document name → list of chunk strings.
        batch_size:   Batch size forwarded to encode().

    Returns:
        Dict mapping document name → numpy array of embeddings (shape: N×384).
    """
    embeddings = {}
    for doc_name, chunks in chunked_docs.items():
        if not chunks:
            print(f"[embedding_model] Warning: '{doc_name}' has no chunks. Skipping.")
            embeddings[doc_name] = np.array([])
            continue
        embeddings[doc_name] = embed_chunks(chunks, batch_size=batch_size)

    return embeddings


def get_document_embedding(doc_embedding: np.ndarray) -> np.ndarray:
    """
    Compute a single document-level embedding by averaging its chunk embeddings.

    Using mean pooling over chunks gives a compact representation
    of the whole document for document-level similarity comparisons.

    Args:
        doc_embedding: Array of shape (N, 384) for N chunks.

    Returns:
        1-D array of shape (384,).
    """
    if doc_embedding.ndim == 1:
        return doc_embedding  # Already a single embedding
    return np.mean(doc_embedding, axis=0)
