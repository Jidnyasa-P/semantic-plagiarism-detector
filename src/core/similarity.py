"""
similarity.py
-------------
Computes semantic similarity between documents at two levels:
  1. Document-level  – single score per pair (mean-pooled embeddings)
  2. Chunk-level     – max-similarity per chunk pair (detects local plagiarism)

Uses cosine similarity. Since embeddings are L2-normalised in embedding_model.py,
cosine similarity reduces to the dot product, making this very fast.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple

# ── Threshold ──────────────────────────────────────────────────────────────────
# Empirically determined optimal value via evaluation/evaluate.py (F1 = 1.0).
# Previous arbitrary default was 0.75; data-driven analysis found 0.59 to be
# the lowest threshold achieving perfect precision AND recall on the benchmark.
PLAGIARISM_THRESHOLD = 0.59


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validated_batch_size(batch_size: Optional[int]) -> Optional[int]:
    """Return a safe integer batch size or None for unbatched execution."""
    if batch_size is None:
        return None
    if isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer")
    if isinstance(batch_size, (float, np.floating)) and not float(batch_size).is_integer():
        raise ValueError("batch_size must be an integer")
    try:
        size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be an integer") from exc
    return size if size > 0 else None


# ── Document-level similarity ──────────────────────────────────────────────────

def document_similarity_matrix(
    doc_embeddings: Dict[str, np.ndarray],
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build an N×N cosine similarity matrix between all document pairs.

    Each document is represented by the mean of its chunk embeddings.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array (chunks × 384).
        batch_size: Optional number of documents to compare per batch.
            When set, the similarity computation is carried out in smaller blocks
            to reduce peak memory usage for larger datasets.

    Returns:
        Symmetric pandas DataFrame with document names as index and columns.
        Values range 0.0 – 1.0 (1.0 = identical).
    """
    doc_names = list(doc_embeddings.keys())
    n = len(doc_names)

    # Build document-level vectors (mean pool over chunks)
    doc_vectors = []
    for name in doc_names:
        emb = doc_embeddings[name]
        if emb.ndim == 2 and emb.shape[0] > 0:
            vec = np.mean(emb, axis=0)
        elif emb.ndim == 1 and emb.shape[0] > 0:
            vec = emb
        else:
            vec = np.zeros(384)  # Fallback for empty docs
        doc_vectors.append(vec)

    matrix = np.zeros((n, n))
    if doc_vectors:
        stacked = np.vstack(doc_vectors)           # (N, 384)
        safe_batch_size = _validated_batch_size(batch_size)
        if safe_batch_size is None:
            sim = cosine_similarity(stacked)       # (N, N)
            matrix = np.clip(sim, 0.0, 1.0)       # Numerical safety
        else:
            for start in range(0, n, safe_batch_size):
                end = min(start + safe_batch_size, n)
                sim = cosine_similarity(stacked[start:end], stacked)
                matrix[start:end] = np.clip(sim, 0.0, 1.0)

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return df


# ── Chunk-level similarity (local plagiarism detection) ────────────────────────

def chunk_max_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    batch_size: Optional[int] = None,
) -> float:
    """
    Compute the maximum pairwise cosine similarity between chunks of two documents.

    This catches cases where only a section of one document was plagiarised
    from another – even if the overall document similarity is low.

    Args:
        emb_a: Chunk embeddings for document A  (Na × 384)
        emb_b: Chunk embeddings for document B  (Nb × 384)
        batch_size: Optional number of rows/columns to compare per batch.
            When set, the comparison is processed in smaller blocks to lower
            peak memory usage for large chunk sets.

    Returns:
        Maximum cosine similarity across all chunk pairs (float 0–1).
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0

    safe_batch_size = _validated_batch_size(batch_size)
    if safe_batch_size is None:
        sim_matrix = cosine_similarity(emb_a, emb_b)    # (Na, Nb)
        return float(np.max(sim_matrix))

    max_score = 0.0
    for start_a in range(0, emb_a.shape[0], safe_batch_size):
        end_a = min(start_a + safe_batch_size, emb_a.shape[0])
        for start_b in range(0, emb_b.shape[0], safe_batch_size):
            end_b = min(start_b + safe_batch_size, emb_b.shape[0])
            sim_matrix = cosine_similarity(emb_a[start_a:end_a], emb_b[start_b:end_b])
            max_score = max(max_score, float(np.max(sim_matrix)))
            if max_score >= 1.0:
                return max_score
    return max_score


def chunk_similarity_matrix(
    doc_embeddings: Dict[str, np.ndarray],
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build an N×N matrix where each cell is the MAX chunk-pair similarity.

    This is more sensitive than document-level similarity for detecting
    partial plagiarism.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array.
        batch_size: Optional number of chunks to compare per batch for each
            document pair.

    Returns:
        Symmetric pandas DataFrame with max-chunk similarity values.
    """
    doc_names = list(doc_embeddings.keys())
    n = len(doc_names)
    matrix = np.zeros((n, n))

    for i, name_a in enumerate(doc_names):
        for j, name_b in enumerate(doc_names):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                score = chunk_max_similarity(
                    doc_embeddings[name_a],
                    doc_embeddings[name_b],
                    batch_size=batch_size,
                )
                matrix[i][j] = score
                matrix[j][i] = score   # Symmetric

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return df


# ── Plagiarism flagging ────────────────────────────────────────────────────────

def flag_plagiarism(
    similarity_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD
) -> List[Dict]:
    """
    Identify document pairs whose similarity exceeds the threshold.

    Args:
        similarity_df: Symmetric similarity DataFrame (doc × doc).
        threshold:     Minimum similarity to flag (default: 0.75).

    Returns:
        List of dicts, each containing:
          - doc_a     : Name of first document
          - doc_b     : Name of second document
          - similarity: Cosine similarity score (float)
          - severity  : "High" (≥0.90) | "Medium" (≥0.75)
    """
    flags = []
    doc_names = similarity_df.columns.tolist()
    n = len(doc_names)

    for i in range(n):
        for j in range(i + 1, n):   # Upper triangle only (avoid duplicates)
            score = similarity_df.iloc[i, j]
            if score >= threshold:
                severity = "🔴 High" if score >= 0.90 else "🟡 Medium"
                flags.append({
                    "doc_a": doc_names[i],
                    "doc_b": doc_names[j],
                    "similarity": round(float(score), 4),
                    "severity": severity,
                })

    # Sort by similarity descending
    flags.sort(key=lambda x: x["similarity"], reverse=True)
    return flags


def find_most_similar_chunks(
    chunks_a: List[str],
    chunks_b: List[str],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    top_k: int = 3,
    threshold: float = PLAGIARISM_THRESHOLD
) -> List[Tuple[str, str, float]]:
    """
    Find the top-K most similar chunk pairs between two documents.

    Useful for showing teachers WHICH paragraphs are suspicious.

    Args:
        chunks_a: Raw text chunks from document A.
        chunks_b: Raw text chunks from document B.
        emb_a:    Embeddings for document A (Na × 384).
        emb_b:    Embeddings for document B (Nb × 384).
        top_k:    Number of top pairs to return.
        threshold: Only return pairs above this threshold.

    Returns:
        List of (chunk_from_A, chunk_from_B, similarity_score) tuples.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return []

    sim_matrix = cosine_similarity(emb_a, emb_b)   # (Na, Nb)

    # Flatten and sort
    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            score = sim_matrix[i, j]
            if score >= threshold:
                pairs.append((chunks_a[i], chunks_b[j], float(score)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
