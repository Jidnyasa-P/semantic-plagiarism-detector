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
from typing import Dict, List, Tuple

# ── Threshold ──────────────────────────────────────────────────────────────────
PLAGIARISM_THRESHOLD = 0.75   # Similarity ≥ this value → flag as potential plagiarism


# ── Document-level similarity ──────────────────────────────────────────────────

def document_similarity_matrix(doc_embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build an N×N cosine similarity matrix between all document pairs.

    Each document is represented by the mean of its chunk embeddings.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array (chunks × 384).

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
        sim = cosine_similarity(stacked)           # (N, N)
        matrix = np.clip(sim, 0.0, 1.0)           # Numerical safety

    df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
    return df


# ── Chunk-level similarity (local plagiarism detection) ────────────────────────

def chunk_max_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray
) -> float:
    """
    Compute the maximum pairwise cosine similarity between chunks of two documents.

    This catches cases where only a section of one document was plagiarised
    from another – even if the overall document similarity is low.

    Args:
        emb_a: Chunk embeddings for document A  (Na × 384)
        emb_b: Chunk embeddings for document B  (Nb × 384)

    Returns:
        Maximum cosine similarity across all chunk pairs (float 0–1).
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0

    sim_matrix = cosine_similarity(emb_a, emb_b)    # (Na, Nb)
    return float(np.max(sim_matrix))


def chunk_similarity_matrix(doc_embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build an N×N matrix where each cell is the MAX chunk-pair similarity.

    This is more sensitive than document-level similarity for detecting
    partial plagiarism.

    Args:
        doc_embeddings: Dict mapping doc name → embedding array.

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
                    doc_embeddings[name_a], doc_embeddings[name_b]
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
