"""
lexical_similarity.py
---------------------
Computes lexical similarity between documents using TF-IDF vectorization.

This module provides a TF-IDF based baseline for plagiarism detection,
which excels at identifying identical lexical copy-pasting.
"""

import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import functools


def _make_documents_hash(documents: Dict[str, str]) -> str:
    """
    Create a stable hash from document contents for caching.

    Args:
        documents: Dict mapping doc name → raw text content.

    Returns:
        SHA256 hash string of the sorted document contents.
    """
    # Sort by document name to ensure consistent hashing
    sorted_items = sorted(documents.items())
    hash_input = str(sorted_items).encode("utf-8")
    return hashlib.sha256(hash_input).hexdigest()


@functools.lru_cache(maxsize=32)
def _cached_lexical_similarity_matrix(
    documents_hash: str, documents_tuple: tuple
) -> pd.DataFrame:
    """
    Internal cached implementation of lexical similarity matrix.

    This function uses lru_cache for Python-level caching. The documents
    are passed as a tuple to make them hashable for the cache.

    Args:
        documents_hash: Hash of the document contents (for cache key).
        documents_tuple: Tuple of (doc_name, doc_text) pairs.

    Returns:
        Symmetric pandas DataFrame with document names as index and columns.
        Values range 0.0 – 1.0 (1.0 = identical).
    """
    documents = dict(documents_tuple)
    doc_names = list(documents.keys())
    n = len(doc_names)

    if n == 0:
        return pd.DataFrame()

    # Extract texts in the same order as doc_names
    texts = [documents[name] for name in doc_names]

    # Fit a single TfidfVectorizer across all documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)  # (N, vocab_size)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)  # (N, N)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)  # Numerical safety

    df = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
    return df


def lexical_similarity_matrix(
    documents: Dict[str, str], use_cache: bool = True
) -> pd.DataFrame:
    """
    Build an N×N TF-IDF cosine similarity matrix between all document pairs.

    A single TfidfVectorizer is fitted across all documents to ensure
    consistent vocabulary across the entire corpus, then cosine similarity
    is computed between all document pairs.

    Args:
        documents: Dict mapping doc name → raw text content.
        use_cache: If True (default), use LRU cache to avoid recomputing
                   TF-IDF matrices for identical document sets. Set to False
                   to force recomputation.

    Returns:
        Symmetric pandas DataFrame with document names as index and columns.
        Values range 0.0 – 1.0 (1.0 = identical).
    """
    if use_cache:
        # Convert dict to tuple for hashability
        documents_tuple = tuple(sorted(documents.items()))
        documents_hash = _make_documents_hash(documents)
        return _cached_lexical_similarity_matrix(documents_hash, documents_tuple)
    else:
        # Uncached path for testing or when cache should be bypassed
        doc_names = list(documents.keys())
        n = len(doc_names)

        if n == 0:
            return pd.DataFrame()

        # Extract texts in the same order as doc_names
        texts = [documents[name] for name in doc_names]

        # Fit a single TfidfVectorizer across all documents
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)  # (N, vocab_size)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)  # (N, N)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)  # Numerical safety

        df = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
        return df
