"""
lexical_similarity.py
---------------------
Computes lexical similarity between documents using TF-IDF vectorization.

This module provides a TF-IDF based baseline for plagiarism detection,
which excels at identifying identical lexical copy-pasting.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict


def lexical_similarity_matrix(documents: Dict[str, str]) -> pd.DataFrame:
    """
    Build an N×N TF-IDF cosine similarity matrix between all document pairs.

    A single TfidfVectorizer is fitted across all documents to ensure
    consistent vocabulary across the entire corpus, then cosine similarity
    is computed between all document pairs.

    Args:
        documents: Dict mapping doc name → raw text content.

    Returns:
        Symmetric pandas DataFrame with document names as index and columns.
        Values range 0.0 – 1.0 (1.0 = identical).
    """
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
