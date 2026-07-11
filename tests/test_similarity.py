import pytest
import numpy as np
import pandas as pd
from utils.similarity import (
    document_similarity_matrix,
    chunk_max_similarity,
    chunk_similarity_matrix,
    flag_plagiarism,
    find_most_similar_chunks,
)

@pytest.fixture
def dummy_embeddings():
    # 3 docs, 384-dim embeddings
    # Doc A and B are very similar, Doc C is completely different
    emb_a = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]) # 2 chunks (dummy dim)
    emb_b = np.array([[0.9, 0.1, 0.0], [0.8, 0.5, 0.0]]) # 2 chunks
    emb_c = np.array([[0.0, 0.0, 1.0]])                  # 1 chunk
    
    # Pad to 384 dimensions to match actual model output expectations if needed,
    # but cosine_similarity handles arbitrary dimensions
    return {
        "doc_A": emb_a,
        "doc_B": emb_b,
        "doc_C": emb_c
    }

def test_chunk_max_similarity(dummy_embeddings):
    emb_a = dummy_embeddings["doc_A"]
    emb_b = dummy_embeddings["doc_B"]
    emb_c = dummy_embeddings["doc_C"]
    
    # Similarity should be high between A and B
    sim_ab = chunk_max_similarity(emb_a, emb_b)
    assert sim_ab > 0.8
    
    # Similarity should be low between A and C
    sim_ac = chunk_max_similarity(emb_a, emb_c)
    assert sim_ac < 0.1
    
    # Empty embedding handling
    assert chunk_max_similarity(emb_a, np.array([])) == 0.0


def test_chunk_max_similarity_supports_batching(dummy_embeddings):
    unbatched = chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"])
    batched = chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=1)
    assert np.isclose(batched, unbatched)


def test_document_similarity_matrix(dummy_embeddings):
    df = document_similarity_matrix(dummy_embeddings)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["doc_A", "doc_B", "doc_C"]
    
    # Diagonal should be ~1.0
    assert np.isclose(df.loc["doc_A", "doc_A"], 1.0)
    
    # A and B should be more similar to each other than A and C
    assert df.loc["doc_A", "doc_B"] > df.loc["doc_A", "doc_C"]


def test_document_similarity_matrix_accepts_batch_size(dummy_embeddings):
    unbatched = document_similarity_matrix(dummy_embeddings)
    batched = document_similarity_matrix(dummy_embeddings, batch_size=2)
    assert isinstance(batched, pd.DataFrame)
    assert np.allclose(unbatched.values, batched.values)


def test_chunk_similarity_matrix(dummy_embeddings):
    df = chunk_similarity_matrix(dummy_embeddings)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    
    # Diagonal should be 1.0
    assert df.loc["doc_A", "doc_A"] == 1.0
    
    # Symmetric
    assert df.loc["doc_A", "doc_B"] == df.loc["doc_B", "doc_A"]


def test_chunk_similarity_matrix_accepts_batch_size(dummy_embeddings):
    unbatched = chunk_similarity_matrix(dummy_embeddings)
    batched = chunk_similarity_matrix(dummy_embeddings, batch_size=1)
    assert isinstance(batched, pd.DataFrame)
    assert np.allclose(unbatched.values, batched.values)


def test_batch_size_rejects_non_integer(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        document_similarity_matrix(dummy_embeddings, batch_size=0.5)
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=0.5)
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_similarity_matrix(dummy_embeddings, batch_size=0.5)


def test_document_similarity_matrix_1d_embedding():
    # Covers the elif emb.ndim == 1 branch (line 48)
    emb_1d = np.array([1.0, 0.0, 0.0])
    df = document_similarity_matrix({"doc_1d": emb_1d})
    assert np.isclose(df.loc["doc_1d", "doc_1d"], 1.0)


def test_document_similarity_matrix_empty_embedding():
    # Covers the zero-vector fallback branch (line 50-51)
    df = document_similarity_matrix({"empty": np.array([])})
    assert df.shape == (1, 1)


def test_find_most_similar_chunks_returns_top_pairs():
    emb_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    emb_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    chunks_a = ["chunk a0", "chunk a1"]
    chunks_b = ["chunk b0", "chunk b1"]
    pairs = find_most_similar_chunks(chunks_a, chunks_b, emb_a, emb_b, top_k=2, threshold=0.5)
    assert len(pairs) >= 1
    assert pairs[0][0] == "chunk a0"
    assert pairs[0][1] == "chunk b0"
    assert pairs[0][2] > 0.5


def test_find_most_similar_chunks_empty_embeddings():
    # Covers the early-return branch (line 187)
    result = find_most_similar_chunks([], [], np.array([]), np.array([]), top_k=3)
    assert result == []


def test_find_most_similar_chunks_threshold_filters():
    emb_a = np.array([[1.0, 0.0, 0.0]])
    emb_b = np.array([[0.0, 1.0, 0.0]])  # orthogonal → similarity 0.0
    pairs = find_most_similar_chunks(["a"], ["b"], emb_a, emb_b, top_k=3, threshold=0.5)
    assert pairs == []


def test_flag_plagiarism():
    # Mock a similarity DataFrame
    data = [
        [1.0, 0.95, 0.60],
        [0.95, 1.0, 0.80],
        [0.60, 0.80, 1.0]
    ]
    df = pd.DataFrame(data, index=["D1", "D2", "D3"], columns=["D1", "D2", "D3"])
    
    # Default threshold is 0.59 (or 0.75 in older versions), let's use 0.75 for testing
    flags = flag_plagiarism(df, threshold=0.75)
    
    assert len(flags) == 2
    # Should flag D1-D2 (0.95 -> High) and D2-D3 (0.80 -> Medium)
    
    d1_d2 = next(f for f in flags if f["doc_a"] == "D1" and f["doc_b"] == "D2")
    assert d1_d2["similarity"] == 0.95
    assert "High" in d1_d2["severity"]
    
    d2_d3 = next(f for f in flags if f["doc_a"] == "D2" and f["doc_b"] == "D3")
    assert d2_d3["similarity"] == 0.80
    assert "Medium" in d2_d3["severity"]
