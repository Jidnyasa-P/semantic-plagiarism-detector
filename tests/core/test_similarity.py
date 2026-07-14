import pytest
import numpy as np
import pandas as pd
from src.core.similarity import (
    document_similarity_matrix,
    chunk_max_similarity,
    chunk_similarity_matrix,
    flag_plagiarism,
    find_most_similar_chunks,
    hybrid_similarity_matrix,
)
from src.core.lexical_similarity import lexical_similarity_matrix

@pytest.fixture
def dummy_embeddings():
    # 3 docs, 384-dim embeddings
    # Doc A and B are very similar, Doc C is completely different
    emb_a = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]) # 2 chunks (dummy dim)
    emb_b = np.array([[0.9, 0.1, 0.0], [0.8, 0.5, 0.0]]) # 2 chunks
    emb_c = np.array([[0.0, 0.0, 1.0]])                  # 1 chunk
    
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
    sim_unbatched = chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"])
    sim_batched = chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=1)
    assert np.isclose(sim_batched, sim_unbatched)


def test_chunk_max_similarity_rejects_invalid_batch_size(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=0.5)


def test_chunk_max_similarity_rejects_invalid_batch_size(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        chunk_max_similarity(dummy_embeddings["doc_A"], dummy_embeddings["doc_B"], batch_size=0.5)


def test_document_similarity_matrix(dummy_embeddings):
    df = document_similarity_matrix(dummy_embeddings)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["doc_A", "doc_B", "doc_C"]


def test_document_similarity_matrix_accepts_batch_size_basic(dummy_embeddings):
    df = document_similarity_matrix(dummy_embeddings, batch_size=2)
    assert isinstance(df, pd.DataFrame)
    
    # Diagonal should be ~1.0
    assert np.isclose(df.loc["doc_A", "doc_A"], 1.0)
    
    # A and B should be more similar to each other than A and C
    assert df.loc["doc_A", "doc_B"] > df.loc["doc_A", "doc_C"]


def test_document_similarity_matrix_accepts_batch_size(dummy_embeddings):
    unbatched = document_similarity_matrix(dummy_embeddings)
    batched = document_similarity_matrix(dummy_embeddings, batch_size=2)
    assert isinstance(batched, pd.DataFrame)
    assert np.allclose(unbatched.values, batched.values)


def test_document_similarity_matrix_rejects_invalid_batch_size(dummy_embeddings):
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        document_similarity_matrix(dummy_embeddings, batch_size=0.5)


def test_chunk_similarity_matrix(dummy_embeddings):
    df = chunk_similarity_matrix(dummy_embeddings)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)


def test_chunk_similarity_matrix_accepts_batch_size_basic(dummy_embeddings):
    df = chunk_similarity_matrix(dummy_embeddings, batch_size=1)
    assert isinstance(df, pd.DataFrame)
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
    emb_1d = np.array([1.0, 0.0, 0.0])
    df = document_similarity_matrix({"doc_1d": emb_1d})
    assert np.isclose(df.loc["doc_1d", "doc_1d"], 1.0)


def test_document_similarity_matrix_empty_embedding():
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
    result = find_most_similar_chunks([], [], np.array([]), np.array([]), top_k=3)
    assert result == []


def test_find_most_similar_chunks_threshold_filters():
    emb_a = np.array([[1.0, 0.0, 0.0]])
    emb_b = np.array([[0.0, 1.0, 0.0]])  # orthogonal → similarity 0.0
    pairs = find_most_similar_chunks(["a"], ["b"], emb_a, emb_b, top_k=3, threshold=0.5)
    assert pairs == []


def test_flag_plagiarism():
    data = [
        [1.0, 0.95, 0.60],
        [0.95, 1.0, 0.80],
        [0.60, 0.80, 1.0]
    ]
    df = pd.DataFrame(data, index=["D1", "D2", "D3"], columns=["D1", "D2", "D3"])
    
    flags = flag_plagiarism(df, threshold=0.75)
    
    assert len(flags) == 2
    
    d1_d2 = next(f for f in flags if f["doc_a"] == "D1" and f["doc_b"] == "D2")
    assert d1_d2["similarity"] == 0.95
    assert "High" in d1_d2["severity"]
    
    d2_d3 = next(f for f in flags if f["doc_a"] == "D2" and f["doc_b"] == "D3")
    assert d2_d3["similarity"] == 0.80
    assert "Medium" in d2_d3["severity"]


def test_lexical_similarity_matrix_identical_documents():
    documents = {
        "doc1": "This is a test document with some text.",
        "doc2": "This is a test document with some text.",
        "doc3": "This is completely different content."
    }
    df = lexical_similarity_matrix(documents)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["doc1", "doc2", "doc3"]
    
    # Identical documents should have similarity 1.0
    assert np.isclose(df.loc["doc1", "doc2"], 1.0)
    assert np.isclose(df.loc["doc1", "doc1"], 1.0)
    
    # Different documents should have lower similarity
    assert df.loc["doc1", "doc3"] < 0.9


def test_lexical_similarity_matrix_empty_documents():
    df = lexical_similarity_matrix({})
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (0, 0)


def test_lexical_similarity_matrix_single_document():
    documents = {"doc1": "Single document content."}
    df = lexical_similarity_matrix(documents)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 1)
    assert np.isclose(df.loc["doc1", "doc1"], 1.0)


def test_hybrid_similarity_matrix_boundary_conditions():
    semantic_df = pd.DataFrame({
        "doc1": [1.0, 0.8, 0.3],
        "doc2": [0.8, 1.0, 0.4],
        "doc3": [0.3, 0.4, 1.0]
    }, index=["doc1", "doc2", "doc3"])
    
    lexical_df = pd.DataFrame({
        "doc1": [1.0, 0.6, 0.2],
        "doc2": [0.6, 1.0, 0.3],
        "doc3": [0.2, 0.3, 1.0]
    }, index=["doc1", "doc2", "doc3"])
    
    # w=1.0 should return pure semantic
    hybrid_pure_semantic = hybrid_similarity_matrix(semantic_df, lexical_df, w=1.0)
    assert np.allclose(hybrid_pure_semantic.values, semantic_df.values)
    
    # w=0.0 should return pure lexical
    hybrid_pure_lexical = hybrid_similarity_matrix(semantic_df, lexical_df, w=0.0)
    assert np.allclose(hybrid_pure_lexical.values, lexical_df.values)
    
    # w=0.5 should be average
    hybrid_avg = hybrid_similarity_matrix(semantic_df, lexical_df, w=0.5)
    expected = (semantic_df + lexical_df) / 2
    assert np.allclose(hybrid_avg.values, expected.values)


def test_hybrid_similarity_matrix_default_weight():
    semantic_df = pd.DataFrame({
        "doc1": [1.0, 0.8],
        "doc2": [0.8, 1.0]
    }, index=["doc1", "doc2"])
    
    lexical_df = pd.DataFrame({
        "doc1": [1.0, 0.6],
        "doc2": [0.6, 1.0]
    }, index=["doc1", "doc2"])
    
    # Default weight should be 0.7
    hybrid_df = hybrid_similarity_matrix(semantic_df, lexical_df)
    expected = 0.7 * semantic_df + 0.3 * lexical_df
    assert np.allclose(hybrid_df.values, expected.values)


def test_hybrid_similarity_matrix_invalid_weight():
    semantic_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
    lexical_df = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]])
    
    with pytest.raises(ValueError, match="Weight w must be between 0.0 and 1.0"):
        hybrid_similarity_matrix(semantic_df, lexical_df, w=1.5)
    
    with pytest.raises(ValueError, match="Weight w must be between 0.0 and 1.0"):
        hybrid_similarity_matrix(semantic_df, lexical_df, w=-0.1)


def test_hybrid_similarity_matrix_shape_mismatch():
    semantic_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
    lexical_df = pd.DataFrame([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    
    with pytest.raises(ValueError, match="Semantic and lexical matrices must have the same shape"):
        hybrid_similarity_matrix(semantic_df, lexical_df)


def test_hybrid_similarity_matrix_index_mismatch():
    semantic_df = pd.DataFrame({
        "doc1": [1.0, 0.8],
        "doc2": [0.8, 1.0]
    }, index=["doc1", "doc2"])
    
    lexical_df = pd.DataFrame({
        "docA": [1.0, 0.6],
        "docB": [0.6, 1.0]
    }, index=["docA", "docB"])
    
    with pytest.raises(ValueError, match="Semantic and lexical matrices must have the same index and columns"):
        hybrid_similarity_matrix(semantic_df, lexical_df)
