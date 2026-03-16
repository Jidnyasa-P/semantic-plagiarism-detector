"""
faiss_index.py
--------------
Builds and queries a FAISS flat index over all document chunk embeddings.

Why FAISS?
----------
Pairwise cosine similarity is O(N²) — fine for 10 documents, painful for 1000+.
FAISS performs approximate nearest-neighbour (ANN) search in O(N log N),
making it practical for thousands of assignments.

Index type used: IndexFlatIP (exact inner product search)
  - Since embeddings are L2-normalised, inner product == cosine similarity.
  - "Flat" means exact (no approximation), safe for academic use cases.
  - For 100k+ chunks, swap to IndexIVFFlat for speed (see comment in build_index).
"""

"""
faiss_index.py
--------------
Builds and queries a FAISS flat index over all document chunk embeddings.
"""
# FAISS has no official type stubs; suppress Pylance false positives
import faiss  # type: ignore
import numpy as np
from typing import Dict, List, Tuple, Optional


class ChunkRecord:
    """Stores metadata for a single chunk stored in the FAISS index."""
    __slots__ = ("doc_name", "chunk_index", "chunk_text")

    def __init__(self, doc_name: str, chunk_index: int, chunk_text: str):
        self.doc_name    = doc_name
        self.chunk_index = chunk_index
        self.chunk_text  = chunk_text

    def __repr__(self):
        preview = self.chunk_text[:60].replace("\n", " ")
        return f"ChunkRecord({self.doc_name!r}, idx={self.chunk_index}, '{preview}…')"


def build_index(
    embeddings:   Dict[str, np.ndarray],
    chunked_docs: Dict[str, List[str]],
) -> Tuple[faiss.Index, List[ChunkRecord]]:
    dim = 384
    all_vectors: List[np.ndarray] = []
    registry:    List[ChunkRecord] = []

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue
        for i, (vec, text) in enumerate(zip(emb, chunks)):
            all_vectors.append(vec.astype("float32"))
            registry.append(ChunkRecord(doc_name, i, text))

    if not all_vectors:
        return faiss.IndexFlatIP(dim), registry

    matrix = np.vstack(all_vectors)
    index  = faiss.IndexFlatIP(dim)
    index.add(matrix)  # type: ignore[arg-type]
    return index, registry


def search_similar_chunks(
    query_embedding: np.ndarray,
    index:           faiss.Index,
    registry:        List[ChunkRecord],
    top_k:           int = 10,
    exclude_doc:     Optional[str] = None,
    threshold:       float = 0.0,
) -> List[Tuple[ChunkRecord, float]]:
    vec     = query_embedding.astype("float32").reshape(1, -1)
    fetch_k = min(top_k * 3, index.ntotal) if exclude_doc else top_k
    fetch_k = max(fetch_k, 1)

    scores, indices = index.search(vec, fetch_k) # type: ignore[call-arg]

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        record = registry[idx]
        if exclude_doc and record.doc_name == exclude_doc:
            continue
        if score < threshold:
            continue
        results.append((record, float(score)))
        if len(results) >= top_k:
            break

    return results


def find_plagiarised_chunks(
    embeddings:   Dict[str, np.ndarray],
    chunked_docs: Dict[str, List[str]],
    index:        faiss.Index,
    registry:     List[ChunkRecord],
    threshold:    float = 0.75,
    top_k:        int = 5,
) -> List[Dict]:
    matches    = []
    seen_pairs = set()

    for doc_name, emb in embeddings.items():
        chunks = chunked_docs.get(doc_name, [])
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue

        for chunk_idx, vec in enumerate(emb):
            results = search_similar_chunks(
                vec, index, registry,
                top_k=top_k,
                exclude_doc=doc_name,
                threshold=threshold,
            )
            for record, score in results:
                pair_key = tuple(sorted([
                    (doc_name, chunk_idx),
                    (record.doc_name, record.chunk_index)
                ]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                matches.append({
                    "source_doc":        doc_name,
                    "source_chunk_text": chunks[chunk_idx],
                    "match_doc":         record.doc_name,
                    "match_chunk_text":  record.chunk_text,
                    "similarity":        round(score, 4),
                })

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)
    print(f"[faiss_index] Index saved → {path}  ({index.ntotal} vectors)")


def load_index(path: str) -> faiss.Index:
    index = faiss.read_index(path)
    print(f"[faiss_index] Index loaded ← {path}  ({index.ntotal} vectors)")
    return index