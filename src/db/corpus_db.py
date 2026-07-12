"""
corpus_db.py
------------
SQLite database manager to persist document metadata, chunk text, and embeddings.
Enables incremental updates and index rebuilding without re-embedding.
"""

import sqlite3
import os
import numpy as np
from datetime import datetime

_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "corpus.db"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_corpus_db() -> None:
    """Create the corpus and chunks tables if they do not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                filename     TEXT    UNIQUE NOT NULL,
                file_hash    TEXT    UNIQUE NOT NULL,
                upload_date  TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id    INTEGER PRIMARY KEY,
                filename     TEXT    NOT NULL,
                chunk_index  INTEGER NOT NULL,
                chunk_text   TEXT    NOT NULL,
                embedding    BLOB    NOT NULL,
                FOREIGN KEY (filename) REFERENCES documents(filename) ON DELETE CASCADE
            )
        """)
        conn.commit()


def add_document(filename: str, file_hash: str) -> bool:
    """
    Insert a new document metadata row.
    Returns True if successfully inserted, False if it already exists.
    """
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                (filename, file_hash, datetime.now().isoformat())
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_document_by_hash(file_hash: str) -> str | None:
    """Check if a file with this hash is already indexed and return its filename."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return row[0] if row else None


def get_all_documents() -> list:
    """Return all indexed documents sorted by upload date descending."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, file_hash, upload_date FROM documents ORDER BY upload_date DESC"
        ).fetchall()
    return [{"filename": r[0], "file_hash": r[1], "upload_date": r[2]} for r in rows]


def add_chunks(chunks_to_add: list) -> None:
    """
    Insert a batch of chunks with their raw text and embedded BLOBs.
    
    chunks_to_add: list of tuples: (vector_id, filename, chunk_index, chunk_text, embedding_np_array)
    """
    formatted_chunks = []
    for vid, fname, idx, text, emb in chunks_to_add:
        # Convert float32 numpy array to raw bytes BLOB
        emb_blob = emb.astype(np.float32).tobytes()
        formatted_chunks.append((vid, fname, idx, text, emb_blob))

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            formatted_chunks
        )
        conn.commit()


def get_chunk_registry() -> list:
    """Reconstructs the registry of ChunkRecord objects ordered by vector_id."""
    from src.core.faiss_index import ChunkRecord
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, chunk_index, chunk_text FROM chunks ORDER BY vector_id ASC"
        ).fetchall()
    return [ChunkRecord(r[0], r[1], r[2]) for r in rows]


def get_all_embeddings() -> np.ndarray:
    """Load all chunk embeddings from the database to rebuild the FAISS index."""
    with _connect() as conn:
        rows = conn.execute("SELECT embedding FROM chunks ORDER BY vector_id ASC").fetchall()
    
    if not rows:
        return np.empty((0, 384), dtype=np.float32)
        
    embeddings = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
    return np.vstack(embeddings)


def delete_document(filename: str) -> None:
    """
    Delete a document and all its associated chunks (cascade).
    After deletion, vector_ids will have gaps, so we need to compact the vector IDs.
    """
    with _connect() as conn:
        # Delete document (triggers cascading delete on chunks)
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()
        
    # Re-index all remaining chunks so vector_ids are sequential [0, 1, ..., N-1]
    _compact_vector_ids()


def _compact_vector_ids() -> None:
    """Re-index the vector_id column to remove any gaps left by deleted documents."""
    with _connect() as conn:
        # Retrieve all chunks ordered by current vector_id
        chunks = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY vector_id ASC"
        ).fetchall()
        
        # Clear chunks table
        conn.execute("DELETE FROM chunks")
        
        # Insert them back with fresh sequential IDs starting at 0
        if chunks:
            formatted = [(i, r[0], r[1], r[2], r[3]) for i, r in enumerate(chunks)]
            conn.executemany(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                formatted
            )
        conn.commit()


def get_document_chunks_count(filename: str) -> int:
    """Return the number of chunks for a given document."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) FROM chunks WHERE filename = ?", (filename,)
        ).fetchone()
    return row[0] if row else 0


def clear_all_data() -> None:
    """Wipe all database records (useful for reset)."""
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
