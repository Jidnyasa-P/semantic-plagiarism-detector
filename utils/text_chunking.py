"""
text_chunking.py
----------------
Splits raw document text into meaningful paragraph-level chunks.
Chunking improves plagiarism detection by enabling localised similarity
comparisons rather than comparing entire documents as single blobs.
"""

import re
from typing import List


# ── Constants ──────────────────────────────────────────────────────────────────
MIN_CHUNK_WORDS = 20        # Discard chunks shorter than this (likely noise)
MAX_CHUNK_WORDS = 200       # Hard ceiling; longer chunks are sub-split


def _clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Collapse multiple blank lines → single blank line
    - Strip leading/trailing whitespace per line
    - Remove non-printable characters
    """
    # Remove non-printable except newlines and tabs
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines → double newline (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text on blank lines (standard paragraph boundary).
    Each paragraph is stripped and empty ones are discarded.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def _word_count(text: str) -> int:
    return len(text.split())


def _sub_split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    If a paragraph exceeds max_words, split it on sentence boundaries
    and group sentences into sub-chunks below the word limit.
    """
    # Simple sentence tokeniser (handles . ! ?)
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current_chunk: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_wc = _word_count(sentence)
        if current_word_count + sentence_wc > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_wc
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_wc

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(text: str) -> List[str]:
    """
    Full chunking pipeline for a single document:
    1. Clean raw text
    2. Split into paragraphs
    3. Discard very short paragraphs (noise)
    4. Sub-split overly long paragraphs

    Args:
        text: Raw extracted text from a PDF.

    Returns:
        A list of text chunks (strings) ready for embedding.
    """
    text = _clean_text(text)
    paragraphs = _split_into_paragraphs(text)

    final_chunks: List[str] = []

    for para in paragraphs:
        wc = _word_count(para)

        if wc < MIN_CHUNK_WORDS:
            # Too short → likely a heading or page artefact; skip
            continue
        elif wc > MAX_CHUNK_WORDS:
            # Too long → sub-split on sentences
            sub_chunks = _sub_split_long_paragraph(para, MAX_CHUNK_WORDS)
            final_chunks.extend(
                [c for c in sub_chunks if _word_count(c) >= MIN_CHUNK_WORDS]
            )
        else:
            final_chunks.append(para)

    return final_chunks


def chunk_documents(documents: dict) -> dict:
    """
    Chunk multiple documents.

    Args:
        documents: Dict mapping document name → raw text string.

    Returns:
        Dict mapping document name → list of text chunks.
    """
    return {name: chunk_document(text) for name, text in documents.items()}
