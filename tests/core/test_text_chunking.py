from src.core.text_chunking import (
    _clean_text,
    _word_count,
    _split_into_paragraphs,
    chunk_document,
)


def test_clean_text():
    raw = "This is a    test.\n\n\n\nToo many newlines."
    cleaned = _clean_text(raw)
    assert "This is a test." in cleaned
    assert "\n\nToo many" in cleaned
    assert "\n\n\n" not in cleaned


def test_word_count():
    assert _word_count("Hello world") == 2
    assert _word_count("  One   two three  ") == 3


def test_split_into_paragraphs():
    text = "Para 1.\n\nPara 2.\n\n\nPara 3."
    cleaned = _clean_text(text)
    paras = _split_into_paragraphs(cleaned)
    assert len(paras) == 3
    assert paras[0] == "Para 1."
    assert paras[1] == "Para 2."
    assert paras[2] == "Para 3."


def test_chunk_document_drops_short_chunks():
    # MIN_CHUNK_WORDS is 20
    text = (
        "Short header\n\n"
        + "This is a longer paragraph that should definitely meet the minimum threshold of twenty words because it just keeps going and going until it crosses the limit."
    )
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert "Short header" not in chunks[0]
    assert "longer paragraph" in chunks[0]


def test_chunk_document_splits_long_chunks():
    # MAX_CHUNK_WORDS is 200. Create a paragraph with 25 sentences of 10 words each.
    sentence = "This is a normal sentence with exactly ten words total. "
    long_para = sentence * 25  # 250 words
    chunks = chunk_document(long_para)
    assert len(chunks) > 1
    assert _word_count(chunks[0]) <= 200
