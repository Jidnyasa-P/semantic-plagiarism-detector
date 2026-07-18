"""Verifies that vectors use translated text while the registry keeps source text."""

import numpy as np

from src.core.cross_lingual import prepare_documents_for_embedding


def test_translation_indexing_contract(monkeypatch):
    source_chunks = {
        "english.pdf": ["Artificial intelligence supports education."],
        "hindi.pdf": ["कृत्रिम बुद्धिमत्ता शिक्षा का समर्थन करती है।"],
    }

    def fake_prepare(text):
        is_hindi = text.startswith("कृत्रिम")
        return {
            "original_text": text,
            "embedding_text": (
                "Artificial intelligence supports education."
                if is_hindi
                else text
            ),
            "detected_language": "hi" if is_hindi else "en",
            "translated": is_hindi,
            "translation_failed": False,
        }

    monkeypatch.setattr(
        "src.core.cross_lingual.prepare_text_for_embedding",
        fake_prepare,
    )

    aligned, metadata = prepare_documents_for_embedding(source_chunks)

    # Embedding input is aligned in English.
    assert aligned["english.pdf"] == aligned["hindi.pdf"]

    # Display/database source remains in its original language.
    assert source_chunks["hindi.pdf"][0].startswith("कृत्रिम")
    assert metadata["hindi.pdf"][0]["original_text"].startswith("कृत्रिम")
