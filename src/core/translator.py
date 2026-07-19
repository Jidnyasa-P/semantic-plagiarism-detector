"""Translation utility for cross-lingual plagiarism alignment."""

from __future__ import annotations

from deep_translator import GoogleTranslator


def translate_text(
    text: str | None,
    target_lang: str = "en",
    source_lang: str = "auto",
) -> str | None:
    """Translate text while preserving the repository's public API.

    Compatibility guarantees:
    - ``None`` returns ``None``.
    - An empty string returns an empty string.
    - Provider/configuration failures return a human-readable string containing
      ``"Translation Error"``.

    The cross-lingual preprocessing layer detects that error prefix and falls
    back to the original source text before embedding, so error messages never
    contaminate FAISS vectors.
    """
    if text is None:
        return None

    original = str(text)
    if not original.strip():
        return original

    try:
        translated = GoogleTranslator(
            source=source_lang or "auto",
            target=target_lang,
        ).translate(original)
    except Exception as exc:
        return f"(Translation Error: {exc})"

    translated = str(translated or "").strip()
    if not translated:
        return f"(Translation Error: empty response for target '{target_lang}')"

    return translated