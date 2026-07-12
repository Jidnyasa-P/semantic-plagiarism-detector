"""
translator.py
-------------
On-demand translation utility to align cross-lingual plagiarized matches into English.
"""

from deep_translator import GoogleTranslator


def translate_text(text: str, target_lang: str = "en") -> str:
    """
    Translate the given text to the target language (default: 'en') on-demand.
    Automatically detects the source language.
    """
    if not text or not text.strip():
        return text
        
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"[translator] Translation failed: {e}")
        # Return fallback error indicator
        return f"(Translation Error: Could not retrieve translation. Detail: {e})"
