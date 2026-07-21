from src.core.translator import translate_text


def test_none_is_preserved():
    assert translate_text(None) is None


def test_empty_string_is_preserved():
    assert translate_text("") == ""


def test_invalid_language_returns_compatible_error_message():
    result = translate_text("Hello", target_lang="invalid_lang")
    assert isinstance(result, str)
    assert "Translation Error" in result


def test_translate_text_basic():
    # Translate a simple French sentence to English
    result = translate_text("Bonjour tout le monde", target_lang="en")
    assert "hello" in result.lower() or "everyone" in result.lower()


def test_translate_text_empty():
    # Empty inputs should be returned as-is
    assert translate_text("") == ""
    assert translate_text("   ") == "   "
    assert translate_text(None) is None


def test_translate_text_error_handling():
    # Using an invalid language code should trigger the exception and return the error detail message
    result = translate_text("Hello", target_lang="invalid_lang")
    assert "Translation Error" in result
