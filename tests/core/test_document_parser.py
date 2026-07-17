import io
import shutil
import pytest
import pypdf
import docx
from unittest.mock import MagicMock, patch
from src.core.document_parser import (
    extract_text,
    extract_texts,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
)

# Skip OCR tests when Tesseract binary is not present on this machine
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF containing the given text."""
    writer = pypdf.PdfWriter()
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    # Ensure there are enough words to bypass OCR fallback (at least 8 words)
    words = (text + " word" * 10).split()
    c.drawString(50, 150, " ".join(words))
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal in-memory DOCX containing the given text."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract OCR is not installed on this machine")
def test_extract_from_pdf_bytes():
    pdf_bytes = _make_pdf_bytes("Hello PDF")
    # For blank page PDF, pdfplumber might return empty string, but it shouldn't error
    result = extract_text_from_pdf(pdf_bytes)
    assert isinstance(result, str)


def test_extract_from_pdf_filters_repeated_headers_page_numbers_and_whitespace():
    # Build mock pages where header/footer lines are REPEATED across pages
    # and page numbers sit on their own dedicated lines (so the filter strips them)
    page_one = MagicMock()
    page_one.extract_text.return_value = (
        "Research Report\n"
        "Introduction\n"
        "This section introduces the topic in detail with enough words.\n"
        "Page 1"
    )
    page_two = MagicMock()
    page_two.extract_text.return_value = (
        "Research Report\n"
        "Body content is written here at length for analysis purposes.\n"
        "Page 2"
    )

    fake_pdf = MagicMock()
    fake_pdf.pages = [page_one, page_two]
    fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
    fake_pdf.__exit__ = MagicMock(return_value=False)

    with patch("src.core.document_parser.pdfplumber.open", return_value=fake_pdf):
        result = extract_text_from_pdf(io.BytesIO(b"fake-pdf"))

    # Repeated header across all pages must be stripped
    assert "Research Report" not in result
    # Standalone page-number lines must be stripped
    assert "Page 1" not in result
    assert "Page 2" not in result
    # Body content must survive
    assert "Introduction" in result
    assert "Body content" in result
    assert "\n\n\n" not in result


def test_extract_from_docx_bytes():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    result = extract_text_from_docx(docx_bytes)
    assert result == "Hello DOCX"


def test_extract_from_txt_bytes():
    txt_bytes = b"Hello TXT"
    result = extract_text_from_txt(txt_bytes)
    assert result == "Hello TXT"


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract OCR is not installed on this machine")
def test_extract_text_routing():
    pdf_bytes = _make_pdf_bytes("Hello PDF")
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    assert isinstance(extract_text(pdf_bytes, "test.pdf"), str)
    assert extract_text(docx_bytes, "test.docx") == "Hello DOCX"
    assert extract_text(txt_bytes, "test.txt") == "Hello TXT"
    # Fallback case
    assert extract_text(txt_bytes, "test.unknown") == "Hello TXT"


def test_extract_texts_mixed():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    mock_file1 = MagicMock()
    mock_file1.name = "doc1.docx"
    mock_file1.read.return_value = docx_bytes

    mock_file2 = MagicMock()
    mock_file2.name = "doc2.txt"
    mock_file2.read.return_value = txt_bytes

    # Mock extract_text to isolate testing of extract_texts structure
    with patch("src.core.document_parser.extract_text", side_effect=lambda f, name, **kwargs: f"Parsed {name}"):
        results = extract_texts([mock_file1, mock_file2])

    assert results["doc1.docx"] == "Parsed doc1.docx"
    assert results["doc2.txt"] == "Parsed doc2.txt"
