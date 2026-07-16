"""Document text extraction with OCR fallback for scanned PDF pages."""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Dict, List, Union

from langdetect import LangDetectException, detect
from src.core.translator import translate_text
import docx
import pdfplumber

# OCR dependencies are imported lazily so TXT/DOCX and normal text PDFs still
# work even when Tesseract is not installed on the machine.
PDFInput = Union[str, bytes, io.BytesIO, BinaryIO]

MIN_NATIVE_WORDS_PER_PAGE = 8
DEFAULT_OCR_DPI = 250

def detect_text_language(text: str) -> str:
    """
    Detect the language of a text chunk.

    Returns language codes such as:
    en, fr, hi, es, de, etc.
    """
    cleaned_text = text.strip()

    if len(cleaned_text) < 20:
        return "unknown"

    try:
        return detect(cleaned_text)
    except LangDetectException:
        return "unknown"


def prepare_text_for_embedding(text: str) -> dict:
    """
    Preserve the original text and prepare English text for embeddings.
    """
    original_text = text.strip()
    detected_language = detect_text_language(original_text)

    translated_text = original_text
    was_translated = False

    if detected_language not in {"en", "unknown"}:
        translated_result = translate_text(
            original_text,
            target_lang="en",
        )

        if translated_result and not translated_result.startswith(
            "(Translation Error:"
        ):
            translated_text = translated_result
            was_translated = True

    return {
        "original_text": original_text,
        "embedding_text": translated_text,
        "detected_language": detected_language,
        "was_translated": was_translated,
    }


class OCRDependencyError(RuntimeError):
    """Raised when OCR is required but its dependencies are unavailable."""


def _is_page_number(line: str) -> bool:
    """Return True for simple standalone page-number lines."""
    cleaned = re.sub(r"[\u00a0\u200b]", " ", line).strip()
    if not cleaned:
        return False
    return bool(
        re.fullmatch(r"(?:page|p\.?)?\s*\d+", cleaned, flags=re.IGNORECASE)
    ) or bool(re.fullmatch(r"\d{1,3}", cleaned))


def _clean_page_text(page_text: str) -> List[str]:
    """Clean one page of extracted text."""
    lines: List[str] = []
    for raw_line in page_text.splitlines():
        cleaned = re.sub(r"[\u00a0\u200b]", " ", raw_line).strip()
        if not cleaned or _is_page_number(cleaned):
            continue
        lines.append(cleaned)
    return lines


def _remove_repeated_boundary_lines(
    page_lines: List[List[str]],
) -> List[List[str]]:
    """Remove repeated first/last lines, typically headers and footers."""
    if not page_lines:
        return []

    cleaned_pages = [list(lines) for lines in page_lines]

    for position in ("start", "end"):
        candidates: List[str] = []
        for lines in cleaned_pages:
            if not lines:
                continue
            candidates.append(lines[0] if position == "start" else lines[-1])

        counts = Counter(candidates)
        repeated = {
            line
            for line, count in counts.items()
            if count > 1 and len(line) <= 60 and not _is_page_number(line)
        }

        for index, lines in enumerate(cleaned_pages):
            if not lines:
                continue
            if position == "start" and lines[0] in repeated:
                cleaned_pages[index] = lines[1:]
            elif position == "end" and lines[-1] in repeated:
                cleaned_pages[index] = lines[:-1]

    return cleaned_pages


def _normalize_whitespace(page_lines: List[List[str]]) -> str:
    """Join cleaned lines and collapse excessive whitespace."""
    cleaned_lines = [line for lines in page_lines for line in lines]
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _read_pdf_bytes(file: PDFInput) -> bytes:
    """Return PDF content without leaving a supplied stream at a new position."""
    if isinstance(file, bytes):
        return file

    if isinstance(file, str):
        return Path(file).read_bytes()

    position = None
    if hasattr(file, "tell"):
        try:
            position = file.tell()
        except (OSError, ValueError):
            position = None

    data = file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")

    if position is not None and hasattr(file, "seek"):
        try:
            file.seek(position)
        except (OSError, ValueError):
            pass

    return data


def _has_meaningful_text(text: str) -> bool:
    """Decide whether native extraction returned enough useful text."""
    words = re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)
    alphanumeric_chars = sum(char.isalnum() for char in text or "")
    return (
        len(words) >= MIN_NATIVE_WORDS_PER_PAGE
        and alphanumeric_chars >= 30
    )


def _configure_tesseract(pytesseract_module) -> None:
    """Use an optional explicit Tesseract path on Windows or other systems."""
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract_module.pytesseract.tesseract_cmd = configured_path


def _ocr_pdf_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    language: str = "eng",
) -> str:
    """Render one PDF page and extract text with Tesseract."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRDependencyError(
            "OCR dependencies are missing. Install pytesseract, PyMuPDF and "
            "Pillow using: python -m pip install pytesseract pymupdf pillow"
        ) from exc

    _configure_tesseract(pytesseract)

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page = document.load_page(page_index)
            scale = dpi / 72
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            return pytesseract.image_to_string(
                image,
                lang=language,
                config="--oem 3 --psm 3",
            ).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRDependencyError(
            "Tesseract OCR was not found. Install Tesseract and either add it "
            "to PATH or set TESSERACT_CMD to tesseract.exe."
        ) from exc


def extract_text_from_pdf(
    file: PDFInput,
    *,
    ocr_language: str = "eng",
    ocr_dpi: int = DEFAULT_OCR_DPI,
) -> str:
    """Extract PDF text and OCR only pages with insufficient native text.

    Text-based PDFs continue to use pdfplumber. Fully scanned and mixed PDFs
    are handled page by page, allowing OCR results to enter the unchanged
    chunking, embedding and FAISS pipeline.
    """
    pdf_bytes = _read_pdf_bytes(file)
    page_lines: List[List[str]] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                native_text = (page.extract_text() or "").strip()
                selected_text = native_text

                if not _has_meaningful_text(native_text):
                    selected_text = _ocr_pdf_page(
                        pdf_bytes,
                        page_index,
                        dpi=ocr_dpi,
                        language=ocr_language,
                    )

                if selected_text.strip():
                    page_lines.append(_clean_page_text(selected_text))
    except OCRDependencyError:
        # Preserve a clear, actionable message for Streamlit and callers.
        raise
    except Exception as exc:
        print(f"[document_parser] Error reading PDF: {exc}")
        return ""

    if not page_lines:
        return ""

    cleaned_pages = _remove_repeated_boundary_lines(page_lines)
    return _normalize_whitespace(cleaned_pages)


def extract_text_from_docx(file: PDFInput) -> str:
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc_file = io.BytesIO(file) if isinstance(file, bytes) else file
        document = docx.Document(doc_file)
        text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        print(f"[document_parser] Error reading DOCX: {exc}")
    return text.strip()


def extract_text_from_txt(file: PDFInput) -> str:
    """Extract text from a TXT file with UTF-8 fallback."""
    text = ""
    try:
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        elif isinstance(file, bytes):
            text = file.decode("utf-8", errors="ignore")
        else:
            data = file.read()
            text = (
                data.decode("utf-8", errors="ignore")
                if isinstance(data, bytes)
                else data
            )
    except Exception as exc:
        print(f"[document_parser] Error reading TXT: {exc}")
    return text.strip()


def extract_text(file: PDFInput, filename: str) -> str:
    """Route extraction according to a filename extension."""
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(file)
    if extension == "docx":
        return extract_text_from_docx(file)
    return extract_text_from_txt(file)


def extract_texts_from_pdfs(files: list) -> Dict[str, str]:
    """Legacy compatibility wrapper."""
    return extract_texts(files)


def extract_texts(files: list) -> Dict[str, str]:
    """Extract text from multiple uploaded files."""
    results: Dict[str, str] = {}

    for file in files:
        if hasattr(file, "name"):
            name = file.name
        elif isinstance(file, str):
            name = Path(file).name
        else:
            name = f"document_{len(results) + 1}"

        results[name] = extract_text(file, name)

    return results
