"""
document_parser.py
------------------
Handles extraction of raw text from uploaded PDF, DOCX, and TXT files.
Supports both file paths and file-like objects.
"""

import io
import re
import pdfplumber
import docx
from collections import Counter
from typing import Union, List, Dict


def _is_page_number(line: str) -> bool:
    """Return True for simple standalone page-number lines."""
    cleaned = re.sub(r"[\u00a0\u200b]", " ", line).strip()
    if not cleaned:
        return False

    return bool(re.fullmatch(r"(?:page|p\.? )?\s*\d+", cleaned, flags=re.IGNORECASE)) or bool(
        re.fullmatch(r"\d{1,3}", cleaned)
    )


def _clean_page_text(page_text: str) -> List[str]:
    """Clean one page of extracted text by removing page numbers and repeated boundary text."""
    lines = []
    for raw_line in page_text.splitlines():
        cleaned = re.sub(r"[\u00a0\u200b]", " ", raw_line).strip()
        if not cleaned:
            continue
        if _is_page_number(cleaned):
            continue
        lines.append(cleaned)

    return lines


def _remove_repeated_boundary_lines(page_lines: List[List[str]]) -> List[List[str]]:
    """Remove repeated first/last lines that appear across pages, typically headers/footers."""
    if not page_lines:
        return []

    cleaned_pages = [list(lines) for lines in page_lines]
    for position in ("start", "end"):
        candidates = []
        for lines in cleaned_pages:
            if not lines:
                continue
            candidate = lines[0] if position == "start" else lines[-1]
            candidates.append(candidate)

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


def extract_text_from_pdf(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract all text from a PDF file using pdfplumber for robust layout handling.
    """
    page_lines = []
    try:
        if isinstance(file, str):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        page_lines.append(_clean_page_text(extracted))
        elif isinstance(file, bytes):
            with pdfplumber.open(io.BytesIO(file)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        page_lines.append(_clean_page_text(extracted))
        else:
            # Assume it is already a file-like object
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        page_lines.append(_clean_page_text(extracted))
    except Exception as e:
        print(f"[document_parser] Error reading PDF: {e}")

    if not page_lines:
        return ""

    cleaned_pages = _remove_repeated_boundary_lines(page_lines)
    return _normalize_whitespace(cleaned_pages)


def extract_text_from_docx(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text from a DOCX file using python-docx.
    """
    text = ""
    try:
        if isinstance(file, bytes):
            doc_file = io.BytesIO(file)
        else:
            doc_file = file
        
        doc = docx.Document(doc_file)
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[document_parser] Error reading DOCX: {e}")
    
    return text.strip()


def extract_text_from_txt(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text from a TXT file with UTF-8 decoding fallback.
    """
    text = ""
    try:
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif isinstance(file, bytes):
            text = file.decode("utf-8", errors="ignore")
        else:
            data = file.read()
            if isinstance(data, bytes):
                text = data.decode("utf-8", errors="ignore")
            else:
                text = data
    except Exception as e:
        print(f"[document_parser] Error reading TXT: {e}")
    
    return text.strip()


def extract_text(file: Union[str, bytes, io.BytesIO], filename: str) -> str:
    """
    Unified text extraction function routing based on filename extension.
    """
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    else:
        return extract_text_from_txt(file)


def extract_texts_from_pdfs(files: list) -> dict:
    """
    Legacy compatibility function for extracting multiple PDF/document files.
    """
    return extract_texts(files)


def extract_texts(files: list) -> dict:
    """
    Extract text from multiple files (PDF, DOCX, TXT).
    """
    results = {}
    for file in files:
        if hasattr(file, "name"):
            name = file.name
        elif isinstance(file, str):
            name = file.split("/")[-1].split("\\")[-1]
        else:
            name = f"document_{len(results) + 1}"

        results[name] = extract_text(file, name)
    return results
