"""
pdf_reader.py
-------------
Handles extraction of raw text from uploaded PDF files using PyPDF2.
Supports both file paths and file-like objects (e.g., Streamlit UploadedFile).
"""

import PyPDF2
import io
from typing import Union


def extract_text_from_pdf(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file: A file path (str), raw bytes, or a file-like object (BytesIO / UploadedFile).

    Returns:
        A single string containing all extracted text from the PDF.
        Returns an empty string if extraction fails.
    """
    text = ""

    try:
        # Normalise input → always work with a file-like object
        if isinstance(file, str):
            pdf_file = open(file, "rb")
            close_after = True
        elif isinstance(file, bytes):
            pdf_file = io.BytesIO(file)
            close_after = False
        else:
            # Assume it is already a file-like object (BytesIO, UploadedFile …)
            pdf_file = file
            close_after = False

        reader = PyPDF2.PdfReader(pdf_file)

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if close_after:
            pdf_file.close()

    except Exception as e:
        print(f"[pdf_reader] Error reading PDF: {e}")

    return text.strip()


def extract_texts_from_pdfs(files: list) -> dict:
    """
    Extract text from multiple PDF files.

    Args:
        files: List of file paths or file-like objects.

    Returns:
        A dict mapping file name → extracted text string.
    """
    results = {}

    for file in files:
        # Determine a display name for the file
        if hasattr(file, "name"):
            name = file.name  # Streamlit UploadedFile exposes .name
        elif isinstance(file, str):
            name = file.split("/")[-1]
        else:
            name = f"document_{len(results) + 1}.pdf"

        results[name] = extract_text_from_pdf(file)

    return results
