"""Tests for PDF plagiarism report generation."""

from datetime import datetime, timezone
from io import BytesIO

from PyPDF2 import PdfReader

from utils.pdf_report import (
    generate_plagiarism_report,
    report_filename,
    similarity_severity,
)


def _read_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_generates_valid_pdf_with_required_fields():
    pdf = generate_plagiarism_report(
        doc_a="student_a.pdf",
        doc_b="student_b.pdf",
        similarity=0.934,
        severity="High",
        top_pairs=[
            ("First matching paragraph.", "Second matching paragraph.", 0.96),
        ],
        submission_date_a="2026-07-12",
        submission_date_b="2026-07-13",
        database_id_a=101,
        database_id_b=102,
        generated_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1500

    text = _read_text(pdf)
    assert "student_a.pdf" in text
    assert "student_b.pdf" in text
    assert "93.4%" in text
    assert "High" in text
    assert "2026-07-12" in text
    assert "101" in text
    assert "First matching paragraph" in text


def test_long_paragraphs_generate_multiple_pages_without_layout_error():
    long_left = " ".join(["semantic comparison evidence"] * 1800)
    long_right = " ".join(["paraphrased assignment passage"] * 1800)

    pdf = generate_plagiarism_report(
        doc_a="very-long-left.pdf",
        doc_b="very-long-right.pdf",
        similarity=0.91,
        top_pairs=[
            (long_left, long_right, 0.97),
            (long_left, long_right, 0.95),
            (long_left, long_right, 0.93),
        ],
    )

    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) >= 2
    assert pdf.startswith(b"%PDF")


def test_only_top_three_matches_are_included():
    pairs = [
        (f"left-{index}", f"right-{index}", 0.99 - index / 100) for index in range(5)
    ]

    text = _read_text(
        generate_plagiarism_report(
            doc_a="a.pdf",
            doc_b="b.pdf",
            similarity=0.95,
            top_pairs=pairs,
        )
    )

    assert "left-0" in text
    assert "left-2" in text
    assert "left-3" not in text


def test_missing_metadata_is_reported_honestly():
    text = _read_text(
        generate_plagiarism_report(
            doc_a="a.pdf",
            doc_b="b.pdf",
            similarity=0.70,
            top_pairs=[],
        )
    )

    assert "Not available" in text
    assert "No paragraph pairs were available" in text


def test_filename_is_windows_safe():
    filename = report_filename("A: invalid?.pdf", "B/other*.pdf")

    assert filename.endswith(".pdf")
    assert ":" not in filename
    assert "?" not in filename
    assert "/" not in filename
    assert "*" not in filename


def test_severity_mapping():
    assert similarity_severity(0.95) == "High"
    assert similarity_severity(0.70, threshold=0.59) == "Medium"
    assert similarity_severity(0.40, threshold=0.59) == "Below warning threshold"
