"""Formal PDF report generation for selected plagiarism pairs."""

from __future__ import annotations

import html
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_TITLE = "Semantic Plagiarism Analysis Report"
MAX_MATCHES = 3
TEXT_SEGMENT_CHAR_LIMIT = 900


def _safe_text(value: Any, fallback: str = "Not available") -> str:
    """Return escaped printable text suitable for a ReportLab Paragraph."""
    if value is None:
        return html.escape(fallback)

    text = str(value).strip()
    if not text:
        text = fallback

    # Remove control characters that can break PDF rendering.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.escape(text)


def _normalise_similarity(value: Any) -> float:
    """Normalise similarity into the inclusive [0, 1] range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(1.0, max(0.0, number))


def similarity_severity(similarity: float, threshold: float = 0.59) -> str:
    """Return the same simple severity convention used by the dashboard."""
    score = _normalise_similarity(similarity)
    if score >= 0.90:
        return "High"
    if score >= threshold:
        return "Medium"
    return "Below warning threshold"


def _normalise_match(match: Any) -> dict[str, Any]:
    """Support tuple-based and dictionary-based matching paragraph records."""
    if isinstance(match, Mapping):
        left = (
            match.get("source_chunk_text")
            or match.get("text_a")
            or match.get("chunk_a")
            or match.get("left")
            or ""
        )
        right = (
            match.get("match_chunk_text")
            or match.get("text_b")
            or match.get("chunk_b")
            or match.get("right")
            or ""
        )
        similarity = match.get("similarity", match.get("score", 0.0))
    elif isinstance(match, Sequence) and not isinstance(match, (str, bytes)):
        left = match[0] if len(match) > 0 else ""
        right = match[1] if len(match) > 1 else ""
        similarity = match[2] if len(match) > 2 else 0.0
    else:
        left, right, similarity = str(match), "", 0.0

    return {
        "text_a": str(left or ""),
        "text_b": str(right or ""),
        "similarity": _normalise_similarity(similarity),
    }


def _split_long_text(text: str, limit: int = TEXT_SEGMENT_CHAR_LIMIT) -> list[str]:
    """Split long paragraph text at word boundaries for safe table pagination."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ["No matching paragraph text available."]

    words = cleaned.split(" ")
    segments: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > limit:
            segments.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += extra

    if current:
        segments.append(" ".join(current))

    return segments or ["No matching paragraph text available."]


def _filename_for_report(doc_a: str, doc_b: str) -> str:
    """Build a Windows-safe download filename."""
    stem_a = Path(str(doc_a)).stem or "document-a"
    stem_b = Path(str(doc_b)).stem or "document-b"
    combined = f"plagiarism_report_{stem_a}_vs_{stem_b}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", combined).strip("._")
    return f"{safe[:140] or 'plagiarism_report'}.pdf"


def report_filename(doc_a: str, doc_b: str) -> str:
    """Public helper for a safe PDF download filename."""
    return _filename_for_report(doc_a, doc_b)


def _header_footer(canvas, doc) -> None:
    """Draw page number and generation label on every page."""
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D9DEE8"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(18 * mm, 9.5 * mm, "Semantic Plagiarism Detector")
    canvas.drawRightString(
        width - 18 * mm,
        9.5 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def generate_plagiarism_report(
    *,
    doc_a: str,
    doc_b: str,
    similarity: float,
    severity: str | None = None,
    top_pairs: Iterable[Any] | None = None,
    submission_date_a: str | None = None,
    submission_date_b: str | None = None,
    database_id_a: str | int | None = None,
    database_id_b: str | int | None = None,
    threshold: float = 0.59,
    generated_at: datetime | None = None,
) -> bytes:
    """Generate a printable PDF report and return its bytes.

    Long matching paragraphs are split into table-safe segments so they wrap and
    continue across pages without clipping or horizontal overflow.
    """
    score = _normalise_similarity(similarity)
    severity_text = severity or similarity_severity(score, threshold)
    matches = [_normalise_match(item) for item in (top_pairs or [])]
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    matches = matches[:MAX_MATCHES]

    generated_at = generated_at or datetime.now(timezone.utc)

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=REPORT_TITLE,
        author="Semantic Plagiarism Detector",
        subject=f"Comparison of {doc_a} and {doc_b}",
        allowSplitting=True,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#172B4D"),
        spaceAfter=5 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#667085"),
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#172B4D"),
        spaceBefore=4 * mm,
        spaceAfter=2.5 * mm,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#263238"),
        splitLongWords=True,
        wordWrap="LTR",
    )
    small_style = ParagraphStyle(
        "SmallText",
        parent=body_style,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#475467"),
    )
    label_style = ParagraphStyle(
        "LabelText",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#344054"),
    )
    match_heading_style = ParagraphStyle(
        "MatchHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#7A1F3D"),
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )

    story = [
        Paragraph(REPORT_TITLE, title_style),
        Paragraph(
            f"Generated {generated_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        ),
    ]

    severity_color = (
        colors.HexColor("#B42318")
        if "high" in severity_text.lower()
        else colors.HexColor("#B54708")
        if "medium" in severity_text.lower()
        else colors.HexColor("#027A48")
    )

    summary_data = [
        [
            Paragraph("<b>Overall similarity</b>", label_style),
            Paragraph(f"<b>{score * 100:.1f}%</b>", body_style),
            Paragraph("<b>Warning status</b>", label_style),
            Paragraph(f"<b>{_safe_text(severity_text)}</b>", body_style),
        ]
    ]
    summary_table = Table(
        summary_data,
        colWidths=[36 * mm, 32 * mm, 36 * mm, 50 * mm],
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E4E7EC")),
                ("TEXTCOLOR", (3, 0), (3, 0), severity_color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 4 * mm)])

    story.append(Paragraph("Compared submissions", section_style))

    metadata_rows = [
        [
            Paragraph("<b>Field</b>", label_style),
            Paragraph(f"<b>{_safe_text(doc_a)}</b>", body_style),
            Paragraph(f"<b>{_safe_text(doc_b)}</b>", body_style),
        ],
        [
            Paragraph("Filename / student", label_style),
            Paragraph(_safe_text(doc_a), body_style),
            Paragraph(_safe_text(doc_b), body_style),
        ],
        [
            Paragraph("Submission date", label_style),
            Paragraph(_safe_text(submission_date_a), body_style),
            Paragraph(_safe_text(submission_date_b), body_style),
        ],
        [
            Paragraph("Database ID", label_style),
            Paragraph(_safe_text(database_id_a), body_style),
            Paragraph(_safe_text(database_id_b), body_style),
        ],
    ]

    metadata_table = Table(
        metadata_rows,
        colWidths=[34 * mm, 61 * mm, 61 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D0D5DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E4E7EC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metadata_table, Spacer(1, 4 * mm)])

    story.append(Paragraph("Top matching paragraph blocks", section_style))

    if not matches:
        story.append(
            Paragraph(
                "No paragraph pairs were available above the selected threshold.",
                body_style,
            )
        )
    else:
        for index, match in enumerate(matches, start=1):
            heading = (
                f"Match {index} - paragraph similarity {match['similarity'] * 100:.1f}%"
            )
            story.append(Paragraph(heading, match_heading_style))

            left_segments = _split_long_text(match["text_a"])
            right_segments = _split_long_text(match["text_b"])
            row_count = max(len(left_segments), len(right_segments))

            comparison_rows = [
                [
                    Paragraph(f"<b>{_safe_text(doc_a)}</b>", body_style),
                    Paragraph(f"<b>{_safe_text(doc_b)}</b>", body_style),
                ]
            ]

            for row_index in range(row_count):
                left_text = (
                    left_segments[row_index] if row_index < len(left_segments) else ""
                )
                right_text = (
                    right_segments[row_index] if row_index < len(right_segments) else ""
                )
                comparison_rows.append(
                    [
                        Paragraph(_safe_text(left_text, ""), body_style),
                        Paragraph(_safe_text(right_text, ""), body_style),
                    ]
                )

            comparison_table = Table(
                comparison_rows,
                colWidths=[78 * mm, 78 * mm],
                repeatRows=1,
                hAlign="LEFT",
                splitByRow=True,
            )
            comparison_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FCE7F3")),
                        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D0D5DD")),
                        (
                            "INNERGRID",
                            (0, 0),
                            (-1, -1),
                            0.35,
                            colors.HexColor("#E4E7EC"),
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([comparison_table, Spacer(1, 3 * mm)])

    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph("Interpretation notice", section_style),
            Paragraph(
                (
                    "This report highlights semantic similarity for review. "
                    "Similarity alone does not prove academic misconduct. "
                    "A teacher or authorised reviewer should inspect the source "
                    "material, citations, assignment context, and institutional policy "
                    "before reaching a conclusion."
                ),
                small_style,
            ),
        ]
    )

    document.build(
        story,
        onFirstPage=_header_footer,
        onLaterPages=_header_footer,
    )

    return buffer.getvalue()
