"""Search, multi-column sorting, and pagination for plagiarism warnings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st


SORT_FIELDS = {
    "Similarity": "similarity",
    "Document A": "doc_a",
    "Document B": "doc_b",
    "Severity": "severity_rank",
}


@dataclass(frozen=True)
class WarningPage:
    items: list[dict[str, Any]]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    start_index: int
    end_index: int


def _normalise_warning(warning: Mapping[str, Any]) -> dict[str, Any]:
    severity = str(warning.get("severity", "")).strip()
    severity_key = severity.lower()

    if "high" in severity_key:
        severity_rank = 2
    elif "medium" in severity_key:
        severity_rank = 1
    else:
        severity_rank = 0

    try:
        similarity = float(warning.get("similarity", 0.0))
    except (TypeError, ValueError):
        similarity = 0.0

    return {
        **dict(warning),
        "doc_a": str(warning.get("doc_a", "")).strip(),
        "doc_b": str(warning.get("doc_b", "")).strip(),
        "similarity": similarity,
        "severity": severity,
        "severity_rank": severity_rank,
    }


def filter_warnings(
    warnings: Iterable[Mapping[str, Any]],
    search_query: str = "",
) -> list[dict[str, Any]]:
    normalised = [_normalise_warning(item) for item in warnings]
    query = search_query.strip().casefold()

    if not query:
        return normalised

    return [
        item
        for item in normalised
        if query in item["doc_a"].casefold()
        or query in item["doc_b"].casefold()
    ]


def sort_warnings(
    warnings: Iterable[Mapping[str, Any]],
    *,
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
) -> list[dict[str, Any]]:
    items = [_normalise_warning(item) for item in warnings]
    allowed = {"similarity", "doc_a", "doc_b", "severity_rank"}

    if primary_field not in allowed:
        primary_field = "similarity"
    if secondary_field not in allowed:
        secondary_field = "doc_a"

    def key_for(field: str):
        def key(item: Mapping[str, Any]):
            value = item[field]
            return value.casefold() if isinstance(value, str) else value
        return key

    items.sort(key=key_for(secondary_field), reverse=secondary_descending)
    items.sort(key=key_for(primary_field), reverse=primary_descending)
    return items


def paginate_warnings(
    warnings: Sequence[Mapping[str, Any]],
    *,
    page: int = 1,
    page_size: int = 10,
) -> WarningPage:
    safe_page_size = max(1, int(page_size))
    total_items = len(warnings)
    total_pages = max(1, math.ceil(total_items / safe_page_size))
    safe_page = min(max(1, int(page)), total_pages)

    start = (safe_page - 1) * safe_page_size
    end = min(start + safe_page_size, total_items)

    return WarningPage(
        items=[dict(item) for item in warnings[start:end]],
        total_items=total_items,
        page=safe_page,
        page_size=safe_page_size,
        total_pages=total_pages,
        start_index=start + 1 if total_items else 0,
        end_index=end,
    )


def prepare_warning_page(
    warnings: Iterable[Mapping[str, Any]],
    *,
    search_query: str = "",
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[dict[str, Any]], WarningPage]:
    filtered = filter_warnings(warnings, search_query)
    sorted_items = sort_warnings(
        filtered,
        primary_field=primary_field,
        primary_descending=primary_descending,
        secondary_field=secondary_field,
        secondary_descending=secondary_descending,
    )
    return sorted_items, paginate_warnings(
        sorted_items,
        page=page,
        page_size=page_size,
    )


def _reset_page() -> None:
    st.session_state.warning_page = 1


def _severity_badge(severity: str) -> tuple[str, str]:
    value = severity.lower()
    if "high" in value:
        return "#ff4b4b", severity or "High"
    if "medium" in value:
        return "#ffa500", severity or "Medium"
    return "#6c757d", severity or "Low"


def render_warning_controls(
    flags: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> None:
    if "warning_page" not in st.session_state:
        st.session_state.warning_page = 1

    st.caption(f"Pairs with similarity ≥ **{threshold:.2f}**")

    if not flags:
        st.success("✅ No suspicious pairs found above the current threshold.")
        return

    search_col, size_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input(
            "Search warnings",
            placeholder="Search by either document name…",
            key="warning_search",
            on_change=_reset_page,
        )
    with size_col:
        page_size = st.selectbox(
            "Warnings per page",
            [10, 25, 50],
            key="warning_page_size",
            on_change=_reset_page,
        )

    p1, d1, p2, d2 = st.columns([2, 1, 2, 1])
    with p1:
        primary_label = st.selectbox(
            "Primary sort",
            list(SORT_FIELDS),
            key="warning_primary_sort",
            on_change=_reset_page,
        )
    with d1:
        primary_direction = st.selectbox(
            "Direction",
            ["Descending", "Ascending"],
            key="warning_primary_direction",
            on_change=_reset_page,
        )
    with p2:
        secondary_label = st.selectbox(
            "Then sort by",
            list(SORT_FIELDS),
            index=1,
            key="warning_secondary_sort",
            on_change=_reset_page,
        )
    with d2:
        secondary_direction = st.selectbox(
            "Then direction",
            ["Ascending", "Descending"],
            key="warning_secondary_direction",
            on_change=_reset_page,
        )

    sorted_flags, current_page = prepare_warning_page(
        flags,
        search_query=search_query,
        primary_field=SORT_FIELDS[primary_label],
        primary_descending=primary_direction == "Descending",
        secondary_field=SORT_FIELDS[secondary_label],
        secondary_descending=secondary_direction == "Descending",
        page=st.session_state.warning_page,
        page_size=page_size,
    )

    if current_page.page != st.session_state.warning_page:
        st.session_state.warning_page = current_page.page

    export_df = pd.DataFrame([
        {
            "Document A": item["doc_a"],
            "Document B": item["doc_b"],
            "Similarity": item["similarity"],
            "Severity": item["severity"],
        }
        for item in sorted_flags
    ])

    left, right = st.columns([3, 2])
    with left:
        if current_page.total_items:
            st.markdown(
                f"Showing **{current_page.start_index}–{current_page.end_index}** "
                f"of **{current_page.total_items}** matching warnings"
            )
        else:
            st.info("No warnings match the current search.")
    with right:
        st.download_button(
            "⬇️ Download filtered report (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            "plagiarism_warnings_filtered.csv",
            "text/csv",
            use_container_width=True,
            disabled=export_df.empty,
        )

    for flag in current_page.items:
        color, severity_text = _severity_badge(flag["severity"])
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{flag['doc_a']}** ↔ **{flag['doc_b']}**")
                st.progress(
                    min(1.0, max(0.0, float(flag["similarity"]))),
                    text=f"Similarity: {flag['similarity'] * 100:.1f}%",
                )
            with c2:
                st.markdown(
                    (
                        "<div style='text-align:center;padding:8px;"
                        f"border-radius:8px;background:{color};color:white;"
                        "font-weight:bold;'>"
                        f"{severity_text}</div>"
                    ),
                    unsafe_allow_html=True,
                )

    if current_page.total_items == 0:
        return

    prev_col, page_col, next_col = st.columns([1, 2, 1])

    with prev_col:
        if st.button(
            "← Previous",
            use_container_width=True,
            disabled=current_page.page <= 1,
            key="warning_previous_page",
        ):
            st.session_state.warning_page = current_page.page - 1
            st.rerun()

    with page_col:
        selected_page = st.selectbox(
            "Page",
            list(range(1, current_page.total_pages + 1)),
            index=current_page.page - 1,
            key=f"warning_page_selector_{current_page.total_pages}",
            format_func=lambda value: f"Page {value} of {current_page.total_pages}",
            label_visibility="collapsed",
        )
        if selected_page != current_page.page:
            st.session_state.warning_page = selected_page
            st.rerun()

    with next_col:
        if st.button(
            "Next →",
            use_container_width=True,
            disabled=current_page.page >= current_page.total_pages,
            key="warning_next_page",
        ):
            st.session_state.warning_page = current_page.page + 1
            st.rerun()
