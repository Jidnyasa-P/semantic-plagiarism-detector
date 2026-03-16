"""
heatmap.py
----------
Generates a styled similarity heatmap using Seaborn and Matplotlib.
Returns a Matplotlib Figure object so it can be rendered inside Streamlit
with st.pyplot() without file I/O.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
from typing import Optional

matplotlib.use("Agg")  # Non-interactive backend (safe for Streamlit / servers)


# ── Colour palette ─────────────────────────────────────────────────────────────
# Green (low similarity) → Yellow → Red (high / plagiarism risk)
_CMAP = sns.diverging_palette(
    h_neg=145, h_pos=10,    # Green → Red
    s=80, l=50, sep=10,
    as_cmap=True
)

PLAGIARISM_THRESHOLD = 0.75


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Semantic Similarity Matrix",
    threshold: float = PLAGIARISM_THRESHOLD,
    figsize: Optional[tuple] = None,
    annotate: bool = True,
) -> Figure:
    """
    Plot a heatmap of the document similarity matrix.

    Args:
        similarity_df:  Square DataFrame (N×N) with similarity scores.
        title:          Plot title string.
        threshold:      Similarity value above which to draw a warning border.
        figsize:        Tuple (width, height) in inches. Auto-sized if None.
        annotate:       Whether to annotate cells with numeric scores.

    Returns:
        Matplotlib Figure object ready for st.pyplot() or savefig().
    """
    n = len(similarity_df)

    # Auto figure size based on matrix dimensions
    if figsize is None:
        cell_size = max(1.2, 6 / n)
        width = max(6, n * cell_size + 2)
        height = max(5, n * cell_size + 1.5)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    # ── Draw heatmap ───────────────────────────────────────────────────────────
    mask = np.zeros_like(similarity_df.values, dtype=bool)  # Show all cells

    sns.heatmap(
        similarity_df,
        ax=ax,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap=_CMAP,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#e0e0e0",
        square=True,
        cbar_kws={
            "label": "Cosine Similarity",
            "shrink": 0.8,
            "pad": 0.02,
        },
        annot_kws={"size": max(7, 14 - n), "weight": "bold"},
        mask=mask,
    )

    # ── Highlight diagonal (self-similarity = 1.0) with a subtle grey ─────────
    for i in range(n):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i, i), 1, 1,
                boxstyle="square,pad=0",
                linewidth=2,
                edgecolor="#555555",
                facecolor="none",
                zorder=3,
            )
        )

    # ── Draw red border around cells exceeding threshold ──────────────────────
    data = similarity_df.values
    for i in range(n):
        for j in range(n):
            if i != j and data[i, j] >= threshold:
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (j, i), 1, 1,
                        boxstyle="square,pad=0",
                        linewidth=2.5,
                        edgecolor="#d62728",   # Red border
                        facecolor="none",
                        zorder=4,
                    )
                )

    # ── Labels & styling ──────────────────────────────────────────────────────
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Documents", fontsize=11, labelpad=10)
    ax.set_ylabel("Documents", fontsize=11, labelpad=10)

    # Rotate x-axis labels for readability
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        fontsize=max(8, 11 - n // 3),
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=max(8, 11 - n // 3),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    red_patch = mpatches.Patch(
        edgecolor="#d62728", facecolor="none", linewidth=2,
        label=f"Potential Plagiarism (≥ {threshold:.0%})"
    )
    ax.legend(
        handles=[red_patch],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_chunk_similarity_comparison(
    doc_a_name: str,
    doc_b_name: str,
    chunks_a: list,
    chunks_b: list,
    sim_matrix: np.ndarray,
) -> Figure:
    """
    Plot a chunk-level similarity heatmap between two specific documents.

    Args:
        doc_a_name:  Name of document A.
        doc_b_name:  Name of document B.
        chunks_a:    List of chunk strings from document A.
        chunks_b:    List of chunk strings from document B.
        sim_matrix:  (Na × Nb) cosine similarity matrix.

    Returns:
        Matplotlib Figure.
    """
    na, nb = sim_matrix.shape

    # Truncate chunk labels for readability
    def short_label(text, max_chars=40):
        return text[:max_chars].strip() + "…" if len(text) > max_chars else text

    row_labels = [f"A{i+1}: {short_label(c)}" for i, c in enumerate(chunks_a)]
    col_labels = [f"B{j+1}: {short_label(c)}" for j, c in enumerate(chunks_b)]

    fig_w = max(8, nb * 1.5)
    fig_h = max(6, na * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        sim_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=_CMAP,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#e0e0e0",
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={"size": 8},
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.7},
    )

    ax.set_title(
        f"Chunk-Level Similarity: {doc_a_name}  vs  {doc_b_name}",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel(f"Chunks from {doc_b_name}", fontsize=10)
    ax.set_ylabel(f"Chunks from {doc_a_name}", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    fig.tight_layout()
    return fig
