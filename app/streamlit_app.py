# ruff: noqa: E402

import hashlib
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io as _io
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from app.theme import (
    empty_state_html,
    format_similarity_html,
    get_colors,
    get_theme_name,
    inject_css,
    pipeline_progress_html,
    set_theme,
    sidebar_user_badge_html,
)
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from src.utils.warning_list import render_warning_controls
from src.core.text_chunking import chunk_documents
from src.core.embedding_model import embed_documents
from src.core.similarity import (
    document_similarity_matrix,
    flag_plagiarism,
    find_most_similar_chunks,
    PLAGIARISM_THRESHOLD,
)
from src.core.faiss_index import (
    build_index,
    find_plagiarised_chunks,
    search_similar_chunks,
    save_index,
    load_index,
    build_index_from_matrix,
)
from src.core.webhook import send_plagiarism_alert
from src.core.ai_detector import detect_documents_ai_probability
from src.visualization.network_graph import plot_similarity_network
from src.db import (
    init_corpus_db,
    get_all_documents,
    delete_document,
    get_all_embeddings,
    get_chunk_registry,
    add_document,
    get_document_by_hash,
    add_chunks,
    get_unique_class_sections,
    get_documents_by_class,
)
from src.utils.pdf_report import generate_plagiarism_report
from src.utils.badge_generator import (
    generate_badge_png,
    generate_badge_pdf,
)
from src.utils.redis_cache import (
    cache_session_state,
    get_session_state,
    clear_session,
    cache_faiss_index,
    get_faiss_index,
    cache_analysis_results,
    get_analysis_results,
)
from src.visualization.heatmap import (
    plot_chunk_similarity_comparison,
    plot_similarity_heatmap,
)
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    OCRDependencyError,
    SUPPORTED_OCR_LANGUAGES,
    extract_text,
    prepare_text_for_embedding,
)
from src.db.auth import (
    init_db,
    verify_user,
    get_user_role,
    add_user,
    get_all_users,
    delete_user,
    update_password,
    get_tour_completed,
    set_tour_completed,
)

# Initialize corpus database
init_corpus_db()

# Generate unique session ID for this Streamlit session
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)
try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# Initialize database
init_db()

# Must be the first Streamlit command called
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
init_db()

# ── SESSION TIMEOUT & ROUTE PROTECTION MIDDLEWARE ─────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

cached_last_interaction = get_session_state(SESSION_ID, "last_interaction")
if cached_last_interaction is not None:
    last_interaction = cached_last_interaction
elif "last_interaction" in st.session_state:
    last_interaction = st.session_state.last_interaction
else:
    last_interaction = None

if last_interaction and st.session_state.get("authenticated", False):
    elapsed_time = time.time() - last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        st.warning("⏱️ Your session has expired due to 15 minutes of inactivity. Please log in again.")
        st.stop()
    else:
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Log In", use_container_width=True)

        if login_submitted:
            username = username.strip().lower()

            if not username or not password:
                st.error("Please enter both username and password.")
            elif verify_user(username, password):
                role = get_user_role(username)
                if role is not None:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.last_interaction = time.time()
                    st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()

user_role = st.session_state.get("role", "user")

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if user_role == "admin":
        threshold = st.slider(
            "Plagiarism Threshold",
            0.50,
            0.99,
            value=PLAGIARISM_THRESHOLD,
            step=0.01,
            key="threshold_slider",
        )
        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            key="chunk_matrix_checkbox",
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            key="faiss_top_k_slider",
        )

        # ── Customizable Chunk Size & Overlap Sliders (#153) ─────────────────
        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            200,
            2000,
            value=500,
            step=50,
            help="Target character length for text chunks during embedding.",
            key="chunk_size_slider",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            0,
            500,
            value=50,
            step=10,
            help="Character overlap between consecutive chunks to preserve contextual boundary.",
            key="chunk_overlap_slider",
        )

        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI
    else:
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

    unique_classes = ["All Classes"] + get_unique_class_sections()
    selected_class = st.selectbox("Select Class/Section", unique_classes, index=0)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")

uploaded_files = st.file_uploader(
    "📂 Upload Assignments",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader",
)

file_bytes_dict = {f.name: f.getvalue() for f in uploaded_files} if uploaded_files else {}

if len(file_bytes_dict) < 2:
    st.info("Upload at least 2 files to begin analysis.")
    st.stop()

# ── Pipeline Execution ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(
    file_bytes_dict: dict[str, bytes],
    ocr_language: str,
    ocr_dpi: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    raw_texts = {}
    for name, data in file_bytes_dict.items():
        raw_texts[name] = extract_text(
            _io.BytesIO(data),
            name,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )

    chunked_docs = chunk_documents(
        raw_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    translated_chunked_docs = {}

    for doc_name, chunks in chunked_docs.items():
        translated_chunked_docs[doc_name] = []
        for chunk in chunks:
            prepared = prepare_text_for_embedding(chunk)
            translated_chunked_docs[doc_name].append(prepared["embedding_text"])

    embeddings = embed_documents(translated_chunked_docs)
    sim_df = document_similarity_matrix(embeddings)

    names = list(embeddings.keys())
    n = len(names)
    chunk_mat = np.zeros((n, n))

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                chunk_mat[i, j] = 1.0
            elif j > i:
                ea, eb = embeddings[na], embeddings[nb]
                score = float(np.max(cosine_similarity(ea, eb))) if ea.size and eb.size else 0.0
                chunk_mat[i, j] = score
                chunk_mat[j, i] = score

    chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)
    faiss_index, registry = build_index(embeddings, chunked_docs)
    ai_probabilities = detect_documents_ai_probability(chunked_docs)

    return (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
    )

with st.spinner("🧠 Processing files and building embeddings…"):
    analysis_results = run_pipeline(
        file_bytes_dict,
        ocr_language,
        ocr_dpi,
        chunk_size,
        chunk_overlap,
    )

(
    raw_texts,
    chunked_docs,
    embeddings,
    sim_df,
    chunk_sim_df,
    faiss_index,
    registry,
    ai_probabilities,
) = analysis_results

active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
flags = flag_plagiarism(active_sim_df, threshold=threshold)

st.subheader("📊 Analysis Summary")
st.write(f"Processed **{len(raw_texts)}** documents with Chunk Size: `{chunk_size}` and Overlap: `{chunk_overlap}`.")