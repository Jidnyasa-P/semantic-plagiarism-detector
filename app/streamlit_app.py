import asyncio
import sys

# Silence harmless Windows asyncio Proactor connection lost bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ruff: noqa: E402

import base64
import io as _io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from app.theme import (
    empty_state_html,
    get_colors,
    get_theme_name,
    inject_css,
    set_theme,
)
from src.core.ai_detector import detect_documents_ai_probability
from src.core.config import DEFAULT_THRESHOLDS, severity_key
from src.visualization.heatmap import plot_similarity_heatmap, plot_chunk_similarity_comparison
from src.core.faiss_index import build_index, find_plagiarised_chunks, search_similar_chunks, save_index, load_index, build_index_from_matrix, ChunkRecord
from src.visualization.network_graph import plot_similarity_network
from src.core.webhook import send_plagiarism_alert
from src.db import init_corpus_db, get_all_documents, delete_document, get_all_embeddings, get_chunk_registry, add_document, get_document_by_hash, add_chunks
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    extract_text,
    prepare_text_for_embedding,
)
from src.core.embedding_model import embed_documents
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
)
from src.core.text_chunking import chunk_documents
from src.core.webhook import send_plagiarism_alert
from src.db import (
    clear_all_data,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_documents_by_class,
    get_unique_class_sections,
    init_corpus_db,
)
from src.db.auth import (
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_tour_completed,
    get_user_role,
    init_db,
    set_tour_completed,
    verify_user,
)
from src.utils.pdf_report import highlight_pdf_matches
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
)
from src.utils.warning_list import render_warning_controls
from src.visualization.heatmap import plot_similarity_heatmap

try:
    from src.utils.excel_export import export_similarity_matrix_to_excel
    from src.utils.json_export import export_similarity_matrix_to_json
except ImportError:
    from utils.excel_export import export_similarity_matrix_to_excel
    from utils.json_export import export_similarity_matrix_to_json

# Initialize corpus database
init_corpus_db()

# Generate unique session ID for this Streamlit session
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id


_BRANDING_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_config.json")
)
_BRANDING_LOGO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_logo.png")
)
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)
try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# Initialize auth database
init_db()

# Page Configuration
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

init_db()


st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)
# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds


# 1. Handle Automatic Session Expiration (Inactivity Check)
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
        from src.errors import UI_SESSION_EXPIRED
        st.warning(UI_SESSION_EXPIRED)
        st.stop()
    else:
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    if st.session_state.get("pending_2fa", False):
        with st.form("otp_form"):
            st.subheader("🔒 Two-Factor Authentication")
            st.info(
                "Enter the 6-digit verification token from your Google Authenticator/Authy app."
            )
            otp_code = st.text_input(
                "Verification Code", max_chars=6, key="login_otp_code"
            )
            col1, col2 = st.columns(2)
            with col1:
                verify_submitted = st.form_submit_button(
                    "Verify", use_container_width=True
                )
            with col2:
                cancel_submitted = st.form_submit_button(
                    "Cancel", use_container_width=True
                )

            if verify_submitted:
                username = st.session_state.get("pending_username")
                enabled, otp_secret = get_2fa_status(username)
                if enabled and otp_secret:
                    import pyotp

                    totp = pyotp.TOTP(otp_secret)
                    if totp.verify(otp_code.strip()):
                        role = st.session_state.get("pending_role")
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.session_state.last_interaction = time.time()

                        cache_session_state(SESSION_ID, "authenticated", True)
                        cache_session_state(SESSION_ID, "username", username)
                        cache_session_state(SESSION_ID, "role", role)
                        cache_session_state(SESSION_ID, "last_interaction", time.time())

                        # Clear pending state
                        del st.session_state["pending_2fa"]
                        del st.session_state["pending_username"]
                        del st.session_state["pending_role"]

                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid verification code. Please try again.")
                else:
                    st.error("2FA configuration error. Please contact admin.")

            if cancel_submitted:
                del st.session_state["pending_2fa"]
                del st.session_state["pending_username"]
                del st.session_state["pending_role"]
                st.rerun()
        st.stop()

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin")
        login_submitted = st.form_submit_button("Log In", use_container_width=True)

        if login_submitted:
            username = username.strip().lower()

            if not username or not password:
                from src.errors import AUTH_BLANK_CREDENTIALS
                st.error(AUTH_BLANK_CREDENTIALS)
            elif verify_user(username, password):
                role = get_user_role(username)
                if role is None:
                    from src.errors import AUTH_ROLE_UNDETERMINED
                    st.error(AUTH_ROLE_UNDETERMINED)
                else:
                    # Check if 2FA is enabled for this user
                    enabled, _ = get_2fa_status(username)
                    if enabled:
                        st.session_state.pending_2fa = True
                        st.session_state.pending_username = username
                        st.session_state.pending_role = role
                        st.rerun()
                    else:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.session_state.last_interaction = time.time()
                        cache_session_state(SESSION_ID, "authenticated", True)
                        cache_session_state(SESSION_ID, "username", username)
                        cache_session_state(SESSION_ID, "role", role)
                        cache_session_state(SESSION_ID, "last_interaction", time.time())

                        st.success(f"Welcome back, {username}!")
                        st.rerun()
            else:
                from src.errors import AUTH_INVALID_CREDENTIALS
                st.error(AUTH_INVALID_CREDENTIALS)
    st.stop()


# Active user role

user_role = st.session_state.get("role", "user")


@st.dialog("⚠️ Confirm Bulk Clear")
def clear_all_dialog():
    st.markdown(
        "**WARNING:** This action is destructive and cannot be undone. "
        "This will permanently delete all student documents, paragraph chunks, "
        "and plagiarism incidents from the database, and reset the FAISS index."
    )
    st.write("Are you absolutely sure you want to proceed?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_clear_all"):
            st.rerun()
    with col2:
        if st.button(
            "Clear All",
            type="primary",
            use_container_width=True,
            key="confirm_clear_all",
        ):
            # 1. Clear database tables (documents, chunks, incidents)
            clear_all_data()

            # 2. Clear/reset FAISS index file on disk
            if os.path.exists(_INDEX_PATH):
                try:
                    os.remove(_INDEX_PATH)
                except Exception as e:
                    print(f"Error removing FAISS index: {e}")

            # 3. Invalidate Redis cache
            try:
                from src.utils.redis_cache import get_cache

                cache = get_cache()
                if cache.is_available():
                    cache.delete("faiss:index:corpus_index")
                    cache.clear_pattern("analysis:*")
            except Exception as e:
                print(f"Error invalidating cache: {e}")

            # 4. Invalidate Session State cache
            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None
            if "analysis_file_signature" in st.session_state:
                st.session_state.analysis_file_signature = None

            st.success("All documents, chunks, and incidents have been cleared.")
            st.rerun()


# ── Top-right Theme Toggle ───────────────────────────────────────────────────
current_theme = get_theme_name()

# Create a narrow right-aligned column for the theme toggle
_, theme_col = st.columns([0.94, 0.06])

with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"

    if st.button(
        theme_icon,
        key="theme_toggle",
    ):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()


# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_theme = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if current_theme == "Light" else 1,
        horizontal=True,
        key="theme_selector",
    )
    if selected_theme != current_theme:
        set_theme(selected_theme)
        st.rerun()

    if user_role == "admin":
        threshold = st.slider(
            "Plagiarism Threshold",
            min_value=0.0,
            max_value=DEFAULT_THRESHOLDS.medium,
            value=DEFAULT_THRESHOLDS.plagiarism,
            step=0.01,
            help=(
                "Controls which pairs are flagged. Severity remains Medium "
                f"at {DEFAULT_THRESHOLDS.medium:.0%} and High at "
                f"{DEFAULT_THRESHOLDS.high:.0%}."
            ),
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

        with st.expander("🔤 OCR Settings", expanded=False):
            ocr_language_labels = {
                display_name: code
                for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
            }
            language_names = list(ocr_language_labels)
            default_language_name = SUPPORTED_OCR_LANGUAGES[DEFAULT_OCR_LANGUAGE]
            selected_ocr_language_name = st.selectbox(
                "OCR Language",
                options=language_names,
                index=language_names.index(default_language_name),
                key="ocr_language_selector",
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                key="ocr_dpi_slider",
            )

        # ── Document Management & Bulk Clear ──
        st.markdown("---")
        st.markdown("### 📁 Document Management")
        existing_docs = get_all_documents()
        if existing_docs:
            st.write(f"**{len(existing_docs)}** documents in database")
            for doc in existing_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['filename']}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}"):
                        delete_document(doc["filename"])
                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        st.rerun()

        st.markdown('<div class="clear-all-container">', unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear All Documents",
            key="clear_all_documents_button",
            use_container_width=True,
        ):
            clear_all_dialog()
        st.markdown("</div>", unsafe_allow_html=True)

# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption("Paste a text snippet below to check its similarity against existing indexed assignments.")
    
    st.info("🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy.")
    
    query_text = st.text_area("Paste a text snippet to check against index:", height=150,
                              placeholder="Paste a paragraph here to check for plagiarism...")
    
    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        # Load existing index and registry from database
        from src.db.corpus_db import get_chunk_registry, get_all_embeddings
        from src.core.faiss_index import build_index_from_matrix
        
        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()
                
                if embeddings_matrix.shape[0] == 0:
                    from src.errors import UI_NO_DOCUMENTS_INDEXED
                    st.warning(UI_NO_DOCUMENTS_INDEXED)
                else:
                    # Build index from stored embeddings
                    faiss_index = build_index_from_matrix(embeddings_matrix, index_type="auto")
                    
                    # Embed the query
                    from src.core.embedding_model import embed_chunks
                    query_vec = embed_chunks([query_text.strip()])[0]
                    
                    # Search with threshold
                    faiss_threshold = threshold
                    results = search_similar_chunks(query_vec, faiss_index, registry,
                                                      top_k=faiss_top_k, threshold=faiss_threshold)
                    
                    if not results:
                        st.success("✅ No significant matches found in the assignment database.")
                    else:
                        st.success(f"Found **{len(results)}** potentially similar passages.")
                        
                        # Anonymize document names
                        doc_id_map = {}
                        anon_counter = 1
                        
                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = f"Document-{anon_counter:03d}"
                                anon_counter += 1
                        
                        # Display anonymized results
                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"
                            
                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index+1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1)
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(f"**Matching passage in {anon_doc_name}:**")
                                    st.warning(record.chunk_text)
                                
                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score*100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )
                        
                        st.caption("🔒 Document names are anonymized to protect student privacy.")
                        
            except Exception as e:
                from src.errors import UI_INDEX_LOAD_FAILED
                st.error(UI_INDEX_LOAD_FAILED.format(error=str(e)))
                st.info("Please ensure documents have been indexed by an administrator.")
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards
    
    # Load or initialize FAISS index
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        if faiss_index is not None and faiss_index.ntotal != len(registry):
            all_embs = get_all_embeddings()
            if len(all_embs) > 0 and len(all_embs) == len(registry):
                faiss_index = build_index_from_matrix(all_embs)
                save_index(faiss_index, _INDEX_PATH)
            elif len(all_embs) == 0:
                faiss_index = None
                registry = []
        if faiss_index is not None:
            st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        threshold = DEFAULT_THRESHOLDS.plagiarism
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

file_bytes_dict = (
    {f.name: f.getvalue() for f in uploaded_files} if uploaded_files else {}
)

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
                score = (
                    float(np.max(cosine_similarity(ea, eb)))
                    if ea.size and eb.size
                    else 0.0
                )
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

    # Filter out already-uploaded files and process only new ones
    new_files = {}
    skipped_files = []
    for f in uploaded_files:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        f.seek(0)  # Reset file pointer
        existing = get_document_by_hash(file_hash)
        if existing:
            skipped_files.append(f.name)
        else:
            new_files[f.name] = f.read()
            add_document(f.name, file_hash)
    
    if skipped_files:
        st.info(f"⏭️ Skipped {len(skipped_files)} already-uploaded files: {', '.join(skipped_files)}")
    
    if not new_files:
        from src.errors import UI_NO_NEW_FILES
        st.warning(UI_NO_NEW_FILES)
        if faiss_index is None:
            st.stop()
        # Continue with existing index for analysis
    else:
        st.info(f"📤 Processing {len(new_files)} new files...")
        
        # Process new files
        with st.spinner("🧠 Processing new PDFs, building embeddings and FAISS index…"):
            raw_texts_new, chunked_docs_new, embeddings_new, sim_df_new, chunk_sim_df_new, faiss_index_new, registry_new = \
                run_pipeline(new_files)
        
        # If we have an existing index, merge the new data
        if faiss_index is not None:
            # Add new embeddings to existing index
            all_vectors = []
            all_registry = registry.copy()
            
            for doc_name, emb in embeddings_new.items():
                chunks = chunked_docs_new.get(doc_name, [])
                if emb.ndim != 2 or emb.shape[0] == 0:
                    continue
                for i, (vec, text) in enumerate(zip(emb, chunks)):
                    all_vectors.append(vec.astype("float32"))
                    all_registry.append(ChunkRecord(doc_name, i, text))
            
            if all_vectors:
                matrix = np.vstack(all_vectors)
                faiss_index.add(matrix)  # type: ignore[arg-type]
                registry = all_registry
                st.success(f"✅ Added {len(all_vectors)} new vectors to existing index")
            
            raw_texts = raw_texts_new
            chunked_docs = chunked_docs_new
            embeddings = embeddings_new
            sim_df = sim_df_new
            chunk_sim_df = chunk_sim_df_new
        else:
            # No existing index, use the new one
            faiss_index = faiss_index_new
            registry = registry_new
            raw_texts = raw_texts_new
            chunked_docs = chunked_docs_new
            embeddings = embeddings_new
            sim_df = sim_df_new
            chunk_sim_df = chunk_sim_df_new
        
        # Save the updated FAISS index to disk
        save_index(faiss_index, _INDEX_PATH)
        
        # Store chunks in database for persistence
        for doc_name, emb in embeddings_new.items():
            chunks = chunked_docs_new.get(doc_name, [])
            if emb.ndim != 2 or emb.shape[0] == 0:
                continue
            # Get the starting vector_id for this document
            start_id = len(get_chunk_registry())
            chunks_to_add = []
            for i, (vec, text) in enumerate(zip(emb, chunks)):
                chunks_to_add.append((start_id + i, doc_name, i, text, vec))
            add_chunks(chunks_to_add)
    
    # If we have an existing index but no new files, load existing data from database
    if faiss_index is not None and not new_files:
        all_embs = get_all_embeddings()
        chunked_docs = {}
        embeddings = {}
        for i, rec in enumerate(registry):
            doc_name = rec.doc_name
            if doc_name not in chunked_docs:
                chunked_docs[doc_name] = []
                embeddings[doc_name] = []
            chunked_docs[doc_name].append(rec.chunk_text)
            if i < len(all_embs):
                embeddings[doc_name].append(all_embs[i])

        for d in embeddings:
            embeddings[d] = np.array(embeddings[d], dtype=np.float32)

        raw_texts = {d: "\n\n".join(chunks) for d, chunks in chunked_docs.items()}

        if len(embeddings) >= 2:
            sim_df = document_similarity_matrix(embeddings)
            chunk_sim_df = None
            active_sim_df = sim_df
            flags = flag_plagiarism(active_sim_df, threshold=threshold)
        else:
            sim_df = None
            chunk_sim_df = None
            active_sim_df = None
            flags = []
    else:
        # Check for empty PDFs (e.g. scanned images with no OCR)
        empty_docs = [name for name, text in raw_texts.items() if not text.strip()]
        if empty_docs:
            from src.errors import UI_COULD_NOT_EXTRACT_TEXT
            st.warning(UI_COULD_NOT_EXTRACT_TEXT.format(docs=', '.join(empty_docs)))

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
st.write(
    f"Processed **{len(raw_texts)}** documents with Chunk Size: `{chunk_size}` and Overlap: `{chunk_overlap}`."
)

with st.sidebar:
    selected_class = st.selectbox(
        "Select Class/Section",
        unique_classes,
        index=0,
        key="class_filter_selectbox",
    )

    st.markdown("---")
    st.markdown(
        """
**How it works**
1. Upload **PDF, DOCX, or TXT** assignment files or import from Google Drive
2. Text is extracted according to the file type
3. Text is split into **paragraph chunks**
4. Chunks are embedded with **all-MiniLM-L6-v2**
5. A **FAISS index** is built over all chunk vectors
6. Pairs above the threshold are flagged
"""
    )
    st.markdown("---")
    st.caption("Semantic Plagiarism Detector · FAISS edition")

    st.markdown("---")
    if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDF, DOCX, or TXT files. Detects **semantic similarity** "
    "(even paraphrased text) using transformer embeddings + **FAISS vector search**."
)
st.divider()


# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
if (
    Tour is not None
    and user_role == "admin"
    and not get_tour_completed(st.session_state.username)
):
    username = st.session_state.username

    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state.show_tour = True

    if st.session_state.get("show_tour", False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                "threshold_slider",
                title="⚙️ Plagiarism Threshold",
                desc=(
                    "Adjust the flagging threshold. Medium severity starts "
                    f"at {DEFAULT_THRESHOLDS.medium:.0%} and High at "
                    f"{DEFAULT_THRESHOLDS.high:.0%}."
                ),
                side="right",
            ),
            Tour.bind(
                "class_filter_selectbox",
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
            Tour.info(
                title="🎉 You're All Set!",
                desc="You can now start uploading assignments and detecting plagiarism.",
            ),
        ]

        tour = Tour(steps=tour_steps)
        tour.start()

        set_tour_completed(username, True)
        st.session_state.show_tour = False
        st.success("✅ Onboarding tour completed!")
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDF, DOCX, or TXT files. Detects **semantic similarity** "
    "(even paraphrased text) using transformer embeddings + **FAISS vector search**."
)
st.divider()

# ── MAIN APPLICATION SECTIONS ──────────────────────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads are restricted to Administrator access."
    )

    query_text = st.text_area(
        "Paste a text snippet to check against index:",
        height=150,
        placeholder="Paste a paragraph here to check for plagiarism...",
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        from src.core.faiss_index import build_index_from_matrix
        from src.db.corpus_db import get_all_embeddings, get_chunk_registry

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    st.warning("No documents are currently indexed.")
                else:
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]
                    faiss_threshold = threshold
                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=faiss_threshold,
                    )

                    if selected_class != "All Classes":
                        class_docs = get_documents_by_class(selected_class)
                        results = [
                            (record, score)
                            for record, score in results
                            if record.doc_name in class_docs
                        ]

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
else:
    # ADMIN FULL ACCESS VIEW
    faiss_index = None
    registry = []

    # Try to load FAISS index from Redis cache
    cached_index_data = get_faiss_index("corpus_index")
    if cached_index_data is not None:
        try:
            import faiss

            index_buffer = _io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(
                f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors"
            )
        except Exception as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")

    # If Redis loading failed, load from local disk
    if faiss_index is None:
        try:
            from src.core.faiss_index import load_or_rebuild_index

            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)

            if index_recovered:
                if faiss_index.ntotal:
                    st.warning(
                        f"FAISS index was missing, corrupted, or inconsistent and was "
                        f"automatically rebuilt from {faiss_index.ntotal} stored vectors."
                    )
                else:
                    st.info(
                        "No stored embeddings were found. An empty FAISS index was "
                        "initialized safely."
                    )
            else:
                st.info(
                    f"Loaded and validated the existing FAISS index with "
                    f"{faiss_index.ntotal} vectors."
                )
        except Exception:
            faiss_index = None
            registry = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        # Try to load from Redis cache

        cached_results = get_analysis_results(f"{SESSION_ID}:current")
        if cached_results is not None:
            st.session_state.analysis_results = cached_results

    # Initialize analysis_file_signature in session state
    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None

        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state.analysis_file_signature = cached_signature

    # 1. LOCAL FILE UPLOADER
    uploaded_files = st.file_uploader(
        "📂 Upload Assignments",
        type=["pdf", "docx", "txt", "zip"],
        accept_multiple_files=True,
        key="admin_file_uploader",
    )

    # 2. GOOGLE DRIVE IMPORT SECTION (#146)
    from src.utils.google_drive import bulk_download_drive_folder

    if "drive_files_dict" not in st.session_state:
        st.session_state.drive_files_dict = {}

    with st.expander("🌐 Import from Google Drive Folder", expanded=False):
        st.caption(
            "Paste a shared Google Drive folder link or ID to bulk-download assignments."
        )

        drive_folder_input = st.text_input(
            "Google Drive Folder Link / ID:",
            placeholder="https://drive.google.com/drive/folders/1A2B3C...",
            key="drive_folder_url_input",
        )

        drive_api_key = st.text_input(
            "API Key (Optional):",
            type="password",
            key="drive_api_key_input",
        )

        if st.button(
            "📥 Import Files from Drive", type="primary", use_container_width=True
        ):
            if not drive_folder_input.strip():
                st.error("Please enter a valid Google Drive folder link or ID.")
            else:
                with st.spinner(
                    "Connecting to Google Drive API & downloading files..."
                ):
                    try:
                        downloaded_dict, downloaded_names = bulk_download_drive_folder(
                            folder_url_or_id=drive_folder_input,
                            api_key=drive_api_key.strip() if drive_api_key else None,
                        )

                        if downloaded_dict:
                            st.session_state.drive_files_dict.update(downloaded_dict)
                            st.success(
                                f"✅ Imported {len(downloaded_names)} files: {', '.join(downloaded_names)}"
                            )
                            st.rerun()
                        else:
                            st.warning(
                                "No supported files (.pdf, .docx, .txt) found in this Drive folder."
                            )
                    except Exception as err:
                        st.error(f"Failed to import from Google Drive: {str(err)}")

    # 3. MERGE LOCAL AND DRIVE FILE BYTES
    file_bytes_dict = {}

    if uploaded_files:
        for f in uploaded_files:
            if f.name.lower().endswith(".zip"):
                try:
                    from src.utils.zip_processor import process_zip_file

                    zip_files = process_zip_file(f.read())
                    if not zip_files:
                        st.error(
                            f"⚠️ ZIP file '{f.name}' contains no supported documents (.pdf, .docx, .txt)."
                        )
                    else:
                        file_bytes_dict.update(zip_files)
                except ValueError as ve:
                    st.error(f"⚠️ Failed to process ZIP archive '{f.name}': {str(ve)}")
                except Exception:
                    st.error(
                        f"⚠️ Failed to process ZIP archive '{f.name}': Unknown error occurred."
                    )
            else:
                file_bytes_dict[f.name] = f.read()
            f.seek(0)

    if st.session_state.drive_files_dict:
        file_bytes_dict.update(st.session_state.drive_files_dict)

    # 4. PIPELINE STOP CHECK
    if len(file_bytes_dict) < 2:
        if st.session_state.analysis_results is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload or import from Drive at least 2 PDF, DOCX, or TXT assignments to begin.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
            st.stop()

    # ── Metadata Editor Section ──────────────────────────────────────────────────
    st.markdown("### 📝 Set Document Metadata")

    col1, col2 = st.columns(2)
    with col1:
        batch_class = st.text_input(
            "Default Class/Section",
            value="Class A",
            help="Default class section for all files in this batch.",
        )
    with col2:
        batch_assignment = st.text_input(
            "Default Assignment Title",
            value="Assignment 1",
            help="Default assignment title for all files in this batch.",
        )

    metadata_dict = {}
    for filename in file_bytes_dict.keys():
        base_name = os.path.splitext(filename)[0]
        guessed_name = base_name.replace("_", " ").replace("-", " ").title()

        with st.expander(f"📄 {filename}", expanded=False):
            student_name = st.text_input(
                f"Student Name for {filename}",
                value=guessed_name,
                key=f"student_{filename}",
            )
            class_section = st.text_input(
                f"Class/Section for {filename}",
                value=batch_class,
                key=f"class_{filename}",
            )
            assignment_title = st.text_input(
                f"Assignment Title for {filename}",
                value=batch_assignment,
                key=f"assignment_{filename}",
            )

            metadata_dict[filename] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip(),
            }

    # ── Pipeline Execution ────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(
        file_bytes_dict: dict[str, bytes],
        ocr_language: str,
        ocr_dpi: int,
    ):
        raw_texts = {}
        for name, data in file_bytes_dict.items():
            raw_texts[name] = extract_text(
                _io.BytesIO(data),
                name,
                ocr_language=ocr_language,
                ocr_dpi=ocr_dpi,
            )

        chunked_docs = chunk_documents(raw_texts)
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
                    score = (
                        float(np.max(cosine_similarity(ea, eb)))
                        if ea.size and eb.size
                        else 0.0
                    )
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
        analysis_results = run_pipeline(file_bytes_dict, ocr_language, ocr_dpi)

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

    # ── Summary Metrics ───────────────────────────────────────────────────────────

    if not uploaded_files or len(uploaded_files) < 2:
        st.markdown(
            empty_state_html(
                "Waiting for Files",
                "Please upload at least 2 PDF, DOCX, or TXT assignments to begin analysis.",
                "📂",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    # Process files pipeline
    raw_texts = {}
    for name, data in file_bytes_dict.items():
        raw_texts[name] = extract_text(
            _io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi
        )

    for flag in flags:
        try:
            send_plagiarism_alert(
                doc_a=flag["doc_a"],
                doc_b=flag["doc_b"],
                similarity=float(flag["similarity"]),
            )
        except Exception:
            pass

    # ── Summary Metrics ───────────────────────────────────────────────────────

    st.subheader("📊 Analysis Summary")
    doc_names = list(raw_texts.keys())
    n_docs = len(doc_names)
    total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
    n_flagged = len(flags)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📄 Documents", n_docs)
    col2.metric("🔗 Pairs", total_pairs)
    col3.metric("🚨 Flagged", n_flagged)
    col4.metric("🗂️ FAISS Vectors", faiss_index.ntotal if faiss_index is not None else 0)
    col5.metric("🎯 Threshold", f"{threshold:.0%}")
    st.divider()

    # ── Application Tabs ──────────────────────────────────────────────────────
    tab_warnings, tab_faiss, tab_matrix, tab_heatmap, tab_drill, tab_users = st.tabs(
        [
            "⚠️ Plagiarism Warnings",
            "⚡ FAISS Chunk Search",
            "📋 Similarity Matrix",
            "🗺️ Heatmap",
            "🔬 Pair Drill-Down",
            "👥 User Management",
        ]
    )

    # ══ TAB 1: WARNINGS ═══════════════════════════════════════════════════════

    with tab_warnings:
        st.subheader("⚠️ Plagiarism Warnings")
        render_warning_controls(
            flags, threshold=threshold, ai_probabilities=ai_probabilities
        )

        render_warning_controls(
            flags, threshold=threshold, ai_probabilities=ai_probabilities
        )

    # ══ TAB 2: FAISS ══════════════════════════════════════════════════════════
    with tab_faiss:
        st.subheader("⚡ FAISS Vector Search")
        st.info(f"Index total: {faiss_index.ntotal} vectors.")

        faiss_query = st.text_input(
            "Query FAISS Index:",
            placeholder="Type a text snippet to search vector index...",
            key="faiss_query_input",
        )
        if st.button("🔍 Run FAISS Search", key="run_faiss_search_btn"):
            if faiss_query.strip() and faiss_index is not None:
                from src.core.embedding_model import embed_chunks

                q_vec = embed_chunks([faiss_query.strip()])[0]
                q_results = search_similar_chunks(
                    q_vec,
                    faiss_index,
                    registry,
                    top_k=faiss_top_k,
                    threshold=threshold,
                )
                if q_results:
                    for rec, score in q_results:
                        st.markdown(
                            f"**{rec.doc_name}** (Chunk #{rec.chunk_index}) — Similarity: `{score:.1%}`"
                        )
                        st.caption(rec.chunk_text)
                else:
                    st.info("No matching vector chunks found above threshold.")

    # ══ TAB 3: MATRIX ═════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD
            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        else:

            def _highlight(val: Any) -> str:
                tier = severity_key(float(val))
                if tier == "high":
                    return (
                        "background-color:#ff4b4b;color:white;"
                        "font-weight:bold;"
                    )
                if tier == "medium":
                    return (
                        "background-color:#ffa500;color:white;"
                        "font-weight:bold;"
                    )
                return ""

            styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
            st.dataframe(styled_df, use_container_width=True)

            # Export options row
            col_csv, col_json, col_excel = st.columns(3)

            with col_csv:
                st.download_button(
                    "⬇️ Download CSV",
                    active_sim_df.to_csv().encode("utf-8"),
                    "similarity_matrix.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with col_json:
                json_data = export_similarity_matrix_to_json(active_sim_df).encode(
                    "utf-8"
                )
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    "similarity_matrix.json",
                    "application/json",
                    use_container_width=True,
                    key="json_export_button",
                )

            with col_excel:
                excel_data = export_similarity_matrix_to_excel(
                    active_sim_df, threshold=threshold
                )
                st.download_button(
                    "📊 Export as Styled Excel (.xlsx)",
                    excel_data,
                    "similarity_matrix_styled.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    # ══ TAB 4: HEATMAP ════════════════════════════════════════════════════════
    with tab_heatmap:
        st.subheader("🗺️ Similarity Heatmap")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD
            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        else:
            heatmap_fig = plot_similarity_heatmap(
                active_sim_df,
                title="Document Semantic Similarity",
                threshold=threshold,
                theme_colors=get_colors(),
            )

            st.pyplot(
                heatmap_fig,
                use_container_width=True,
            )

            buf = _io.BytesIO()
            heatmap_fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
            )

            buf.seek(0)

            st.download_button(
                "⬇️ Download Heatmap PNG",
                buf,
                "heatmap.png",
                "image/png",
            )

            st.divider()

            st.subheader("🕸️ Interactive Plagiarism Network")
            st.caption(
                "Documents are shown as nodes. Connections appear when "
                "their similarity is greater than or equal to the selected threshold."
            )

            network_fig = plot_similarity_network(
                similarity_df=active_sim_df,
                threshold=threshold,
                title="Interactive Document Plagiarism Network",
            )

            st.plotly_chart(
                network_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

    # ══ TAB 5: PAIR DRILL-DOWN ══════════════════════════════════════════════════
    with tab_drill:
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD
            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        elif len(active_sim_df) < 2:
            from src.errors import UI_NEED_MIN_DOCUMENTS
            st.warning(UI_NEED_MIN_DOCUMENTS)
        else:
            c1, c2 = st.columns(2)
            with c1:
                doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2:
                doc_b = st.selectbox(
                    "Document B", [d for d in doc_names if d != doc_a], index=0, key="db"
                )

        score = float(active_sim_df.loc[doc_a, doc_b])
        st.markdown(f"**Overall Similarity:** `{score:.1%}`")
        st.progress(float(score))
        st.divider()

        drill_tab_analysis, drill_tab_viewer = st.tabs(
            ["📊 Chunk Matches & Report", "📄 Document Viewer"]
        )

        chunks_a = chunked_docs.get(doc_a, [])
        chunks_b = chunked_docs.get(doc_b, [])

        with drill_tab_analysis:
            top_pairs = find_most_similar_chunks(
                chunks_a,
                chunks_b,
                embeddings[doc_a],
                embeddings[doc_b],
                top_k=5,
                threshold=threshold,
            )
            for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                with st.expander(f"#{rank} — Similarity: {sim:.1%}"):
                    st.write(f"**{doc_a}:** {ca}")
                    st.write(f"**{doc_b}:** {cb}")

        # --- In-App PDF Preview with Highlighted Matches (#145) ---
        with drill_tab_viewer:
            st.subheader("📄 In-App PDF Preview with Highlighted Matches")
            selected_view_doc = st.radio(
                "Select Document to Preview:",
                options=[doc_a, doc_b],
                horizontal=True,
                key="doc_viewer_select",
            )

            # Retrieve file bytes directly from uploaded files dict
            doc_source = file_bytes_dict.get(selected_view_doc)
            matching_chunks_to_highlight = (
                chunks_a if selected_view_doc == doc_a else chunks_b
            )

            if doc_source and str(selected_view_doc).lower().endswith(".pdf"):
                with st.spinner("Generating highlighted PDF preview..."):
                    try:
                        highlighted_pdf_bytes = highlight_pdf_matches(
                            pdf_source=doc_source,
                            matching_chunks=matching_chunks_to_highlight,
                        )

                        base64_pdf = base64.b64encode(highlighted_pdf_bytes).decode(
                            "utf-8"
                        )
                        pdf_display = f"""
                            <iframe
                                src="data:application/pdf;base64,{base64_pdf}"
                                width="100%"
                                height="850px"
                                type="application/pdf">
                            </iframe>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as err:
                        st.error(f"Unable to render PDF preview: {str(err)}")
            else:
                st.info("PDF Preview is only available for uploaded `.pdf` files.")

    # ══ TAB 6: USERS ══════════════════════════════════════════════════════════
    with tab_users:
        st.subheader("👥 User Management")
        users = get_all_users()
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True)

        st.write("---")
        st.subheader("🔐 Two-Factor Authentication (2FA)")

        current_user = st.session_state.get("username", "admin")
        enabled, otp_secret = get_2fa_status(current_user)

        if enabled:
            st.success(
                "✔️ Two-Factor Authentication is currently **enabled** for your account."
            )
            with st.expander("Deactivate Two-Factor Authentication", expanded=False):
                st.warning(
                    "⚠️ Disabling 2FA will lower your account security. Please confirm by entering your 6-digit verification code."
                )
                with st.form("disable_2fa_form"):
                    disable_code = st.text_input(
                        "Verification Code", max_chars=6, key="disable_2fa_code"
                    )
                    submit_disable = st.form_submit_button(
                        "Disable 2FA", use_container_width=True
                    )
                    if submit_disable:
                        import pyotp

                        totp = pyotp.TOTP(otp_secret)
                        if totp.verify(disable_code.strip()):
                            disable_2fa(current_user)
                            st.success("Two-factor authentication has been disabled.")
                            st.rerun()
                        else:
                            st.error("Invalid verification code. 2FA remains enabled.")
        else:
            st.info(
                "🔒 Two-Factor Authentication (2FA) is currently **disabled** for your account. We highly recommend enabling it."
            )

            if not st.session_state.get("show_2fa_setup", False):
                if st.button("Setup 2FA", use_container_width=True):
                    st.session_state.show_2fa_setup = True
                    import pyotp

                    # Generate a new random secret
                    st.session_state.temp_2fa_secret = pyotp.random_base32()
                    st.rerun()
            else:
                temp_secret = st.session_state.get("temp_2fa_secret")
                if temp_secret:
                    import pyotp

                    totp = pyotp.TOTP(temp_secret)
                    provisioning_uri = totp.provisioning_uri(
                        name=current_user, issuer_name="PlagiarismDetector"
                    )

                    st.markdown("### ⚙️ Step 1: Scan this QR Code")
                    st.write(
                        "Scan this QR code with your authenticator app (e.g., Google Authenticator, Authy)."
                    )

                    # Generate QR code offline using qrcode library
                    from io import BytesIO

                    import qrcode

                    qr = qrcode.QRCode(version=1, box_size=5, border=3)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    qr_bytes = buf.getvalue()

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(qr_bytes, width=250)
                    with col2:
                        st.markdown("**Manual Setup Details:**")
                        st.code(f"Account: {current_user}")
                        st.code(f"Secret Key: {temp_secret}")
                        st.caption(
                            "If your app does not support scanning QR codes, enter the secret key manually."
                        )

                    st.markdown("### ⚙️ Step 2: Verify and Enable 2FA")
                    st.write(
                        "Enter the 6-digit code shown in your authenticator app to complete setup."
                    )

                    with st.form("verify_2fa_setup_form"):
                        setup_code = st.text_input(
                            "6-digit Code", max_chars=6, key="setup_2fa_code"
                        )
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            submit_setup = st.form_submit_button(
                                "Verify and Enable", use_container_width=True
                            )
                        with col_btn2:
                            cancel_setup = st.form_submit_button(
                                "Cancel Setup", use_container_width=True
                            )

                        if submit_setup:
                            if totp.verify(setup_code.strip()):
                                enable_2fa(current_user, temp_secret)
                                # Clean up session state
                                st.session_state.show_2fa_setup = False
                                if "temp_2fa_secret" in st.session_state:
                                    del st.session_state.temp_2fa_secret
                                st.success(
                                    "🎉 Two-Factor Authentication has been successfully enabled!"
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "Invalid verification code. Please check the code in your app and try again."
                                )

                        if cancel_setup:
                            st.session_state.show_2fa_setup = False
                            if "temp_2fa_secret" in st.session_state:
                                del st.session_state.temp_2fa_secret
                            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Semantic Plagiarism Detection System · Streamlit")
