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
    load_or_rebuild_index,
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
# Branding config persistence
_BRANDING_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "branding_config.json"))
_BRANDING_LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "branding_logo.png"))
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
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── SESSION TIMEOUT & ROUTE PROTECTION MIDDLEWARE ─────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

# 1. Handle Automatic Session Expiration (Inactivity Check)
# Try to get last_interaction from Redis cache first, fallback to session_state
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
        # Clear sensitive session variables on timeout
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        # Clear Redis session data
        clear_session(SESSION_ID)
        st.warning(
            "⏱️ Your session has expired due to 15 minutes of inactivity. Please log in again."
        )
        st.stop()
    else:
        # Record a timestamp of the latest user interaction
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# 2. Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    st.markdown(
        '<div class="login-container">'
        '<div class="login-header">'
        '<div class="login-icon">🔍</div>'
        '<div class="login-title">Plagiarism Detection Portal</div>'
        '<div class="login-subtitle">Sign in to access the system</div>'
        '</div>'
        '<div class="login-accent-bar"></div>',
        unsafe_allow_html=True,
    )

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

                if role is None:
                    st.error("Unable to determine the user role.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.last_interaction = time.time()
                    # Cache session state in Redis
                    cache_session_state(SESSION_ID, "authenticated", True)
                    cache_session_state(SESSION_ID, "username", username)
                    cache_session_state(SESSION_ID, "role", role)
                    cache_session_state(SESSION_ID, "last_interaction", time.time())

                    st.success(f"Welcome back, {username}!")
                    st.rerun()

            else:
                st.error("Invalid username or password.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Get secure role for this active interaction
user_role = st.session_state.get("role", "user")

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand-title">🔍 Plagiarism Detector</div>'
        '<div class="sidebar-brand-kicker">Semantic Analysis Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        sidebar_user_badge_html(
            st.session_state.get("username", "user"), user_role
        ),
        unsafe_allow_html=True,
    )

    current_theme = get_theme_name()

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

    # Only show administrative settings to ADMIN users
    if user_role == "admin":
        threshold = st.slider(
            "Plagiarism Threshold",
            0.50,
            0.99,
            value=PLAGIARISM_THRESHOLD,
            step=0.01,
            help="Cosine similarity above which a pair is flagged. (Recommended: 0.59 based on benchmark evaluation)",
            key="threshold_slider",
        )
        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            help="Use MAX chunk-pair similarity instead of mean doc vectors.",
            key="chunk_matrix_checkbox",
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            help="Nearest neighbours per chunk in FAISS search.",
            key="faiss_top_k_slider",
        )

        with st.expander("🔤 OCR Settings", expanded=False):
            st.caption(
                "Used only for scanned or image-only PDF pages. "
                "Text-based PDFs continue to use native extraction."
            )

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
                help=(
                    "The matching Tesseract language pack must be installed "
                    "on the operating system."
                ),
                key="ocr_language_selector",
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                help=(
                    "Higher DPI may improve recognition on small or blurred "
                    "text, but uses more memory and processing time."
                ),
                key="ocr_dpi_slider",
            )

            st.caption(f"Active OCR configuration: `{ocr_language}` at `{ocr_dpi} DPI`")
    else:
        # Fallbacks for Standard Users (Cannot alter thresholds or configs)
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI
        st.info("ℹ️ Settings configuration is restricted to Administrators.")

    st.markdown("---")
    st.markdown("### 🔍 Class Filter")
    unique_classes = ["All Classes"] + get_unique_class_sections()
    selected_class = st.selectbox(
        "Select Class/Section",
        unique_classes,
        index=0,
        help="Filter the analysis dashboard and similarity matrices by a specific class section.",
        key="class_filter_selectbox",
    )

    st.markdown("---")
    st.markdown("""
**How it works**
1. Upload **PDF, DOCX, or TXT** assignment files
2. Text is extracted according to the file type
3. Text is split into **paragraph chunks**
4. Chunks are embedded with **all-MiniLM-L6-v2**
5. A **FAISS index** is built over all chunk vectors
6. Pairs above the threshold are flagged
""")
    st.markdown("---")
    st.caption("Semantic Plagiarism Detector · FAISS edition")

    # Document management (admin only)
    if user_role == "admin":
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
                        # Rebuild FAISS index from remaining embeddings
                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            # No embeddings left, remove the index file
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        st.rerun()
        else:
            st.markdown(
                empty_state_html(
                    "No documents in database",
                    "Upload assignments to start detecting plagiarism.",
                    "📁",
                ),
                unsafe_allow_html=True,
            )
        st.divider()

    # Log out button
    if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        # Clear Redis session data
        clear_session(SESSION_ID)
        st.rerun()

# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
if Tour is not None and user_role == "admin" and not get_tour_completed(st.session_state.username):
    username = st.session_state.username
    
    # Add a button to start the tour
    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state.show_tour = True
    
    # Show tour when triggered
    if st.session_state.get("show_tour", False):
        # Create tour steps
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started with detecting plagiarism in student assignments."
            ),
            Tour.bind("threshold_slider", 
                      title="⚙️ Plagiarism Threshold",
                      desc="Adjust the similarity threshold for flagging potential plagiarism. Higher values = stricter detection. Recommended: 0.59",
                      side="right"),
            Tour.bind("class_filter_selectbox",
                      title="🔍 Class Filter",
                      desc="Filter analysis results by specific class sections to focus on particular groups of students.",
                      side="right"),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="After uploading files, the system will analyze them and display similarity metrics, flagged pairs, and detailed comparisons in the tabs below."
            ),
            Tour.info(
                title="🔬 Drill-Down Tab",
                desc="Use the Pair Drill-Down tab to examine specific document pairs in detail, view matching text chunks, and understand the similarity scores."
            ),
            Tour.info(
                title="🎉 You're All Set!",
                desc="You can now start uploading assignments and detecting plagiarism. This tour won't appear again."
            ),
        ]
        
        # Create and start the tour
        tour = Tour(steps=tour_steps)
        tour.start()
        
        # Mark tour as completed
        set_tour_completed(username, True)
        st.session_state.show_tour = False
        st.success("✅ Onboarding tour completed!")
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDF, DOCX, or TXT files. Detects **semantic similarity** "
    "(even paraphrased text) using transformer embeddings + "
    "**FAISS vector search**."
)
st.divider()

# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy."
    )

    query_text = st.text_area(
        "Paste a text snippet to check against index:",
        height=150,
        placeholder="Paste a paragraph here to check for plagiarism...",
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        # Load existing index and registry from database
        from src.db.corpus_db import get_chunk_registry, get_all_embeddings
        from src.core.faiss_index import build_index_from_matrix

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    st.warning(
                        "No documents are currently indexed. Please contact your administrator."
                    )
                else:
                    # Build index from stored embeddings
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    # Embed the query
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]

                    # Search with threshold
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
                        
                        # Display gamification badge for plagiarism-free result
                        st.markdown("---")
                        st.markdown("### 🎉 Congratulations! Your Work is Original")
                        st.info(
                            "Your text has been verified as plagiarism-free. "
                            "Download your 'Originality Verified' badge below!"
                        )
                        
                        # Badge preview and download options
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            # Generate badge preview
                            try:
                                badge_png = generate_badge_png(
                                    student_name=st.session_state.username,
                                    text_preview=query_text.strip()[:100]
                                )
                                st.image(badge_png, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Badge preview unavailable: {str(e)}")
                            
                            # Download buttons
                            dl_col1, dl_col2 = st.columns(2)
                            
                            with dl_col1:
                                if st.button("📥 Download PNG", key="download_png", use_container_width=True):
                                    try:
                                        badge_png = generate_badge_png(
                                            student_name=st.session_state.username,
                                            text_preview=query_text.strip()[:100]
                                        )
                                        st.download_button(
                                            label="⬇️ Click to Save PNG",
                                            data=badge_png.getvalue(),
                                            file_name=f"originality_badge_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating PNG badge: {str(e)}")
                            
                            with dl_col2:
                                if st.button("📄 Download PDF", key="download_pdf", use_container_width=True):
                                    try:
                                        badge_pdf = generate_badge_pdf(
                                            student_name=st.session_state.username,
                                            text_preview=query_text.strip()[:100]
                                        )
                                        st.download_button(
                                            label="⬇️ Click to Save PDF",
                                            data=badge_pdf.getvalue(),
                                            file_name=f"originality_certificate_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating PDF badge: {str(e)}")
                        
                        st.markdown("---")
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

                        # Anonymize document names
                        doc_id_map = {}
                        anon_counter = 1

                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = (
                                    f"Document-{anon_counter:03d}"
                                )
                                anon_counter += 1

                        # Display anonymized results
                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"

                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index + 1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1),
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(
                                        f"**Matching passage in {anon_doc_name}:**"
                                    )
                                    st.warning(record.chunk_text)

                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"{format_similarity_html(score, threshold)}</div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )

            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
                st.info(
                    "Please ensure documents have been indexed by an administrator."
                )
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards
    
    # Load or initialize FAISS index (try Redis cache first, fallback to disk)
    index_key = "corpus_index"
    cached_index_data = get_faiss_index(index_key)
    
    if cached_index_data is not None and os.path.exists(_INDEX_PATH):
        try:
            # Load from Redis cache
            import faiss
            import io
            index_buffer = io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")
            from src.core.faiss_index import load_or_rebuild_index

            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)

            if index_recovered:
                if faiss_index.ntotal:
                    st.warning("FAISS index was missing, corrupted, or inconsistent and was "f"automatically rebuilt from {faiss_index.ntotal} stored vectors.")
                else:
                    st.info(
                    "No stored embeddings were found. An empty FAISS index was "
                    "initialized safely.")
            else:
                st.info(f"Loaded and validated the existing FAISS index with "f"{faiss_index.ntotal} vectors.")

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        # Try to load from Redis cache
        cached_results = get_analysis_results(f"{SESSION_ID}:current")
        if cached_results is not None:
            st.session_state.analysis_results = cached_results

    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None
        # Try to load from Redis cache
        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state.analysis_file_signature = cached_signature
    

    uploaded_files = st.file_uploader(
        "📂 Upload Assignments",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload 2 or more PDF, DOCX, or TXT files.",
        key="file_uploader",
    )

    file_bytes_dict = {
        uploaded_file.name: uploaded_file.getvalue() for uploaded_file in uploaded_files
    }
    if uploaded_files:
        for f in uploaded_files:
            file_bytes_dict[f.name] = f.read()
            f.seek(0)

    # Allow analysis with existing index even without new uploads
    if not uploaded_files or len(uploaded_files) < 2:
        if st.session_state.analysis_results is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload at least 2 PDF, DOCX, or TXT assignments to begin the analysis.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
            st.stop()
        else:
            st.success(
                f"📂 Using existing index with {faiss_index.ntotal} vectors from {len(get_all_documents())} documents"
            )
            # Skip to analysis section with existing index
            file_bytes_dict = {}
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            # We'll need to handle this case differently for the analysis
            st.warning(
                "⚠️ Full similarity matrix requires re-uploading files. FAISS search is available with existing index."
            )
            # For now, require uploads for full functionality
            st.stop()

    if len(uploaded_files) < 2:
        st.info(
            "👆 Please upload **at least 2** PDF, DOCX, or TXT assignment files to begin."
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

    st.markdown("Customize metadata for individual files if needed:")
    metadata_dict = {}
    for f in uploaded_files:
        base_name = os.path.splitext(f.name)[0]
        guessed_name = base_name.replace("_", " ").replace("-", " ").title()

        with st.expander(f"📄 {f.name}", expanded=False):
            student_name = st.text_input(
                f"Student Name for {f.name}",
                value=guessed_name,
                key=f"student_{f.name}",
            )
            class_section = st.text_input(
                f"Class/Section for {f.name}", value=batch_class, key=f"class_{f.name}"
            )
            assignment_title = st.text_input(
                f"Assignment Title for {f.name}",
                value=batch_assignment,
                key=f"assignment_{f.name}",
            )

            metadata_dict[f.name] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip(),
            }

    # ── OCR failure UI ─────────────────────────────────────────────────────────────
    class OCRFileBatchError(OCRDependencyError):
        """OCR dependency failure containing the names of affected PDF files."""

        def __init__(
            self,
            failed_files: list[str],
            details: list[str],
        ) -> None:
            self.failed_files = failed_files
            self.details = details
            super().__init__("; ".join(details))

    def show_ocr_dependency_error(
        failed_files: list[str],
        error_message: str,
    ) -> None:
        """Render a user-friendly OCR dependency and installation guide."""
        st.error("⚠️ OCR could not process one or more scanned PDF files.")

        with st.expander("View OCR setup instructions", expanded=True):
            st.markdown(
                """
                The affected PDF files appear to contain scanned or image-based
                pages. These pages require **Tesseract OCR**, but Tesseract or
                one of the required Python OCR packages is unavailable.
                """
            )

            st.markdown("#### Files that could not be processed")
            for filename in failed_files:
                st.markdown(f"- `{filename}`")

            st.markdown("#### Install Tesseract OCR")
            st.markdown(
                """
                **Windows**

                1. Install Tesseract OCR.
                2. Add the Tesseract installation directory to your system
                   `PATH`.
                3. Alternatively, set `TESSERACT_CMD` to the full path of
                   `tesseract.exe`.

                **macOS**

                ```bash
                brew install tesseract
                ```

                **Ubuntu/Debian Linux**

                ```bash
                sudo apt update
                sudo apt install tesseract-ocr
                ```

                **Required Python packages**

                ```bash
                python -m pip install pytesseract pymupdf pillow
                ```

                Restart the Streamlit application after installation.
                """
            )

            st.markdown("### Technical details")
            st.code(error_message)

    # ── Pipeline (cached) ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(
        file_bytes_dict: dict[str, bytes],
        ocr_language: str,
        ocr_dpi: int,
        existing_index=None,
        existing_registry=None,
    ):
        """Extract, chunk, embed and index uploaded assignment files."""
        raw_texts = {}
        failed_files = []
        failure_details = []

        # Process every uploaded file and collect OCR failures from scanned PDFs.
        for name, data in file_bytes_dict.items():
            try:
                raw_texts[name] = extract_text(
                    _io.BytesIO(data),
                    name,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                )
            except OCRDependencyError as exc:
                failed_files.append(name)
                failure_details.append(f"{name}: {exc}")

        if failed_files:
            raise OCRFileBatchError(failed_files, failure_details)

        # Original chunks are preserved for UI display.
        chunked_docs = chunk_documents(raw_texts)

        # Translated English chunks are used only for embeddings.
        translated_chunked_docs = {}

        for doc_name, chunks in chunked_docs.items():
            translated_chunked_docs[doc_name] = []

            for chunk in chunks:
                prepared = prepare_text_for_embedding(chunk)
                translated_chunked_docs[doc_name].append(prepared["embedding_text"])

        # Generate embeddings from translated English text.
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
                    ea = embeddings[na]
                    eb = embeddings[nb]

                    score = (
                        float(np.max(cosine_similarity(ea, eb)))
                        if ea.size and eb.size
                        else 0.0
                    )

                    chunk_mat[i, j] = score
                    chunk_mat[j, i] = score

        chunk_sim_df = pd.DataFrame(
            chunk_mat,
            index=names,
            columns=names,
        )

        # Original chunks remain in the registry, while vectors are generated
        # from translated English text.
        faiss_index, registry = build_index(
            embeddings,
            chunked_docs,
        )

        # Detect AI-generated text probability for each document
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

    # Filter already-uploaded files, but do not save new files to the database
    # until PDF extraction and the complete pipeline succeed.
    new_files = {}
    skipped_files = []

    for uploaded_file in uploaded_files:
        file_data = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_data).hexdigest()
        existing = get_document_by_hash(file_hash)

        if existing:
            skipped_files.append(uploaded_file.name)
        else:
            new_files[uploaded_file.name] = {
                "data": file_data,
                "hash": file_hash,
            }
    if skipped_files:
        st.info(
            f"⏭️ Skipped {len(skipped_files)} already-uploaded files: "
            f"{', '.join(skipped_files)}"
        )

    if not new_files:
        st.warning(
            "No new files to upload. All uploaded files are already in the database."
        )
        if st.session_state.analysis_results is None:
            st.stop()
        (
            raw_texts,
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
        ) = st.session_state.analysis_results
    else:
        st.markdown(
            pipeline_progress_html(
                ["Extract Text", "Analyze Semantics", "Build Vector Index"],
                1,
            ),
            unsafe_allow_html=True,
        )

        pipeline_files = {
            name: file_info["data"] for name, file_info in new_files.items()
        }

        try:
            with st.spinner(
                "🧠 Processing new files, building embeddings and FAISS index…"
            ):
                analysis_results = run_pipeline(
                    pipeline_files,
                    ocr_language,
                    ocr_dpi,
                )
                file_signature = hashlib.sha256(
                    "".join(sorted(new_files.keys())).encode("utf-8")
                ).hexdigest()

        except OCRDependencyError as exc:
            failed_files = getattr(
                exc,
                "failed_files",
                list(pipeline_files.keys()),
            )

            show_ocr_dependency_error(
                failed_files=failed_files,
                error_message=str(exc),
            )
            st.stop()

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

        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_file_signature = file_signature
        # Cache analysis results in Redis
        cache_analysis_results(f"{SESSION_ID}:current", analysis_results)
        cache_session_state(SESSION_ID, "analysis_file_signature", file_signature)

        # Persist only documents that are not already stored. Database duplicate
        # detection must not decide whether dashboard data remains visible.
        saved_documents = 0
        skipped_documents = []

        for uploaded_file in uploaded_files:
            file_data = file_bytes_dict[uploaded_file.name]
            file_hash = hashlib.sha256(file_data).hexdigest()
            existing = get_document_by_hash(file_hash)

            if existing:
                skipped_documents.append(uploaded_file.name)
                continue

        # The complete pipeline succeeded, so the documents can now be stored.
        for doc_name, file_info in new_files.items():
            meta = metadata_dict.get(
                doc_name,
                {"student_name": "", "class_section": "", "assignment_title": ""},
            )
            add_document(
                doc_name,
                file_info["hash"],
                class_section=meta["class_section"],
                student_name=meta["student_name"],
                assignment_title=meta["assignment_title"],
            )

        # If an index already exists, save and cache it.
        save_index(faiss_index, _INDEX_PATH)
        try:
            import io
            import faiss
            index_buffer = io.BytesIO()
            faiss.write_index(faiss_index, index_buffer)
            index_buffer.seek(0)
            cache_faiss_index(SESSION_ID, index_buffer.getvalue())
        except Exception as e:
            print(f"[Redis] Error caching FAISS index: {e}")

        # Store chunks in the database for persistence.
        for doc_name, emb in embeddings.items():
            chunks = chunked_docs.get(doc_name, [])
            if emb.ndim != 2 or emb.shape[0] == 0:
                continue
            start_id = len(
                [record for record in registry if record.doc_name != doc_name]
            )
            chunks_to_add = []
            for i, (vec, chunk_text) in enumerate(zip(emb, chunks)):
                chunks_to_add.append((start_id + i, doc_name, i, chunk_text, vec))
            add_chunks(chunks_to_add)
    # If we have an existing index but no new files, load existing data
    if faiss_index is not None and not new_files:
        # For now, we need to rebuild the full pipeline for similarity matrix
        # This is a limitation - we'd need to store raw_texts in DB to avoid this
        st.warning(
            "⚠️ Similarity matrix requires re-uploading files. FAISS search is available with existing index."
        )
        # For full functionality, require uploads
        if not new_files:
            st.info(
                "Please upload files to generate similarity matrix. FAISS search is available below."
            )
            # We'll allow FAISS search but skip the matrix
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            active_sim_df = None
            flags = []
    else:
        # Check for files from which no usable text could be extracted
        empty_docs = [name for name, text in raw_texts.items() if not text.strip()]
        if empty_docs:
            st.warning(
                f"⚠️ **Could not extract text from:** {', '.join(empty_docs)}. "
                "The files may be empty, unsupported, scanned, corrupted, "
                "or password-protected."
            )

        active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
        flags = flag_plagiarism(active_sim_df, threshold=threshold)

    # Apply Class/Section filter if selected
    if selected_class != "All Classes":
        class_docs = get_documents_by_class(selected_class)
        # Filter raw_texts, chunked_docs, embeddings
        raw_texts = {k: v for k, v in raw_texts.items() if k in class_docs}
        chunked_docs = {k: v for k, v in chunked_docs.items() if k in class_docs}
        embeddings = {k: v for k, v in embeddings.items() if k in class_docs}

        # Slice active_sim_df if available
        if active_sim_df is not None:
            filtered_names = [
                name for name in active_sim_df.index if name in class_docs
            ]
            if len(filtered_names) >= 2:
                active_sim_df = active_sim_df.loc[filtered_names, filtered_names]
                # Re-calculate plagiarism flags for the filtered subset
                flags = flag_plagiarism(active_sim_df, threshold=threshold)
            else:
                active_sim_df = None
                flags = []
                st.warning(
                    f"⚠️ Need at least 2 documents in '{selected_class}' to display the similarity matrix and warnings."
                )

    # ── Webhook notifications for high-similarity matches (>= 90%) ───────────────
    if "notified_pairs" not in st.session_state:
        st.session_state.notified_pairs = set()

    if active_sim_df is not None:
        current_files = sorted(list(new_files.keys()))
        if (
            "last_uploaded_files" not in st.session_state
            or st.session_state.last_uploaded_files != current_files
        ):
            st.session_state.last_uploaded_files = current_files
            st.session_state.notified_pairs = set()

        for flag in flags:
            if flag["similarity"] >= 0.90:
                pair_key = tuple(sorted([flag["doc_a"], flag["doc_b"]]))
                if pair_key not in st.session_state.notified_pairs:
                    send_plagiarism_alert(
                        flag["doc_a"], flag["doc_b"], flag["similarity"]
                    )
                    st.session_state.notified_pairs.add(pair_key)

    # ── Summary metrics ───────────────────────────────────────────────────────────
    st.subheader("📊 Analysis Summary")
    doc_names = list(raw_texts.keys())
    n_docs = len(doc_names)
    total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
    n_flagged = len(flags)
    n_high = sum(1 for f in flags if "High" in f["severity"])
    avg_sim = (
        active_sim_df.values[np.triu_indices(n_docs, k=1)].mean()
        if active_sim_df is not None and n_docs > 1
        else 0.0
    )
    total_chunks = sum(len(v) for v in chunked_docs.values())

    # Calculate average AI probability across all documents
    avg_ai_prob = 0.0
    if ai_probabilities:
        ai_scores = [ai_probabilities.get(doc, {}).get('overall', 0.0) for doc in doc_names]
        avg_ai_prob = np.mean(ai_scores) if ai_scores else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📄 Documents", n_docs)
    col2.metric("🔗 Pairs", total_pairs)
    col3.metric(
        "🚨 Flagged",
        n_flagged,
        delta=f"{n_high} High" if n_high else None,
        delta_color="inverse",
    )
    col4.metric("📈 Avg Similarity", f"{avg_sim:.1%}")
    col5.metric("🤖 Avg AI Prob", f"{avg_ai_prob:.1%}")
    col6.metric(
        "🗂️ FAISS Vectors",
        faiss_index.ntotal if faiss_index is not None else 0,
    )
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────────
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

    # ══ TAB 1 ════════════════════════════════════════════════════════════════════
    with tab_warnings:
        st.subheader("⚠️ Plagiarism Warnings")
        render_warning_controls(flags, threshold=threshold, ai_probabilities=ai_probabilities)

    # ══ TAB 2: FAISS ═════════════════════════════════════════════════════════════
    with tab_faiss:
        st.subheader("⚡ FAISS Vector Search — Chunk-Level Plagiarism")
        st.markdown(
            "FAISS searches **every chunk** against all other documents' chunks. "
            "Uses exact search for small collections and **IVF approximate search** "
            "for large ones — scaling to thousands of assignments."
        )

        faiss_col1, faiss_col2 = st.columns([2, 1])
        with faiss_col1:
            faiss_threshold = st.slider(
                "FAISS similarity threshold",
                0.50,
                0.99,
                value=threshold,
                step=0.01,
                key="faiss_thresh",
            )
        with faiss_col2:
            run_faiss = st.button(
                "🔍 Run FAISS Search", type="primary", use_container_width=True
            )

        st.info(
            f"📐 Index: **{faiss_index.ntotal} vectors** across **{n_docs} documents** "
            f"({total_chunks} chunks total)."
        )

        if run_faiss:
            with st.spinner("Searching FAISS index across all chunks…"):
                faiss_matches = find_plagiarised_chunks(
                    embeddings,
                    chunked_docs,
                    faiss_index,
                    registry,
                    threshold=faiss_threshold,
                    top_k=faiss_top_k,
                )

            if selected_class != "All Classes":
                faiss_matches = [
                    m for m in faiss_matches if m["match_doc"] in class_docs
                ]

            if not faiss_matches:
                st.success("✅ No chunk-level matches found above the threshold.")
            else:
                st.success(f"Found **{len(faiss_matches)} suspicious chunk pairs**.")
                summary_rows = [
                    {
                        "Source Document": m["source_doc"],
                        "Matched Document": m["match_doc"],
                        "Similarity": f"{m['similarity']:.1%}",
                        "Severity": "🔴 High"
                        if m["similarity"] >= 0.90
                        else "🟡 Medium",
                    }
                    for m in faiss_matches
                ]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.subheader("🔑 Matching Paragraph Pairs")
                for i, match in enumerate(faiss_matches[:20]):
                    with st.expander(
                        f"#{i + 1} · {match['source_doc']}  ↔  {match['match_doc']} "
                        f"— {match['similarity'] * 100:.1f}%",
                        expanded=(i == 0),
                    ):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**📄 {match['source_doc']}**")
                            st.info(match["source_chunk_text"])
                        with cb:
                            st.markdown(f"**📄 {match['match_doc']}**")
                            st.warning(match["match_chunk_text"])
                        st.markdown(
                            f"<div style='text-align:right;'>"
                            f"{format_similarity_html(match['similarity'], threshold)}</div>",
                            unsafe_allow_html=True,
                        )
                if len(faiss_matches) > 20:
                    st.caption(f"Showing top 20 of {len(faiss_matches)} matches.")

        st.divider()
        st.subheader("🔎 Query: Search Custom Text Against All Assignments")
        st.caption(
            "Paste any text snippet — FAISS finds the most similar paragraphs across all uploads."
        )

        query_text = st.text_area(
            "Paste a text snippet:",
            height=120,
            placeholder="Paste a paragraph from a suspected plagiarised source…",
        )
        if (
            st.button("🔍 Search Assignments", key="custom_query")
            and query_text.strip()
        ):
            from src.core.embedding_model import embed_chunks

            with st.spinner("Embedding query and searching…"):
                query_vec = embed_chunks([query_text.strip()])[0]
                results = search_similar_chunks(
                    query_vec,
                    faiss_index,
                    registry,
                    top_k=faiss_top_k,
                    threshold=faiss_threshold,
                )

            if selected_class != "All Classes":
                results = [
                    (record, score)
                    for record, score in results
                    if record.doc_name in class_docs
                ]

            if not results:
                st.info("No sufficiently similar chunks found.")
            else:
                st.success(f"Top {len(results)} matches:")
                for rank, (record, score) in enumerate(results, 1):
                    with st.expander(
                        f"#{rank} — {record.doc_name} (chunk #{record.chunk_index + 1}) · {score:.1%}",
                        expanded=(rank == 1),
                    ):
                        cq, cm = st.columns(2)
                        with cq:
                            st.markdown("**Your query:**")
                            st.info(query_text.strip())
                        with cm:
                            st.markdown(f"**Match in {record.doc_name}:**")
                            st.warning(record.chunk_text)

    # ══ TAB 3 ════════════════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        if active_sim_df is None:
            st.markdown(
                empty_state_html(
                    "No Similarity Matrix",
                    "Ensure at least 2 documents are uploaded to compute similarities.",
                    "📋",
                ),
                unsafe_allow_html=True,
            )
        else:

            def _highlight(val: Any) -> str:
                numeric_val = float(val)
                if numeric_val >= 0.90:
                    return "background-color:#ff4b4b;color:white;font-weight:bold;"
                elif numeric_val >= threshold:
                    return "background-color:#ffa500;color:white;font-weight:bold;"
                return ""

            styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
            st.dataframe(styled_df, use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                active_sim_df.to_csv().encode("utf-8"),
                "similarity_matrix.csv",
                "text/csv",
            )

    # ══ TAB 4 ════════════════════════════════════════════════════════════════════
    with tab_heatmap:
        st.subheader("🗺️ Similarity Heatmap")
        if active_sim_df is None:
            st.markdown(
                empty_state_html(
                    "No Visualizations Available",
                    "Ensure at least 2 documents are uploaded to generate heatmaps and networks.",
                    "🗺️",
                ),
                unsafe_allow_html=True,
            )
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
                theme_colors=get_colors(),
            )

            st.plotly_chart(
                network_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

    # ══ TAB 5 ════════════════════════════════════════════════════════════════════
    with tab_drill:
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")
        if n_docs < 2:
            st.warning("Need at least 2 documents.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2:
                doc_b = st.selectbox(
                    "Document B",
                    [d for d in doc_names if d != doc_a],
                    index=0,
                    key="db",
                )

            score = float(active_sim_df.loc[doc_a, doc_b])
            _colors = get_colors()
            score_color = (
                _colors["danger"]
                if score >= 0.9
                else (_colors["warning"] if score >= threshold else _colors["success"])
            )
            st.markdown(
                f"**Overall Similarity:** "
                f"<span style='color:{score_color};font-size:1.2rem;font-weight:700;'>"
                f"{score:.1%}</span>",
                unsafe_allow_html=True,
            )
            st.progress(float(score))
            st.divider()

            emb_a, emb_b = (
                embeddings.get(doc_a, np.array([])),
                embeddings.get(doc_b, np.array([])),
            )
            chunks_a, chunks_b = (
                chunked_docs.get(doc_a, []),
                chunked_docs.get(doc_b, []),
            )

            if emb_a.size > 0 and emb_b.size > 0:
                max_d = 15
                fig2 = plot_chunk_similarity_comparison(
                    doc_a,
                    doc_b,
                    chunks_a[:max_d],
                    chunks_b[:max_d],
                    cosine_similarity(emb_a, emb_b)[:max_d, :max_d],
                    theme_colors=get_colors(),
                )
                st.pyplot(fig2, use_container_width=True)

                top_pairs = find_most_similar_chunks(
                    chunks_a, chunks_b, emb_a, emb_b, top_k=5, threshold=threshold
                )
                if top_pairs:
                    st.subheader("🔑 Top Suspicious Paragraph Pairs")
                    for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                        with st.expander(
                            f"#{rank} — Similarity: {sim:.1%}", expanded=(rank == 1)
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**From {doc_a}**")
                                st.info(ca)
                            with col2:
                                st.markdown(f"**From {doc_b}**")
                                st.warning(cb)

                    st.divider()
                    st.subheader("📄 Generate PDF Report")
                    st.caption(
                        "Download a formal plagiarism report for this document pair."
                    )

                    if st.button(
                        "📥 Generate PDF Report",
                        type="primary",
                        use_container_width=True,
                        key="pdf_report",
                    ):
                        with st.spinner("Generating PDF report..."):
                            try:
                                pdf_buffer = generate_plagiarism_report(
                                    doc_a=doc_a,
                                    doc_b=doc_b,
                                    overall_similarity=score,
                                    threshold=threshold,
                                    top_pairs=top_pairs,
                                    report_title="Plagiarism Detection Report",
                                )
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"plagiarism_report_{doc_a}_{doc_b}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                else:
                    st.success("No paragraph pairs above the threshold for this pair.")

            with st.expander("📄 View Raw Extracted Text"):
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**{doc_a}**")
                    st.text_area(
                        "", raw_texts.get(doc_a, "(empty)"), height=300, key="ta"
                    )
                with t2:
                    st.markdown(f"**{doc_b}**")
                    st.text_area(
                        "", raw_texts.get(doc_b, "(empty)"), height=300, key="tb"
                    )

    # ══ TAB 6: User Management ═══════════════════════════════════════════════════
    with tab_users:
        st.subheader("👥 User Management")
        st.caption(
            "Manage user accounts and roles. Only administrators can access this panel."
        )

        current_username = st.session_state.get("username", "")

        # Display existing users
        users = get_all_users()
        if users:
            users_df = pd.DataFrame(users)
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users found in database.")

        st.divider()

        # Add new user form
        st.subheader("Add New User")
        with st.form("add_user_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_username = st.text_input("Username")
            with col2:
                new_password = st.text_input("Password", type="password")
            with col3:
                new_role = st.selectbox("Role", ["teacher", "student", "admin"])

            add_user_submitted = st.form_submit_button("Add User", type="primary")

            if add_user_submitted:
                if new_username and new_password:
                    try:
                        add_user(new_username, new_password, new_role)
                        st.success(
                            f"User '{new_username}' added successfully with role '{new_role}'."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding user: {str(e)}")
                else:
                    st.error("Username and password are required.")

        st.divider()

        # Delete user form
        st.subheader("Delete User")
        with st.form("delete_user_form"):
            if users:
                user_to_delete = st.selectbox(
                    "Select user to delete",
                    [u["username"] for u in users if u["username"] != current_username],
                )
                delete_user_submitted = st.form_submit_button(
                    "Delete User", type="secondary"
                )

                if delete_user_submitted:
                    if user_to_delete == current_username:
                        st.error("You cannot delete your own account.")
                    else:
                        try:
                            delete_user(user_to_delete)
                            st.success(f"User '{user_to_delete}' deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting user: {str(e)}")
            else:
                st.info("No users available to delete.")

        st.divider()

        # Reset password form
        st.subheader("Reset Password")
        with st.form("reset_password_form"):
            if users:
                user_to_reset = st.selectbox(
                    "Select user", [u["username"] for u in users]
                )
                new_password_reset = st.text_input("New Password", type="password")
                reset_password_submitted = st.form_submit_button(
                    "Reset Password", type="secondary"
                )

                if reset_password_submitted:
                    if new_password_reset:
                        try:
                            update_password(user_to_reset, new_password_reset)
                            st.success(
                                f"Password for '{user_to_reset}' reset successfully."
                            )
                        except Exception as e:
                            st.error(f"Error resetting password: {str(e)}")
                    else:
                        st.error("New password is required.")
            else:
                st.info("No users available.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎓 Semantic Plagiarism Detection System · Sentence Transformers + FAISS · Streamlit"
)
