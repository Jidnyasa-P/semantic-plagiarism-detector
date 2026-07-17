import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io as _io
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from utils.warning_list import render_warning_controls
from src.core.text_chunking import chunk_documents
from src.core.embedding_model import embed_documents
from src.core.similarity import (
    document_similarity_matrix, flag_plagiarism,
    find_most_similar_chunks, PLAGIARISM_THRESHOLD,
)
from src.visualization.heatmap import plot_similarity_heatmap, plot_chunk_similarity_comparison
from src.core.faiss_index import build_index, find_plagiarised_chunks, search_similar_chunks, save_index, load_index, build_index_from_matrix
from src.visualization.network_graph import plot_similarity_network
from src.core.faiss_index import build_index, find_plagiarised_chunks, search_similar_chunks
from src.core.webhook import send_plagiarism_alert
from src.db import init_corpus_db, get_all_documents, delete_document, get_all_embeddings, get_chunk_registry, add_document, get_document_by_hash, add_chunks
from src.core.document_parser import (
    extract_text_from_pdf,
    prepare_text_for_embedding,
)
import hashlib

# Initialize corpus database
init_corpus_db()

# FAISS index file path
_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus.index"))
from src.utils.pdf_report import generate_plagiarism_report
from src.db.auth import init_db, verify_user, get_user_role

# Must be the first Streamlit command called
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)
init_db()
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SESSION TIMEOUT & ROUTE PROTECTION MIDDLEWARE ─────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

# 1. Handle Automatic Session Expiration (Inactivity Check)
if "last_interaction" in st.session_state and st.session_state.get("authenticated", False):
    elapsed_time = time.time() - st.session_state.last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        # Clear sensitive session variables on timeout
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        st.warning("⏱️ Your session has expired due to 15 minutes of inactivity. Please log in again.")
        st.stop()
    else:
        # Record a timestamp of the latest user interaction
        st.session_state.last_interaction = time.time()

# 2. Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    st.title("🔒 Plagiarism Detection Portal Login")
    st.markdown("Please log in with your credentials to access the system.")
    
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

                    st.success(f"Welcome back, {username}!")
                    st.rerun()

            else:
                st.error("Invalid username or password.")
    st.stop()

# Get secure role for this active interaction
user_role = st.session_state.get("role", "user")

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size: 72px; line-height: 1;'>🕵️‍♂️</div>", unsafe_allow_html=True)
    st.title("⚙️ Settings")
    st.write(f"Logged in as: **{user_role.upper()}**")

    # Only show administrative settings to ADMIN users
    if user_role == "admin":
        threshold = st.slider("Plagiarism Threshold", 0.50, 0.99,
                              value=PLAGIARISM_THRESHOLD, step=0.01,
                              help="Cosine similarity above which a pair is flagged. (Recommended: 0.59 based on benchmark evaluation)")
        use_chunk_matrix = st.checkbox("Use chunk-level similarity matrix", value=False,
                                       help="Use MAX chunk-pair similarity instead of mean doc vectors.")
        faiss_top_k = st.slider("FAISS: matches per chunk", 1, 20, value=5,
                                help="Nearest neighbours per chunk in FAISS search.")
    else:
        # Fallbacks for Standard Users (Cannot alter thresholds or configs)
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        st.info("ℹ️ Settings configuration is restricted to Administrators.")

    st.markdown("---")
    st.markdown("""
**How it works**
1. PDFs parsed with **PyPDF2**
2. Text split into **paragraph chunks**
3. Chunks embedded with **all-MiniLM-L6-v2**
4. **FAISS index** built over all chunk vectors
5. Pairwise **cosine similarity** computed
6. Pairs above threshold flagged
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
                        delete_document(doc['filename'])
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
            st.info("No documents in database")
        st.markdown("---")
    
    # Log out button
    if st.button("🚪 Log Out", use_container_width=True):
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDFs. Detects **semantic similarity** (even paraphrased text) "
    "using transformer embeddings + **FAISS vector search**."
)
st.divider()

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
                    st.warning("No documents are currently indexed. Please contact your administrator.")
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
                st.error(f"Error loading index: {str(e)}")
                st.info("Please ensure documents have been indexed by an administrator.")
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards
    
    # Load or initialize FAISS index
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        faiss_index = None
        registry = []
    
    uploaded_files = st.file_uploader(
        "📂 Upload Assignment PDFs", type=["pdf"],
        accept_multiple_files=True, help="Upload 2 or more PDF files.",
    )
    
    # Allow analysis with existing index even without new uploads
    if not uploaded_files:
        if faiss_index is None or faiss_index.ntotal == 0:
            st.info("👆 Please upload **at least 2** PDF assignment files to begin.")
            st.stop()
        else:
            st.success(f"📂 Using existing index with {faiss_index.ntotal} vectors from {len(get_all_documents())} documents")
            # Skip to analysis section with existing index
            file_bytes_dict = {}
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            # We'll need to handle this case differently for the analysis
            st.warning("⚠️ Full similarity matrix requires re-uploading files. FAISS search is available with existing index.")
            # For now, require uploads for full functionality
            st.stop()
    
    if len(uploaded_files) < 2:
        st.info("👆 Please upload **at least 2** PDF assignment files to begin.")
        st.stop()

    # ── Pipeline (cached) ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(file_bytes_dict: dict, existing_index=None, existing_registry=None):
      raw_texts = {
        name: extract_text_from_pdf(_io.BytesIO(data))
        for name, data in file_bytes_dict.items()
    }

    # Original chunks are preserved for UI display.
      chunked_docs = chunk_documents(raw_texts)

    # Translated English chunks are used only for embeddings.
      translated_chunked_docs = {}

      for doc_name, chunks in chunked_docs.items():
        translated_chunked_docs[doc_name] = []

        for chunk in chunks:
            prepared = prepare_text_for_embedding(chunk)

            translated_chunked_docs[doc_name].append(
                prepared["embedding_text"]
            )

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

    # Store original chunks in the FAISS registry for UI display,
    # while the vectors themselves come from translated English text.
      faiss_index, registry = build_index(
        embeddings,
        chunked_docs,
    )

      return (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
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
        st.warning("No new files to upload. All uploaded files are already in the database.")
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
            start_id = len([r for r in registry if r.doc_name != doc_name])
            chunks_to_add = []
            for i, (vec, text) in enumerate(zip(emb, chunks)):
                chunks_to_add.append((start_id + i, doc_name, i, text, vec))
            add_chunks(chunks_to_add)
    
    # If we have an existing index but no new files, load existing data
    if faiss_index is not None and not new_files:
        # For now, we need to rebuild the full pipeline for similarity matrix
        # This is a limitation - we'd need to store raw_texts in DB to avoid this
        st.warning("⚠️ Similarity matrix requires re-uploading files. FAISS search is available with existing index.")
        # For full functionality, require uploads
        if not new_files:
            st.info("Please upload files to generate similarity matrix. FAISS search is available below.")
            # We'll allow FAISS search but skip the matrix
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            active_sim_df = None
            flags = []
    else:
        # Check for empty PDFs (e.g. scanned images with no OCR)
        empty_docs = [name for name, text in raw_texts.items() if not text.strip()]
        if empty_docs:
            st.warning(f"⚠️ **Could not extract text from:** {', '.join(empty_docs)}. These might be scanned images or password-protected PDFs.")

        active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
        flags         = flag_plagiarism(active_sim_df, threshold=threshold)

    # ── Webhook notifications for high-similarity matches (>= 90%) ───────────────
    if "notified_pairs" not in st.session_state:
        st.session_state.notified_pairs = set()
    
    if active_sim_df is not None:
        current_files = sorted(list(file_bytes_dict.keys()))
        if "last_uploaded_files" not in st.session_state or st.session_state.last_uploaded_files != current_files:
            st.session_state.last_uploaded_files = current_files
            st.session_state.notified_pairs = set()

        for flag in flags:
            if flag["similarity"] >= 0.90:
                pair_key = tuple(sorted([flag["doc_a"], flag["doc_b"]]))
                if pair_key not in st.session_state.notified_pairs:
                    send_plagiarism_alert(flag["doc_a"], flag["doc_b"], flag["similarity"])
                    st.session_state.notified_pairs.add(pair_key)

    # ── Summary metrics ───────────────────────────────────────────────────────────
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    doc_names    = list(raw_texts.keys())
    n_docs       = len(doc_names)
    total_pairs  = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
    n_flagged    = len(flags)
    n_high       = sum(1 for f in flags if "High" in f["severity"])
    avg_sim      = active_sim_df.values[np.triu_indices(n_docs, k=1)].mean() if active_sim_df is not None and n_docs > 1 else 0.0
    total_chunks = sum(len(v) for v in chunked_docs.values())

    col1.metric("📄 Documents",   n_docs)
    col2.metric("🔗 Pairs",       total_pairs)
    col3.metric("🚨 Flagged",     n_flagged,
                delta=f"{n_high} High" if n_high else None, delta_color="inverse")
    col4.metric("📈 Avg Similarity", f"{avg_sim:.1%}")
    col5.metric("🗂️ FAISS Vectors", faiss_index.ntotal)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_warnings, tab_faiss, tab_matrix, tab_heatmap, tab_drill = st.tabs([
        "⚠️ Plagiarism Warnings",
        "⚡ FAISS Chunk Search",
        "📋 Similarity Matrix",
        "🗺️ Heatmap",
        "🔬 Pair Drill-Down",
    ])

    # ══ TAB 1 ════════════════════════════════════════════════════════════════════
    with tab_warnings:
        st.subheader("⚠️ Plagiarism Warnings")
        render_warning_controls(flags, threshold=threshold)

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
            faiss_threshold = st.slider("FAISS similarity threshold", 0.50, 0.99,
                                        value=threshold, step=0.01, key="faiss_thresh")
        with faiss_col2:
            run_faiss = st.button("🔍 Run FAISS Search", type="primary", use_container_width=True)

        st.info(f"📐 Index: **{faiss_index.ntotal} vectors** across **{n_docs} documents** "
                f"({total_chunks} chunks total).")

        if run_faiss:
            with st.spinner("Searching FAISS index across all chunks…"):
                faiss_matches = find_plagiarised_chunks(
                    embeddings, chunked_docs, faiss_index, registry,
                    threshold=faiss_threshold, top_k=faiss_top_k,
                )

            if not faiss_matches:
                st.success("✅ No chunk-level matches found above the threshold.")
            else:
                st.success(f"Found **{len(faiss_matches)} suspicious chunk pairs**.")
                summary_rows = [{
                    "Source Document": m["source_doc"],
                    "Matched Document": m["match_doc"],
                    "Similarity": f"{m['similarity']:.1%}",
                    "Severity": "🔴 High" if m["similarity"] >= 0.90 else "🟡 Medium",
                } for m in faiss_matches]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.subheader("🔑 Matching Paragraph Pairs")
                for i, match in enumerate(faiss_matches[:20]):
                    color = "#ff4b4b" if match["similarity"] >= 0.90 else "#ffa500"
                    with st.expander(
                        f"#{i+1} · {match['source_doc']}  ↔  {match['match_doc']} "
                        f"— {match['similarity']*100:.1f}%", expanded=(i == 0)
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
                            f"<span style='background:{color};color:white;padding:3px 12px;"
                            f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                            f"Similarity: {match['similarity']*100:.1f}%</span></div>",
                            unsafe_allow_html=True,
                        )
                if len(faiss_matches) > 20:
                    st.caption(f"Showing top 20 of {len(faiss_matches)} matches.")

        st.divider()
        st.subheader("🔎 Query: Search Custom Text Against All Assignments")
        st.caption("Paste any text snippet — FAISS finds the most similar paragraphs across all uploads.")

        query_text = st.text_area("Paste a text snippet:", height=120,
                                  placeholder="Paste a paragraph from a suspected plagiarised source…")
        if st.button("🔍 Search Assignments", key="custom_query") and query_text.strip():
            from src.core.embedding_model import embed_chunks
            with st.spinner("Embedding query and searching…"):
                query_vec = embed_chunks([query_text.strip()])[0]
                results   = search_similar_chunks(query_vec, faiss_index, registry,
                                                  top_k=faiss_top_k, threshold=faiss_threshold)
            if not results:
                st.info("No sufficiently similar chunks found.")
            else:
                st.success(f"Top {len(results)} matches:")
                for rank, (record, score) in enumerate(results, 1):
                    with st.expander(
                        f"#{rank} — {record.doc_name} (chunk #{record.chunk_index+1}) · {score:.1%}",
                        expanded=(rank == 1)
                    ):
                        cq, cm = st.columns(2)
                        with cq:
                            st.markdown("**Your query:**"); st.info(query_text.strip())
                        with cm:
                            st.markdown(f"**Match in {record.doc_name}:**"); st.warning(record.chunk_text)

    # ══ TAB 3 ════════════════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        def _highlight(val: Any) -> str:
            numeric_val = float(val)
            if numeric_val >= 0.90:         return "background-color:#ff4b4b;color:white;font-weight:bold;"
            elif numeric_val >= threshold:  return "background-color:#ffa500;color:white;font-weight:bold;"
            return ""
        styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
        st.dataframe(styled_df, use_container_width=True)
        st.download_button("⬇️ Download CSV", active_sim_df.to_csv().encode("utf-8"),
                           "similarity_matrix.csv", "text/csv")

    # ══ TAB 4 ════════════════════════════════════════════════════════════════════
    with tab_heatmap:
     st.subheader("🗺️ Similarity Heatmap")

     heatmap_fig = plot_similarity_heatmap(
        active_sim_df,
        title="Document Semantic Similarity",
        threshold=threshold,
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

    # ══ TAB 5 ════════════════════════════════════════════════════════════════════
    with tab_drill:
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")
        if n_docs < 2:
            st.warning("Need at least 2 documents.")
        else:
            c1, c2 = st.columns(2)
            with c1: doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2: doc_b = st.selectbox("Document B",
                                           [d for d in doc_names if d != doc_a], index=0, key="db")

            score = float(active_sim_df.loc[doc_a, doc_b])
            score_color = "#ff4b4b" if score >= 0.9 else ("#ffa500" if score >= threshold else "#21c55d")
            st.markdown(
                f"**Overall Similarity:** "
                f"<span style='color:{score_color};font-size:1.2rem;font-weight:700;'>"
                f"{score:.1%}</span>", unsafe_allow_html=True,
            )
            st.progress(float(score))
            st.divider()

            emb_a, emb_b     = embeddings.get(doc_a, np.array([])), embeddings.get(doc_b, np.array([]))
            chunks_a, chunks_b = chunked_docs.get(doc_a, []), chunked_docs.get(doc_b, [])

            if emb_a.size > 0 and emb_b.size > 0:
                max_d  = 15
                fig2   = plot_chunk_similarity_comparison(
                    doc_a, doc_b, chunks_a[:max_d], chunks_b[:max_d],
                    cosine_similarity(emb_a, emb_b)[:max_d, :max_d],
                )
                st.pyplot(fig2, use_container_width=True)

                top_pairs = find_most_similar_chunks(
                    chunks_a, chunks_b, emb_a, emb_b, top_k=5, threshold=threshold)
                if top_pairs:
                    st.subheader("🔑 Top Suspicious Paragraph Pairs")
                    for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                        with st.expander(f"#{rank} — Similarity: {sim:.1%}", expanded=(rank == 1)):
                            col1, col2 = st.columns(2)
                            with col1: st.markdown(f"**From {doc_a}**"); st.info(ca)
                            with col2: st.markdown(f"**From {doc_b}**"); st.warning(cb)
                    
                    st.divider()
                    st.subheader("📄 Generate PDF Report")
                    st.caption("Download a formal plagiarism report for this document pair.")
                    
                    if st.button("📥 Generate PDF Report", type="primary", use_container_width=True, key="pdf_report"):
                        with st.spinner("Generating PDF report..."):
                            try:
                                pdf_buffer = generate_plagiarism_report(
                                    doc_a=doc_a,
                                    doc_b=doc_b,
                                    overall_similarity=score,
                                    threshold=threshold,
                                    top_pairs=top_pairs,
                                    report_title="Plagiarism Detection Report"
                                )
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"plagiarism_report_{doc_a}_{doc_b}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                else:
                    st.success("No paragraph pairs above the threshold for this pair.")

            with st.expander("📄 View Raw Extracted Text"):
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**{doc_a}**")
                    st.text_area("", raw_texts.get(doc_a, "(empty)"), height=300, key="ta")
                with t2:
                    st.markdown(f"**{doc_b}**")
                    st.text_area("", raw_texts.get(doc_b, "(empty)"), height=300, key="tb")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Semantic Plagiarism Detection System · Sentence Transformers + FAISS · Streamlit")