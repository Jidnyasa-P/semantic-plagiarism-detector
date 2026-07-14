import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io as _io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any

from src.core.document_parser import OCRDependencyError, extract_text
from src.core.text_chunking   import chunk_documents, chunk_document
from src.core.embedding_model import embed_documents, embed_chunks, _get_model_name
from src.core.similarity      import (
    document_similarity_matrix, flag_plagiarism,
    find_most_similar_chunks, PLAGIARISM_THRESHOLD,
    hybrid_similarity_matrix,
)
from src.core.lexical_similarity import lexical_similarity_matrix
from src.visualization.heatmap     import plot_similarity_heatmap, plot_similarity_heatmap_plotly, plot_chunk_similarity_comparison
from src.core.faiss_index import build_index, find_plagiarised_chunks, search_similar_chunks, build_index_from_matrix
from src.db.auth        import init_db, verify_user, get_user_role, get_all_users, add_user, delete_user, update_password
from src.db.corpus_db   import (
    init_corpus_db, add_document, get_document_by_hash, get_all_documents,
    add_chunks, get_chunk_registry, get_all_embeddings, delete_document, clear_all_data
)
from src.visualization.network_graph import plot_similarity_network
from src.core.translator import translate_text

from app import theme

@st.cache_resource
def _init_db_once():
    init_db()

_init_db_once()

st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
    
    /* File uploader custom premium styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #30363d !important;
        background-color: #0d1117 !important;
        border-radius: 15px !important;
        padding: 24px !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #388bfd !important;
        background-color: #161b22 !important;
        box-shadow: 0 8px 30px rgba(56, 139, 253, 0.15) !important;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Authentication ─────────────────────────────────────────────────────────────
def _login_page():
    st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC !important; min-height: 100vh; }
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        .stTextInput > label {
            font-size: 0.75rem !important; font-weight: 600 !important;
            color: #475569 !important; text-transform: uppercase !important; letter-spacing: 0.6px !important;
        }
        .stTextInput > div > div > input {
            background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important; color: #0F172A !important;
            font-size: 0.9rem !important; padding: 11px 14px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0D9488 !important;
            box-shadow: 0 0 0 3px rgba(13,148,136,0.15) !important;
        }
        .stFormSubmitButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
            color: #fff !important; border: none !important;
            border-radius: 8px !important; padding: 12px !important;
            font-size: 0.92rem !important; font-weight: 600 !important;
            margin-top: 8px !important;
            box-shadow: 0 4px 12px rgba(13,148,136,0.2) !important;
        }
        .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #0F766E 0%, #115E59 100%) !important;
        }
        div[data-testid="stForm"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important; padding: 36px 30px 32px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
            <div style='text-align:center; margin-bottom:24px; padding-top:15vh;'>
                <div style='display:inline-flex; align-items:center; gap:10px; justify-content:center;'>
                    <span style='font-size:2.2rem;'>🕵️‍♂️</span>
                    <span style='font-size:1.75rem; font-weight:700; color:#0F172A; font-family:"Newsreader", serif;'>Semantic Plagiarism Detector</span>
                </div>
                <div style='font-size:0.82rem; color:#64748B; margin-top:6px; font-weight: 500;'>AI-powered academic integrity platform</div>
            </div>
        """, unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("""
                <div style='font-size:1.4rem; font-weight:700; color:#0F172A; text-align:center;
                            font-family:"Newsreader", serif; letter-spacing:2px; margin-bottom:24px;'>LOGIN</div>
                <div style='height:1px; background:linear-gradient(90deg,transparent,#E2E8F0,transparent); margin-bottom:20px;'></div>
            """, unsafe_allow_html=True)
            username  = st.text_input("Username", placeholder="Enter your username")
            password  = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if verify_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"]      = username
                st.session_state["role"]          = get_user_role(username)
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state.get("authenticated"):
    _login_page()
    st.stop()

_is_admin = st.session_state.get("role") == "admin"

# Initialize persistent corpus DB
init_corpus_db()

# Load corpus into session state if not already done
if "faiss_index" not in st.session_state or "registry" not in st.session_state:
    embeddings_matrix = get_all_embeddings()
    st.session_state["faiss_index"] = build_index_from_matrix(embeddings_matrix)
    st.session_state["registry"] = get_chunk_registry()

def compute_matrices_from_db():
    import sqlite3
    from src.db.corpus_db import _DB_PATH
    
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute("SELECT filename, embedding FROM chunks").fetchall()
    conn.close()
    
    doc_embs = {}
    for fname, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(-1, 384)
        if fname not in doc_embs:
            doc_embs[fname] = []
        doc_embs[fname].append(emb)
        
    embeddings = {fname: np.vstack(embs) for fname, embs in doc_embs.items() if embs}
    
    if not embeddings:
        return pd.DataFrame(), pd.DataFrame(), {}
        
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
                score  = float(np.max(cosine_similarity(ea, eb))) if ea.size and eb.size else 0.0
                chunk_mat[i, j] = chunk_mat[j, i] = score
    chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)
    
    return sim_df, chunk_sim_df, embeddings

def get_all_raw_texts():
    import sqlite3
    from src.db.corpus_db import _DB_PATH
    
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute("SELECT filename, chunk_text FROM chunks ORDER BY filename, chunk_index").fetchall()
    conn.close()
    
    raw_texts = {}
    for fname, text in rows:
        if fname not in raw_texts:
            raw_texts[fname] = []
        raw_texts[fname].append(text)
        
    return {fname: "\n\n".join(texts) for fname, texts in raw_texts.items()}

def get_all_chunk_texts():
    import sqlite3
    from src.db.corpus_db import _DB_PATH
    
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute("SELECT filename, chunk_text FROM chunks ORDER BY filename, chunk_index").fetchall()
    conn.close()
    
    chunked_docs = {}
    for fname, text in rows:
        if fname not in chunked_docs:
            chunked_docs[fname] = []
        chunked_docs[fname].append(text)
    return chunked_docs

def process_new_files(uploaded_files):
    import hashlib
    import faiss
    
    new_files_processed = 0
    skipped_files = []
    empty_docs = []
    
    if not uploaded_files:
        return 0, [], []
        
    total_files = len(uploaded_files)
    progress_placeholder = st.empty()
    
    try:
        with progress_placeholder.container():
            with st.status("🧠 Processing documents...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                
                for idx, f in enumerate(uploaded_files):
                    file_progress_start = idx / total_files
                    file_progress_end = (idx + 1) / total_files
                    
                    # Stage 1: Reading uploaded documents
                    progress_bar.progress(file_progress_start + 0.1 / total_files)
                    status.write(f"📖 **Stage 1/5: Reading uploaded documents** - Extracting text from {f.name}...")
                    
                    file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    existing_name = get_document_by_hash(file_hash)
                    if existing_name:
                        skipped_files.append(f"{f.name} (duplicate of {existing_name})")
                        progress_bar.progress(file_progress_end)
                        continue
                        
                    try:
                        text = extract_text(_io.BytesIO(file_bytes), f.name)
                    except OCRDependencyError as exc:
                        st.error(f"Could not OCR **{f.name}**: {exc}")
                        empty_docs.append(f.name)
                        progress_bar.progress(file_progress_end)
                        continue

                    if not text.strip():
                        empty_docs.append(f.name)
                        progress_bar.progress(file_progress_end)
                        continue
                        
                    if not add_document(f.name, file_hash):
                        unique_name = f"{os.path.splitext(f.name)[0]}_{file_hash[:6]}{os.path.splitext(f.name)[1]}"
                        add_document(unique_name, file_hash)
                        doc_to_save = unique_name
                    else:
                        doc_to_save = f.name
                        
                    # Stage 2: Preparing document chunks
                    progress_bar.progress(file_progress_start + 0.2 / total_files)
                    status.write(f"✂️ **Stage 2/5: Preparing document chunks** for {doc_to_save}...")
                    chunks = chunk_document(text)
                    
                    # Stage 3: Generating embeddings
                    progress_bar.progress(file_progress_start + 0.7 / total_files)
                    status.write(f"🧬 **Stage 3/5: Generating embeddings** for {doc_to_save}...")
                    embs = embed_chunks(chunks)
                    
                    # Stage 4: Building the FAISS index
                    progress_bar.progress(file_progress_start + 0.95 / total_files)
                    status.write(f"⚡ **Stage 4/5: Building the FAISS index** for {doc_to_save}...")
                    
                    if embs.ndim == 2 and embs.shape[0] > 0:
                        start_vid = st.session_state["faiss_index"].ntotal
                        chunks_data = []
                        for i, (chk, emb) in enumerate(zip(chunks, embs)):
                            vid = start_vid + i
                            chunks_data.append((vid, doc_to_save, i, chk, emb))
                            
                        add_chunks(chunks_data)
                        st.session_state["faiss_index"].add(embs.astype("float32"))
                        new_files_processed += 1
                        
                    progress_bar.progress(file_progress_end)
                
                # Stage 5: Finalizing analysis
                progress_bar.progress(0.95)
                status.write("💾 **Stage 5/5: Finalizing analysis** - Saving index and updating registry...")
                
                faiss.write_index(st.session_state["faiss_index"], "corpus.index")
                st.session_state["registry"] = get_chunk_registry()
                
                progress_bar.progress(1.0)
                status.update(label="✅ Processing complete!", state="complete", expanded=False)
                
        # Clean up the progress status component entirely on success so completion messages render beautifully
        progress_placeholder.empty()
        
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ **Error during document processing:** {str(e)}")
        st.stop()
    
    return new_files_processed, skipped_files, empty_docs

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Compact brand header
    st.markdown("""
    <div style='text-align: center; padding-top: 10px; margin-bottom: 10px;'>
        <span style='font-size: 2.5rem;'>🕵️‍♂️</span>
        <div class="sidebar-brand-title">Antigravity Detector</div>
        <div class="sidebar-brand-kicker">Academic Integrity Portal</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section-label'>User Session</div>", unsafe_allow_html=True)
    st.markdown(f"👤 Logged in as: **{st.session_state.get('username', '')}**\n\n🔑 Role: `{st.session_state.get('role', '')}`")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        
    if _is_admin:
        if st.session_state.get("page") == "user_management":
            if st.button("◀ Back to Dashboard", use_container_width=True):
                st.session_state["page"] = "dashboard"
                st.rerun()
        else:
            if st.button("👥 User Management", use_container_width=True):
                st.session_state["page"] = "user_management"
                st.rerun()
                
    st.markdown("<div class='sidebar-section-label'>Configuration</div>", unsafe_allow_html=True)
    threshold = st.slider("Plagiarism Threshold", 0.50, 0.99,
                          value=PLAGIARISM_THRESHOLD, step=0.01,
                          help="Cosine similarity above which a pair is flagged.")
    hybrid_weight = st.slider("Semantic ↔ Lexical weight (w)", 0.0, 1.0, value=0.7, step=0.05,
                               help="Weight for semantic similarity (0.0 = pure lexical, 1.0 = pure semantic).")
    use_chunk_matrix = st.checkbox("Use chunk-level similarity matrix", value=False)
    
    if st.button("🗑️ Clear Entire Corpus", use_container_width=True, type="primary"):
        clear_all_data()
        st.session_state["faiss_index"] = build_index_from_matrix(np.empty((0, 384)))
        st.session_state["registry"] = []
        import faiss
        faiss.write_index(st.session_state["faiss_index"], "corpus.index")
        st.success("Corpus successfully cleared!")
        st.rerun()
    
    with st.expander("⚡ FAISS Options", expanded=False):
        faiss_top_k = st.slider("FAISS: matches per chunk", 1, 20, value=5)
        
    with st.expander("📚 How it Works", expanded=False):
        st.markdown("""
        1. PDFs parsed with **pypdf**
        2. Text split into **paragraph chunks**
        3. Chunks embedded with **paraphrase-multilingual-MiniLM-L12-v2**
        4. **FAISS index** built over all chunk vectors
        5. Pairwise **cosine similarity** computed
        6. Pairs above threshold flagged
        """)
        
    with st.expander("🌐 Supported Languages", expanded=False):
        st.markdown("""
        Uses a multilingual embedding model supporting 50+ languages, including:
        - English, Spanish, French, German, Chinese, Arabic, Japanese, and more.
        - Automatically handles cross-lingual similarity checking.
        """)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.caption("Semantic Plagiarism Detector · FAISS edition")
# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-kicker">Academic Integrity Portal</div>
""", unsafe_allow_html=True)
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student documents to detect **semantic similarity** and paraphrased content "
    "using advanced transformer embeddings and **FAISS vector search**."
)

model_name = _get_model_name()
st.markdown(f"""
<div style='display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.5rem; margin-bottom: 1.5rem;'>
    <div class="meta-chip">🤖 Model: <code>{model_name}</code></div>
    <div class="meta-chip">🎯 Threshold: <code>{threshold:.2f}</code></div>
    <div class="meta-chip">⚙️ Mode: <code>{"Chunk-level" if use_chunk_matrix else "Doc-level"}</code></div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── User Management page (admin only) ─────────────────────────────────────────
if _is_admin and st.session_state.get("page") == "user_management":
    st.title("👥 User Management")
    st.caption("Create, reset passwords, or delete users.")
    st.divider()

    with st.expander("➕ Create New User", expanded=True):
        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role     = st.selectbox("Role", ["teacher", "admin"])
            if st.form_submit_button("Create User", use_container_width=True):
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                else:
                    try:
                        add_user(new_username, new_password, new_role)
                        st.success(f"✅ User **{new_username}** created as `{new_role}`.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()
    st.subheader("📋 Existing Users")
    for user in get_all_users():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"**{user['username']}** &nbsp; `{user['role']}`",
                            unsafe_allow_html=True)
            with c2:
                with st.popover("🔑 Reset Password", use_container_width=True):
                    with st.form(f"reset_{user['id']}"):
                        np_ = st.text_input("New password", type="password",
                                            key=f"np_{user['id']}")
                        if st.form_submit_button("Update"):
                            if np_:
                                update_password(user["username"], np_)
                                st.success("Password updated.")
                            else:
                                st.error("Cannot be empty.")
            with c3:
                if user["username"] != "admin":
                    if st.button("🗑️ Delete", key=f"del_{user['id']}",
                                 use_container_width=True):
                        delete_user(user["username"])
                        st.rerun()
                else:
                    st.caption("(protected)")
    st.stop()

# Render queued indexing results from session state if any
if "indexing_results" in st.session_state:
    results = st.session_state.pop("indexing_results")
    n_added = results.get("n_added", 0)
    skipped = results.get("skipped", [])
    empty = results.get("empty", [])
    if n_added > 0:
        st.success(f"✅ Successfully indexed {n_added} new documents!")
    if skipped:
        st.warning("⚠️ Skipped duplicate files:\n" + "\n".join([f"- {s}" for s in skipped]))
    if empty:
        st.warning("⚠️ Skipped empty files:\n" + "\n".join([f"- {e}" for e in empty]))

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📂 Upload Assignment Documents to Corpus", type=["pdf", "docx", "txt"],
    accept_multiple_files=True, help="Upload files (PDF, DOCX, TXT) to index into the corpus database.",
)
if uploaded_files:
    c_btn, _ = st.columns([1, 3])
    with c_btn:
        if st.button("🚀 Process & Index Uploads", use_container_width=True):
            n_added, skipped, empty = process_new_files(uploaded_files)
            st.session_state["indexing_results"] = {
                "n_added": n_added,
                "skipped": skipped,
                "empty": empty
            }
            st.rerun()

active_docs = get_all_documents()
doc_names = [d["filename"] for d in active_docs]
n_docs = len(doc_names)

if n_docs == 0:
    st.info("👆 The corpus database is currently empty. Please upload and index at least 2 documents to start.")
    st.stop()

# Load states from database and session state
sim_df, chunk_sim_df, embeddings = compute_matrices_from_db()
raw_texts = get_all_raw_texts()
chunked_docs = get_all_chunk_texts()
faiss_index = st.session_state["faiss_index"]
registry = st.session_state["registry"]

# Compute lexical similarity matrix
lexical_df = lexical_similarity_matrix(raw_texts)

# Combine semantic and lexical using hybrid weight
hybrid_df = hybrid_similarity_matrix(sim_df, lexical_df, w=hybrid_weight)

active_sim_df = chunk_sim_df if use_chunk_matrix else hybrid_df
flags         = flag_plagiarism(active_sim_df, threshold=threshold)

# ── Summary metrics ────────────────────────────────────────────────────────────
st.subheader("📊 Analysis Summary")
col1, col2, col3, col4, col5 = st.columns(5)
doc_names    = list(raw_texts.keys())
n_docs       = len(doc_names)
total_pairs  = n_docs * (n_docs - 1) // 2
n_flagged    = len(flags)
n_high       = sum(1 for f in flags if "High" in f["severity"])
avg_sim      = active_sim_df.values[np.triu_indices(n_docs, k=1)].mean() if n_docs > 1 else 0.0
total_chunks = sum(len(v) for v in chunked_docs.values())

col1.metric("📄 Documents",      n_docs)
col2.metric("🔗 Pairs",          total_pairs)
col3.metric("🚨 Flagged",        n_flagged,
            delta=f"{n_high} High" if n_high else None, delta_color="inverse")
col4.metric("📈 Avg Similarity",  f"{avg_sim:.1%}")
col5.metric("🗂️ FAISS Vectors",  faiss_index.ntotal)
st.divider()

# ── Tabs (7 tabs) ──────────────────────────────────────────────────────────────
tab_warnings, tab_faiss, tab_matrix, tab_heatmap, tab_drill, tab_network, tab_corpus = st.tabs([
    "⚠️ Plagiarism Warnings",
    "⚡ FAISS Chunk Search",
    "📋 Similarity Matrix",
    "🗺️ Heatmap",
    "🔬 Pair Drill-Down",
    "🕸️ Plagiarism Network",
    "🗃️ Manage Corpus",
])

# ══ TAB 1 ═════════════════════════════════════════════════════════════════════
with tab_warnings:
    st.subheader("⚠️ Plagiarism Warnings")
    st.caption(f"Pairs with similarity ≥ **{threshold:.2f}**")
    if not flags:
        st.success("✅ No suspicious pairs found above the current threshold.")
    else:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["🔴 High", "🟡 Medium"],
            default=["🔴 High", "🟡 Medium"],
            help="Filter warnings displayed in the list below. This does not change the core detection logic or what gets logged/flagged."
        )
        
        filtered_flags = [f for f in flags if f["severity"] in severity_filter]
        
        if not filtered_flags:
            st.info("No warnings match the selected severity filter.")
        else:
            flags_df = pd.DataFrame(filtered_flags)
            st.download_button(
                "⬇️ Download Plagiarism Report (CSV)",
                flags_df.to_csv(index=False).encode("utf-8"),
                "plagiarism_warnings.csv", "text/csv", use_container_width=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            for flag in filtered_flags:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{flag['doc_a']}**  ↔  **{flag['doc_b']}**")
                        st.progress(float(flag["similarity"]),
                                    text=f"Similarity: {flag['similarity']*100:.1f}%")
                    with c2:
                        tier = theme.severity_tier(float(flag["similarity"]), threshold)
                        badge = theme.badge_html(tier, flag["severity"])
                        st.markdown(
                            f"<div style='text-align:center;padding-top:12px;'>{badge}</div>",
                            unsafe_allow_html=True,
                        )

# ══ TAB 2 ═════════════════════════════════════════════════════════════════════
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
                    
                    tier = theme.severity_tier(match["similarity"], faiss_threshold)
                    badge = theme.badge_html(tier, f"Similarity: {match['similarity']*100:.1f}%")
                    st.markdown(
                        f"<div style='text-align:right; margin-top:10px;'>{badge}</div>",
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
                    
                    tier = theme.severity_tier(score, faiss_threshold)
                    badge = theme.badge_html(tier, f"Similarity: {score:.1%}")
                    st.markdown(
                        f"<div style='text-align:right; margin-top: 10px;'>{badge}</div>",
                        unsafe_allow_html=True,
                    )

# ══ TAB 3 ═════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.subheader("📋 Similarity Matrix")
    st.markdown(f"""
    <div class="legend-container">
        <span style="font-weight: 700; color: #0F172A;">Legend:</span>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #ff4b4b;"></span>
            <span>High Plagiarism (≥ 90%)</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #ffa500;"></span>
            <span>Medium Plagiarism (≥ {threshold:.2f})</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #F1F5F9; border: 1px solid #E2E8F0;"></span>
            <span>Below Threshold</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    def _highlight(val: Any) -> str:
        v = float(val)
        if v >= 0.90:        return "background-color:#ff4b4b;color:white;font-weight:bold;"
        elif v >= threshold: return "background-color:#ffa500;color:white;font-weight:bold;"
        return ""
    st.dataframe(active_sim_df.style.format("{:.4f}").map(_highlight), use_container_width=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button("⬇️ Download CSV", active_sim_df.to_csv().encode("utf-8"),
                           "similarity_matrix.csv", "text/csv", use_container_width=True)
    with btn_col2:
        excel_buffer = _io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            active_sim_df.to_excel(writer, index=True, sheet_name="Similarity Matrix")
        st.download_button(
            "⬇️ Download Excel", excel_buffer.getvalue(),
            "similarity_matrix.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ══ TAB 4 ═════════════════════════════════════════════════════════════════════
with tab_heatmap:
    st.subheader("🗺️ Similarity Heatmap")
    hm_col1, hm_col2 = st.columns(2)
    with hm_col1:
        interactive = st.checkbox("Interactive mode (hover values)", value=True,
                                  help="Plotly chart with hover tooltips. Uncheck for static view.")

    with st.container(border=True):
        if interactive:
            plotly_fig = plot_similarity_heatmap_plotly(
                active_sim_df, title="Document Semantic Similarity", threshold=threshold
            )
            st.plotly_chart(plotly_fig, use_container_width=True)
            st.caption("💡 Hover over any cell to see exact similarity. Red borders = flagged pairs.")
        else:
            fig = plot_similarity_heatmap(
                active_sim_df, title="Document Semantic Similarity",
                threshold=threshold, annotate=True, dpi=200,
            )
            st.pyplot(fig, use_container_width=True)

    buf = _io.BytesIO()
    static_fig = plot_similarity_heatmap(
        active_sim_df, title="Document Semantic Similarity",
        threshold=threshold, annotate=True, dpi=200,
    )
    static_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button("⬇️ Download High-Res PNG", buf, "heatmap.png", "image/png")

# ══ TAB 5 ═════════════════════════════════════════════════════════════════════
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
        tier = theme.severity_tier(score, threshold)
        score_color = theme.tier_color(tier)
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
            max_d = 15
            st.pyplot(plot_chunk_similarity_comparison(
                doc_a, doc_b, chunks_a[:max_d], chunks_b[:max_d],
                cosine_similarity(emb_a, emb_b)[:max_d, :max_d],
            ), use_container_width=True)

            top_pairs = find_most_similar_chunks(
                chunks_a, chunks_b, emb_a, emb_b, top_k=5, threshold=threshold)
            if top_pairs:
                st.subheader("🔑 Top Suspicious Paragraph Pairs")
                for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                    with st.expander(f"#{rank} — Similarity: {sim:.1%}", expanded=(rank == 1)):
                        col1, col2 = st.columns(2)
                        with col1: st.markdown(f"**From {doc_a}**"); st.info(ca)
                        with col2: st.markdown(f"**From {doc_b}**"); st.warning(cb)
                        
                        tier = theme.severity_tier(sim, threshold)
                        badge = theme.badge_html(tier, f"Similarity: {sim:.1%}")
                        st.markdown(
                            f"<div style='text-align:right; margin-top: 10px;'>{badge}</div>",
                            unsafe_allow_html=True,
                        )
                        
                        translate_key = f"trans_{doc_a}_{doc_b}_{rank}"
                        if st.checkbox("🌐 Translate to English", key=translate_key):
                            col_t1, col_t2 = st.columns(2)
                            with col_t1:
                                trans_a = translate_text(ca, "en")
                                if trans_a.lower().strip() != ca.lower().strip():
                                    st.caption(f"**Translated:** {trans_a}")
                            with col_t2:
                                trans_b = translate_text(cb, "en")
                                if trans_b.lower().strip() != cb.lower().strip():
                                    st.caption(f"**Translated:** {trans_b}")
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

# ══ TAB 6 ═════════════════════════════════════════════════════════════════════
with tab_network:
    st.subheader("🕸️ Plagiarism Network Graph")
    st.markdown(
        "Visualize document relationships as a connection network. "
        "Each circle represents a document. Lines connect documents that share similarity "
        "above the connection threshold. Thicker lines indicate higher similarity."
    )
    
    net_col1, net_col2 = st.columns([3, 1])
    with net_col1:
        network_threshold = st.slider(
            "Network Connection Threshold", 0.50, 0.99,
            value=threshold, step=0.01, key="net_threshold_slider",
            help="Minimum similarity score required to connect two documents in the graph."
        )
    with net_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_isolated = st.checkbox("Show isolated documents", value=True,
                                    help="Uncheck to hide documents with no suspicious connections.")

    import networkx as nx
    G_stats = nx.Graph()
    for name in doc_names:
        G_stats.add_node(name)
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            score = float(active_sim_df.iloc[i, j])
            if score >= network_threshold:
                G_stats.add_edge(doc_names[i], doc_names[j])
                
    if not show_isolated:
        nodes_to_keep = [node for node, deg in G_stats.degree() if deg > 0]
        filtered_sim_df = active_sim_df.loc[nodes_to_keep, nodes_to_keep]
    else:
        filtered_sim_df = active_sim_df

    if len(filtered_sim_df) == 0:
        st.info("No connections found above the threshold. Try lowering the threshold or checking 'Show isolated documents'.")
    else:
        with st.spinner("Generating network graph layout..."):
            fig_net = plot_similarity_network(filtered_sim_df, threshold=network_threshold, title="")
            st.plotly_chart(fig_net, use_container_width=True)
            
        st.subheader("📊 Graph Insights")
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        
        components = list(nx.connected_components(G_stats))
        num_clusters = sum(1 for c in components if len(c) > 1)
        
        degrees = dict(G_stats.degree())
        max_deg = max(degrees.values()) if degrees else 0
        most_connected = [node for node, deg in degrees.items() if deg == max_deg and deg > 0]
        
        stat_c1.metric("🕸️ Active Connections", G_stats.number_of_edges())
        stat_c2.metric("🔗 Plagiarism Clusters", num_clusters)
        if most_connected:
            stat_c3.metric("🚨 Most Flagged Node", most_connected[0].split(".")[0], f"{max_deg} connections")
        else:
            stat_c3.metric("🚨 Most Flagged Node", "None", "0 connections")

# ══ TAB 7 ═════════════════════════════════════════════════════════════════════
with tab_corpus:
    st.subheader("🗃️ Manage Corpus Database")
    st.markdown("View, inspect, and delete documents stored in the database.")
    
    docs = get_all_documents()
    if not docs:
        st.info("No documents in the database.")
    else:
        # Create a table showing stats
        table_data = []
        from src.db.corpus_db import get_document_chunks_count
        for d in docs:
            n_chunks = get_document_chunks_count(d["filename"])
            table_data.append({
                "Filename": d["filename"],
                "Upload Date": d["upload_date"][:19].replace("T", " "),
                "Chunks": n_chunks
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Selectbox to delete a document
        st.subheader("🗑️ Delete Document")
        doc_to_delete = st.selectbox("Select document to delete from corpus", [""] + doc_names)
        if doc_to_delete:
            st.warning(f"Are you sure you want to delete **{doc_to_delete}**? This will remove its chunks and rebuild the FAISS index.")
            if st.button("Confirm Delete", type="primary"):
                with st.spinner("Deleting document and rebuilding index..."):
                    delete_document(doc_to_delete)
                    # Rebuild the FAISS index from remaining embeddings
                    embs = get_all_embeddings()
                    st.session_state["faiss_index"] = build_index_from_matrix(embs)
                    # Save index
                    import faiss
                    faiss.write_index(st.session_state["faiss_index"], "corpus.index")
                    st.session_state["registry"] = get_chunk_registry()
                st.success(f"Deleted {doc_to_delete} successfully!")
                st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Semantic Plagiarism Detection System · Sentence Transformers + FAISS · Streamlit")
