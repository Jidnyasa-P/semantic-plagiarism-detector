# 🔍 Semantic Plagiarism Detection System

> **[▶ Live Demo](https://semantic-plagiarism-detector.streamlit.app/)**

A production-ready NLP application that detects **semantic plagiarism** in student
assignments—even when text has been paraphrased—using Sentence Transformers, cosine
similarity, and **FAISS vector search**.

---

## 📸 Screenshots

### Dashboard
![Dashboard](screenshots/screenshot_1_dashboard.png)

### Plagiarism Warnings
![Warnings](screenshots/screenshot_2_warnings.png)

### Similarity Heatmap
![Heatmap](screenshots/screenshot_3_heatmap.png)


---

## ✨ Features

| Feature | Detail |
|---|---|
| **Semantic understanding** | Detects paraphrased plagiarism, not just copy-paste |
| **Transformer embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual, accurate) |
| **FAISS vector search** | Adaptive indexing (Flat / IVF) — scales to thousands of assignments |
| **Paragraph chunking** | Detects localised section-level plagiarism |
| **Similarity matrix** | Full N×N pairwise document comparison; downloadable as CSV or Excel |
| **Interactive heatmap** | Plotly heatmap with hover tooltips; toggle to static Seaborn view |
| **Pair drill-down** | See exactly which paragraphs match |
| **Custom text query** | Paste any snippet to search against all uploaded assignments |
| **Authentication** | Login system with role-based access (admin / teacher) |
| **User management** | Admin can create, reset passwords, and delete users |
| **Streamlit dashboard** | Clean, teacher-friendly web interface |
| **Configurable threshold** | Adjustable via sidebar slider (default 0.59) |

---

## 🏗️ System Architecture

```
                   ┌─────────────────────────────────────────────────┐
                   │              Streamlit Dashboard                │
                   │                (app/streamlit_app.py)           │
                   └────────────────────┬────────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │                  Processing Pipeline                │
              │                                                     │
              │  PDF Upload → Text Extraction → Paragraph Chunking  │
              │    → Embedding → FAISS Index → Similarity → Flags   │
              └─────────────────────────────────────────────────────┘
                    │         │          │         │        │       │
              ┌─────▼──┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐ ┌▼─────┐ ┌▼──────┐
              │document│ │text_   │ │embed-  │ │faiss_ │ │simi- │ │heat-  │
              │_parser │ │chunking│ │ding_   │ │index  │ │larity│ │map.py │
              │.py     │ │.py     │ │model.py│ │.py    │ │.py   │ │       │
              └────────┘ └────────┘ └────────┘ └───────┘ └──────┘ └───────┘
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/core/document_parser.py` | Extract raw text from PDF, DOCX, and TXT files |
| `src/core/text_chunking.py` | Split text into paragraph chunks (20–200 words) |
| `src/core/embedding_model.py` | Generate L2-normalised embeddings via SentenceTransformers |
| `src/core/faiss_index.py` | Build FAISS index (Flat/IVF); chunk-level search across all documents |
| `src/core/similarity.py` | Compute cosine similarity matrices; flag plagiarism |
| `src/core/translator.py` | Translate non-English matching paragraphs to English |
| `src/db/auth.py` | SQLite-backed authentication with bcrypt password hashing |
| `src/db/corpus_db.py` | SQLite database manager for metadata, text chunks, and embedding vectors |
| `src/visualization/heatmap.py` | Render Seaborn/Plotly heatmaps (document-level & chunk-level) |
| `src/visualization/network_graph.py` | Render interactive Plotly plagiarism networks using spring layout |
| `app/streamlit_app.py` | Streamlit UI: login, upload, warnings, FAISS search, heatmap, drill-down |

---

## 📁 Project Structure

```
semantic_plagiarism_detector/
├── .github/                  # CI/CD workflows and issue templates
│   ├── ISSUE_TEMPLATE/       # Bug report and feature request forms
│   └── workflows/            # GitHub Actions CI and lint workflows
├── app/                      # Streamlit application interface
│   ├── components/           # Incident export and UI helper components
│   ├── streamlit_app.py      # Main Streamlit dashboard entrypoint
│   └── theme.py              # Visual design system and CSS injection
├── src/                      # Core backend source package
│   ├── core/                 # Parsing, chunking, embedding, FAISS & similarity
│   ├── db/                   # SQLite authentication, corpus & incident databases
│   ├── utils/                # PDF reports, warning lists, badges & caching
│   └── visualization/        # Seaborn/Plotly heatmaps and network graphs
├── tests/                    # Comprehensive unit and integration test suite
│   ├── app/                  # UI and dashboard smoke tests
│   ├── core/                 # Core NLP, translation, and indexing tests
│   ├── db/                   # Database authentication and corpus tests
│   ├── utils/                # PDF reports, email, and cache tests
│   └── visualization/        # Network graph and heatmap tests
├── docs/                     # Detailed setup guides and integration docs
├── evaluation/               # Benchmark dataset and evaluation harness
├── screenshots/              # Dashboard UI preview images
├── CHANGELOG.md              # Version release history
├── CODE_OF_CONDUCT.md        # Contributor Covenant v2.1
├── CONTRIBUTING.md           # Developer setup and contribution guidelines
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── SECURITY.md               # Vulnerability reporting policy
├── SUPPORT.md                # Help channels and FAQ
├── pytest.ini                # Pytest configuration
└── requirements.txt          # Python dependencies
```

---

## 🚀 Setup & Running

### 1. Clone / download the project

```bash
git clone https://github.com/your-org/semantic-plagiarism-detector.git
cd semantic-plagiarism-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `paraphrase-multilingual-MiniLM-L12-v2` model (~420 MB).
> Subsequent runs use the local cache.

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501**.

### Default credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin — full access + user management |

Additional users can be created from the **User Management** page (admin only).

---

## OCR support for scanned PDFs

Scanned and image-only PDFs are automatically detected page by page. Pages that
do not contain enough embedded text are rendered with PyMuPDF and processed
locally with Tesseract OCR. The extracted text then follows the same paragraph
chunking, embedding, FAISS, and similarity pipeline as regular PDFs.

### Python dependencies

```bash
python -m pip install pytesseract pymupdf pillow
```

### Tesseract system dependency

Tesseract must also be installed on the operating system.

On Windows, it is commonly installed at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

When it is not available on PATH, set:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify the installation:

```powershell
tesseract --version
```

OCR is performed locally; uploaded documents are not sent to an external OCR
service.

---

## 🖥️ Dashboard — 5 Tabs

| Tab | What it shows |
|---|---|
| **Plagiarism Warnings** | All flagged pairs sorted by severity (High / Medium); downloadable CSV |
| **FAISS Chunk Search** | Chunk-level ANN search across all documents; custom text query box |
| **Similarity Matrix** | Full N×N similarity table; downloadable as CSV or Excel |
| **Heatmap** | Interactive Plotly heatmap (hover values) or static Seaborn view; downloadable PNG |
| **Pair Drill-Down** | Select any two docs to see which specific paragraphs match |

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Plagiarism threshold | `0.59` | Pairs above this score are flagged |
| FAISS matches per chunk | `5` | Nearest neighbours retrieved per chunk |
| Chunk min words | `20` | Paragraphs shorter than this are discarded |
| Chunk max words | `200` | Longer paragraphs are sub-split at sentence boundaries |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | Change in `src/core/embedding_model.py` or set `SEMANTIC_PLAGIARISM_MODEL` |
| Batch size | `64` | Tune for GPU/CPU in `src/core/embedding_model.py` |

---

## 🧠 How It Works

### Step 1 – Text Extraction
PyPDF2 reads each PDF page and concatenates the text.

### Step 2 – Paragraph Chunking
Text is split on blank lines into chunks of 20–200 words.
Short chunks (headers, captions) are discarded; long chunks are sub-split at sentence boundaries.

### Step 3 – Embedding
Each chunk is passed through `paraphrase-multilingual-MiniLM-L12-v2`:
- Output: 384-dimensional, L2-normalised vector
- L2 normalisation means cosine similarity = dot product (fast)

### Step 4 – FAISS Index
All chunk vectors are added to a FAISS index. The system automatically selects the
best index type based on collection size:
- **< 5 000 vectors → `IndexFlatIP`** (exact inner-product search, O(N) per query)
- **≥ 5 000 vectors → `IndexIVFFlat`** (inverted-file approximate search, sub-linear per query)

Since embeddings are L2-normalised, inner product equals cosine similarity.

### Step 5 – Similarity Computation
- **Document-level:** mean-pooled chunk embeddings → cosine similarity matrix
- **Chunk-level:** FAISS ANN search → max similarity per chunk pair

### Step 6 – Flagging
Pairs with similarity >= threshold are flagged:
- **High**: >= 0.90
- **Medium**: >= 0.75 (default)

### Why semantic similarity catches paraphrasing
The model encodes **meaning**, not surface words:
> "The quick brown fox jumped over the lazy dog."
> "A nimble auburn canine leapt above a lethargic hound."

Both sentences produce nearly identical embeddings because the semantic content is the same.

---

## 📊 Performance

| Scenario | Expected time |
|---|---|
| First load (model download) | ~30–60 s (once only) |
| 5 documents, CPU | ~10–15 s |
| 10 documents, CPU | ~20–30 s |
| 10 documents, GPU | ~5–8 s |
| 1000 documents, FAISS | Feasible — auto-switches to IVF index |

Results are **cached by Streamlit** — re-uploading the same files is instant.

---

## 🔒 Privacy & Ethics

- All processing runs **locally**; no data leaves your machine.
- This tool is an **aid** for academic review, not a final verdict.
- A high similarity score should prompt **manual review**, not automatic sanctions.
- Consider informing students that submitted work will be checked.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained transformer embeddings |
| `faiss-cpu` | Vector search (exact / approximate nearest-neighbour) |
| `PyPDF2` | PDF text extraction |
| `streamlit` | Web dashboard |
| `bcrypt` | Password hashing for authentication |
| `python-dotenv` | Load environment variables from `.env` |
| `numpy` | Numerical operations |
| `pandas` | Similarity DataFrame |
| `scikit-learn` | `cosine_similarity` utility |
| `plotly` | Interactive heatmap with hover tooltips |
| `seaborn` | Static heatmap styling |
| `matplotlib` | Figure rendering |
| `openpyxl` | Excel export for similarity matrix |

---

## 📊 Evaluation & Benchmarks

The system is evaluated on a **25-pair benchmark dataset** covering heavy paraphrases,
light paraphrases, same-topic originals, and different-topic negatives.

Run the evaluation yourself:

```bash
python -m evaluation.evaluate
```

For benchmark schema, contributor guidance, threshold sweeps, and output details,
see the [Evaluation and Benchmark Dataset Guide](evaluation/README.md).

Results are saved to `evaluation/results/` and include:

| Output | Description |
|---|---|
| `metrics.json` | Precision, recall, F1, ROC-AUC at optimal threshold |
| `threshold_sweep_semantic.csv` | Metrics at every threshold (0.30 – 0.95) |
| `roc_curve.png` | ROC curve — Semantic vs TF-IDF baseline |
| `pr_curve.png` | Precision-Recall curve |
| `similarity_distribution.png` | Score histograms by label |

### Benchmark Results

Evaluated on 25 text pairs (10 plagiarized, 15 not plagiarized):

| Metric | Sentence Transformers | TF-IDF Baseline | Δ |
|---|---|---|---|
| **ROC-AUC** | **1.000** | 0.973 | +0.027 |
| **Best F1** | **1.000** | 0.667 | +0.333 |
| Precision | 1.000 | 1.000 | — |
| Recall | **1.000** | 0.500 | +0.500 |
| Accuracy | **1.000** | 0.800 | +0.200 |
| Optimal Threshold | 0.59 | 0.30 | — |

**Key finding:** TF-IDF misses **all 5 heavy paraphrases** (scoring 0.18–0.27) while
Sentence Transformers correctly flags them (scoring 0.60–0.82). Light paraphrases are
detected by both, but the semantic model provides much stronger signal separation.

### Why semantic beats lexical

The TF-IDF baseline relies on exact word overlap — it fails when students paraphrase.
Sentence Transformers encode **meaning**, catching paraphrases that surface-level
methods miss entirely.

---

## 📄 License

MIT License. Free for academic and educational use.
