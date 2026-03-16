# 🔍 Semantic Plagiarism Detection System

A production-ready NLP application that detects **semantic plagiarism** in student
assignments—even when text has been paraphrased—using Sentence Transformers and
cosine similarity.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Semantic understanding** | Detects paraphrased plagiarism, not just copy-paste |
| **Transformer embeddings** | `all-MiniLM-L6-v2` (384-dim, fast, accurate) |
| **Paragraph chunking** | Detects localised section-level plagiarism |
| **Similarity matrix** | Full N×N pairwise document comparison |
| **Heatmap visualisation** | Green–Red heatmap with flagged-pair borders |
| **Pair drill-down** | See exactly which paragraphs match |
| **Streamlit dashboard** | Clean, teacher-friendly web interface |
| **Configurable threshold** | Adjustable via sidebar slider (default 0.75) |

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
              │         → Embedding → Similarity → Flagging         │
              └──────────────────────────────────────────────────── ┘
                    │           │           │         │        │
              ┌─────▼──┐  ┌────▼───┐  ┌────▼───┐ ┌──▼─────┐ ┌▼───────┐
              │pdf_    │  │text_   │  │embed-  │ │simila- │ │heat-   │
              │reader  │  │chunking│  │ding_   │ │rity.py │ │map.py  │
              │.py     │  │.py     │  │model.py│ │        │ │        │
              └────────┘  └────────┘  └────────┘ └────────┘ └────────┘
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `utils/pdf_reader.py` | Extract raw text from PDFs via PyPDF2 |
| `utils/text_chunking.py` | Split text into paragraph chunks (20–200 words) |
| `utils/embedding_model.py` | Generate L2-normalised embeddings via SentenceTransformers |
| `utils/similarity.py` | Compute cosine similarity matrices; flag plagiarism |
| `utils/heatmap.py` | Render Seaborn heatmaps (document-level & chunk-level) |
| `app/streamlit_app.py` | Streamlit UI: upload, display, drill-down |

---

## 📁 Project Structure

```
semantic_plagiarism_detector/
│
├── utils/
│   ├── __init__.py          # Package exports
│   ├── pdf_reader.py        # PDF text extraction
│   ├── text_chunking.py     # Paragraph-level chunking
│   ├── embedding_model.py   # Sentence Transformer wrapper
│   ├── similarity.py        # Cosine similarity & plagiarism flagging
│   └── heatmap.py           # Matplotlib/Seaborn visualisations
│
├── app/
│   └── streamlit_app.py     # Main web dashboard
│
├── requirements.txt
└── README.md
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

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB).
> Subsequent runs use the local cache.

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501**.

---

## 🖥️ Dashboard Usage

1. **Upload PDFs** – Use the file uploader to add 2–20 student assignment PDFs.
2. **Wait for processing** – Text extraction, chunking, and embedding happen automatically (cached after first run).
3. **⚠️ Plagiarism Warnings tab** – See all flagged pairs sorted by severity.
4. **📋 Similarity Matrix tab** – View the full N×N similarity table; download as CSV.
5. **🗺️ Heatmap tab** – Visual overview; download as PNG.
6. **🔬 Pair Drill-Down tab** – Select any two documents to see which paragraphs are most similar.
7. **Adjust threshold** – Use the sidebar slider to tighten or loosen detection sensitivity.

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Plagiarism threshold | `0.75` | Pairs above this score are flagged |
| Chunk min words | `20` | Paragraphs shorter than this are discarded |
| Chunk max words | `200` | Longer paragraphs are sub-split at sentence boundaries |
| Embedding model | `all-MiniLM-L6-v2` | Change in `utils/embedding_model.py` |
| Batch size | `64` | Tune for GPU/CPU in `embedding_model.py` |

---

## 🧠 How Semantic Detection Works

### Step 1 – Text Extraction
PyPDF2 reads each PDF page and concatenates text.

### Step 2 – Paragraph Chunking
Text is split on blank lines into paragraph chunks (20–200 words).
Chunks shorter than 20 words (headers, captions) are discarded.
Overly long chunks are sub-split at sentence boundaries.

### Step 3 – Embedding
Each chunk is passed through `all-MiniLM-L6-v2`:
- Output: 384-dimensional L2-normalised vector per chunk
- L2 normalisation means cosine similarity = dot product (fast)

### Step 4 – Similarity Computation
**Document-level:** Each document → mean of its chunk embeddings → cosine similarity matrix  
**Chunk-level (optional):** Maximum pairwise chunk similarity → catches partial plagiarism

### Step 5 – Flagging
Pairs with similarity ≥ threshold are flagged:
- 🔴 **High**: ≥ 0.90
- 🟡 **Medium**: ≥ 0.75 (default threshold)

### Why semantic similarity catches paraphrasing
The model encodes **meaning** rather than surface text:
> "The quick brown fox jumped over the lazy dog."  
> "A nimble auburn canine leapt above a lethargic hound."

Both produce very similar embeddings because the semantic content is identical.

---

## 📊 Performance Notes

| Scenario | Expected time |
|---|---|
| First load (model download) | ~30–60 s (once) |
| 5 documents, CPU | ~10–15 s |
| 10 documents, CPU | ~20–30 s |
| 10 documents, GPU | ~5–8 s |

Results are **cached by Streamlit**, so re-uploads of the same files are instant.

---

## 🔒 Privacy & Ethics

- All processing runs **locally**; no data is sent to external servers.
- This tool is an **aid** for academic review, not a final verdict.
- A high similarity score should prompt **manual review**, not automatic sanctions.
- Consider informing students that their work will be checked.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained transformer embeddings |
| `PyPDF2` | PDF text extraction |
| `streamlit` | Web dashboard |
| `numpy` | Numerical operations |
| `pandas` | Similarity DataFrame |
| `scikit-learn` | `cosine_similarity` utility |
| `seaborn` | Heatmap styling |
| `matplotlib` | Figure rendering |

---

## 📄 License

MIT License. Free for academic and educational use.
