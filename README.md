# RAG (Read data from pdf and csv using chatbots)

A Retrieval-Augmented Generation (RAG) chatbot for customer support. It loads a knowledge base from CSV and/or PDF files, stores embeddings in a local ChromaDB vector database, and uses Anthropic's Claude model to answer user questions grounded in that knowledge base.

The system supports three entry points:
- A Streamlit chat UI (`chatbot.py`)
- A command-line script for asking questions against the CSV knowledge base (`ask_question.py`)
- A command-line script for asking questions against any PDF file (`ask_pdf_question.py`)

---

## 1. Features

- Load knowledge from CSV (Q&A pairs) or PDF documents
- Local vector search via ChromaDB using `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings
- Hybrid retrieval: semantic search + keyword substring search for exact-token lookups (names, IDs, card numbers)
- Answer generation using Anthropic's Claude (`claude-sonnet-4-20250514` by default)
- Source citations, confidence score, and token usage reported for every answer
- Streamlit UI with conversation history and on-the-fly PDF ingestion

---

## 2. Project Structure

```
customer-support-rag/
├── chatbot.py              # Streamlit chat UI
├── ask_question.py         # CLI: ask a question against data/test_kb.csv
├── ask_pdf_question.py     # CLI: ask a question against any PDF
├── test_rag.py             # End-to-end smoke test with sample data
├── requirements.txt        # Python dependencies
├── .env                    # API_KEY lives here (not checked in)
├── data/
│   └── test_kb.csv         # Default knowledge-base CSV
├── src/
│   └── rag_system.py       # Core RAG engine (CustomerSupportRAG)
└── chroma_db/              # Local vector store (auto-created, gitignored)
```

---

## 3. Prerequisites

- **Python 3.10 or newer** (3.11 recommended)
- **pip** (comes with Python)
- An **Anthropic API key** — get one at https://console.anthropic.com/
- ~1 GB free disk space (the sentence-transformers model downloads on first run)

Optional but recommended:
- `git` for version control
- A virtual environment tool (`venv`, `virtualenv`, or `conda`)

---

## 4. Installation

### Step 1 — Clone or open the project

```bash
cd customer-support-rag
```

### Step 2 — Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `anthropic` — Claude API client
- `chromadb` — local vector database
- `pandas` — CSV loading
- `pdfplumber` — PDF text extraction
- `sentence-transformers` — embedding model
- `python-dotenv` — loads `.env` into environment variables
- `requests`, `numpy`
- `streamlit` — chat UI

### Step 4 — Configure your API key

Create a `.env` file in the project root (same folder as `chatbot.py`):

```bash
API_KEY=sk-ant-your-anthropic-api-key-here
```

The `.env` file is already listed in `.gitignore` and will not be committed.

---

## 5. Running the Project

Make sure your virtual environment is activated and `.env` is in place before running any of the commands below.

### Option A — Streamlit chat UI (recommended)

```bash
streamlit run chatbot.py
```

This opens the chatbot in your browser (usually at http://localhost:8501). On first launch it:

1. Loads `data/test_kb.csv` into the vector store
2. Loads every `*.pdf` found in `data/` into the vector store
3. Shows a chat window where you can ask questions

The sidebar also lets you ingest additional PDFs by pasting an absolute file path.

### Option B — Ask a single question from the CLI (CSV knowledge base)

```bash
python ask_question.py "How do I reset my password?"
```

Or run without arguments and you will be prompted:

```bash
python ask_question.py
```

### Option C — Ask a question against a PDF

```bash
python ask_pdf_question.py /path/to/your.pdf "What is the policy number?"
```

Or run without arguments and you will be prompted for both the file path and the question:

```bash
python ask_pdf_question.py
```

### Option D — Run the smoke test

Verifies the full RAG pipeline (embeddings + ChromaDB + Claude) works end to end using three sample questions:

```bash
python test_rag.py
```

---

## 6. Using Your Own Knowledge Base

### CSV format

Put a CSV at `data/test_kb.csv` (or any path — pass it to `load_knowledge_base_from_csv`) with these columns:

| Column     | Required | Description                                |
|------------|----------|--------------------------------------------|
| `question` | yes      | The user question or topic                 |
| `answer`   | yes      | The ground-truth answer                    |
| `category` | no       | Free-text label used in source citations   |
| `tags`     | no       | Comma-separated tags                       |

### PDF ingestion

Drop one or more `.pdf` files into the `data/` folder and restart the Streamlit app — they will be chunked and indexed automatically. You can also ingest a PDF at runtime from the Streamlit sidebar, or from the CLI with `ask_pdf_question.py`.

---

## 7. Troubleshooting

**`API_KEY is not set` error**
Make sure `.env` exists in the project root and contains `API_KEY=...`. Restart the app after editing `.env`.

**First run is very slow**
The `all-MiniLM-L6-v2` sentence-transformer model is ~90 MB and downloads once on first use. Subsequent runs are fast.

**Stale answers / want to reset the vector store**
Delete the `chroma_db/` folder and restart — it will be rebuilt from the source files.

```bash
rm -rf chroma_db
```

**`ModuleNotFoundError`**
Your virtual environment is probably not activated. Reactivate it (see Step 2) and confirm with `which python` (macOS/Linux) or `where python` (Windows).

**Streamlit port already in use**
Start on a different port: `streamlit run chatbot.py --server.port 8502`

---

## 8. Configuration Notes

- **Model** — the Claude model is set in `src/rag_system.py` (`self.llm_model`). Change it there if you want a different Claude variant.
- **Embeddings** — `all-MiniLM-L6-v2` is loaded in `src/rag_system.py`. Swap to any sentence-transformers model if you want higher quality at the cost of speed.
- **Vector DB location** — `chroma_db/` in the project root. Safe to delete; it will be rebuilt.

---

## 9. License

Internal project — add a license here if you plan to distribute it.
