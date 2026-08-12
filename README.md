# 🤖 Enterprise Document Q&A — Agentic RAG with Gemini 2.5

A Generative AI–powered application that lets users upload enterprise
documents (PDF, TXT, CSV, Excel) and ask natural-language questions about
them, answered by autonomous AI agents using a Retrieval-Augmented
Generation (RAG) pipeline built with **LangChain**, **ChromaDB**, and
**Google Gemini 2.5**.

---

## 1. Architecture

```
                       ┌─────────────────────────┐
                       │   Streamlit UI (app.py)  │  ← upload docs / ask Q's
                       └────────────┬─────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │   FastAPI (api/main.py)   │  ← optional REST layer
                       └────────────┬─────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────┐
        │              AgentOrchestrator (agents/)               │
        │   PLAN → RETRIEVE → REASON → RESPOND                   │
        │   ┌─────────────────┐   ┌─────────────────┐            │
        │   │ RetrievalAgent   │   │ ReasoningAgent   │            │
        │   └────────┬────────┘   └────────┬────────┘            │
        └────────────┼─────────────────────┼─────────────────────┘
                      │                     │
        ┌─────────────▼──────┐   ┌──────────▼───────────┐
        │ vector_store_service │   │   llm_service         │
        │ (Chroma, cosine)      │   │ (Gemini 2.5 Chat)     │
        └─────────────┬───────┘   └───────────────────────┘
                      │
        ┌─────────────▼───────┐
        │ embedding_service     │  (Gemini text-embedding-004)
        └───────────────────────┘

        ┌────────────────────────────────────────┐
        │ document_service (PDF/TXT/CSV/XLSX/DOCX) │
        │           → utils/text_splitter (chunking)│
        └────────────────────────────────────────┘
```

### Project structure

```
enterprise-doc-qa/
├── api/                 # Optional FastAPI REST interface (/upload, /ask)
│   └── main.py
├── services/             # Core business logic services
│   ├── document_service.py     # multi-format ingestion + text extraction
│   ├── embedding_service.py    # Gemini embeddings wrapper
│   ├── vector_store_service.py # Chroma persistence + similarity search
│   └── llm_service.py          # Gemini 2.5 chat wrapper
├── agents/               # Autonomous agent layer
│   ├── retrieval_agent.py      # embeds query + cosine similarity search
│   ├── reasoning_agent.py      # LLM generation + groundedness guardrail
│   └── agent_orchestrator.py   # plan → retrieve → reason → respond
├── core/                 # Cross-cutting concerns
│   ├── config.py               # loads .env, exposes `settings`
│   ├── exceptions.py           # custom exception hierarchy
│   └── guardrails.py           # input/output safety + hallucination checks
├── utils/                # Shared helpers
│   ├── text_splitter.py        # LangChain chunking strategy
│   ├── validators.py           # file/input validation
│   └── logger.py               # loguru logging setup
├── data/
│   ├── uploads/                 # saved uploaded files (gitignored content)
│   └── chroma_db/                # persisted Chroma vector DB (gitignored content)
├── env/
│   └── README.md                # how to create your local `env/` virtualenv
├── app.py                # Streamlit entrypoint
├── requirements.txt
├── .env.example           # environment variable template
├── .gitignore
├── Dockerfile
└── README.md
```

---

## 2. RAG Workflow

1. **User uploads a document** (PDF / TXT / CSV / XLSX / DOCX) via Streamlit.
2. **Document ingestion** (`document_service.py`) validates the file, saves
   it to `data/uploads/`, and extracts raw text per format.
3. **Chunking** (`utils/text_splitter.py`) splits text into overlapping
   chunks (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50` by default) using
   LangChain's `RecursiveCharacterTextSplitter`, conceptually similar to:
   ```python
   from langchain.text_splitter import CharacterTextSplitter
   splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
   chunks = splitter.split_text(text)
   ```
4. **Embedding + storage** (`embedding_service.py` + `vector_store_service.py`)
   embeds each chunk with Gemini's `text-embedding-004` model and persists
   it into a local **Chroma** vector database at `data/chroma_db/`
   (configured with cosine similarity).
5. **User asks a question** in the chat UI.
6. **Guardrails validate the question** (length, prompt-injection patterns,
   unsafe keywords) — `core/guardrails.py`.
7. **RetrievalAgent** embeds the question and performs cosine similarity
   search to fetch the top-N most relevant chunks.
8. **ReasoningAgent** builds a grounded prompt (system rules + retrieved
   context + question) and calls **Gemini 2.5** to generate an answer.
9. **Output guardrails** check that the answer's vocabulary sufficiently
   overlaps with the retrieved context (a lightweight hallucination check);
   if not, a disclaimer is appended.
10. The **AgentOrchestrator** returns the final answer, cited sources, and
    a step-by-step agent trace, all rendered in the Streamlit chat UI.

---

## 3. Setup

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### Steps

```bash
git clone <your-repo-url>
cd enterprise-doc-qa

# 1. Create and activate a virtual environment (see env/README.md)
python -m venv env
source env/bin/activate      # Windows: env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# edit .env and set GOOGLE_API_KEY=<your key>

# 4. Run the Streamlit app
streamlit run app.py
```

Open the app at `http://localhost:8501`, upload a document from the
sidebar, click **Process & Index Documents**, then ask questions in the
chat box.

### Optional: run the REST API

```bash
uvicorn api.main:app --reload --port 8000
# POST /upload  (multipart file)
# POST /ask     ({"question": "..."})
```

### Optional: Docker

```bash
docker build -t enterprise-doc-qa .
docker run -p 8501:8501 --env-file .env enterprise-doc-qa
```

### Deployment options
- **Streamlit Community Cloud**: connect the GitHub repo, set `GOOGLE_API_KEY`
  as a secret, point to `app.py`.
- **Docker on any cloud VM / Cloud Run / ECS**: use the provided `Dockerfile`.
- Ensure `data/chroma_db` is on a persistent volume in production so the
  vector index survives restarts.

---

## 4. Guardrails & Reliability

| Concern | Mitigation |
|---|---|
| Invalid/oversized/unsupported files | `utils/validators.py` rejects before ingestion |
| Empty or unparsable documents | `DocumentProcessingError` raised with a clear message |
| Prompt injection in questions | Regex pattern filter in `core/guardrails.py` |
| Unsafe / policy-violating questions | Keyword-based blocklist (extensible) |
| Hallucinated / ungrounded answers | Post-generation vocabulary-overlap check; disclaimer appended if below threshold |
| API/LLM/embedding failures | Custom exception hierarchy (`core/exceptions.py`) caught at every layer, surfaced as friendly UI errors, never a raw stack trace |
| No relevant content for a question | Retrieval agent short-circuits with an honest "not found" response instead of forcing the LLM to answer |

These are intentionally simple, rule-based, and auditable rather than a
black-box moderation model — they should be extended (e.g. with a proper
moderation API) before production use with sensitive data.

---

## 5. Limitations

- **Groundedness check is heuristic**, not a semantic entailment model — it
  can produce false positives/negatives on paraphrased answers.
- **No OCR**: scanned/image-only PDFs will fail extraction (no text layer).
- **Single-tenant vector store**: no per-user/document ACLs; anyone with UI
  access can query all indexed documents. Add auth + collection-per-tenant
  for multi-user enterprise deployment.
- **No re-ranking step**: retrieval is single-pass cosine similarity; adding
  a cross-encoder re-ranker would improve precision on large corpora.
- **No streaming responses** in the current UI (answers arrive as one block).
- **Prompt-injection filter is regex-based** and can be bypassed by creative
  phrasing — pair with model-level safety settings for production.
- **Local Chroma persistence** — for multi-instance/horizontally-scaled
  deployments, use a hosted vector DB (e.g. Chroma server mode, Pinecone,
  Vertex AI Vector Search) instead of the local persistent client.

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Optional API | FastAPI |
| Orchestration | LangChain |
| LLM | Google Gemini 2.5 (`gemini-2.5-flash`) |
| Embeddings | Google `text-embedding-004` |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Document parsing | pypdf, python-docx, pandas/openpyxl |
| Logging | loguru |
