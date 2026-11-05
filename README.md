# Resume Search & Chat (FastAPI + PostgreSQL + FAISS + LangChain + LangGraph)
 
A resume search and question-answering service. Upload resumes (PDF/DOCX), store them in PostgreSQL, split them into chunks, embed the chunks with a Sentence Transformer, and search them by meaning with FAISS. Ask questions in plain English and get an answer grounded in the retrieved resume text, with the source chunks returned alongside it.
 
Two question-answering endpoints are provided:
 
- `/ask/` is a simple RAG chain: retrieve the best chunks, then generate an answer.
- `/ask_graph/` runs the same task as a LangGraph flow that grades the retrieved chunks and retries with a rewritten query when they are not relevant.
---
 
## Features
 
1. **Resume upload:** accepts `.pdf` and `.docx` files and extracts the text (PyPDF2, python-docx).
2. **PostgreSQL storage:** keeps the filename, full text, and upload timestamp for every resume.
3. **Chunked semantic search:** resumes are split into ~700 character chunks (LangChain text splitter), embedded with `all-MiniLM-L6-v2` (pre-trained), and searched in FAISS using cosine similarity. Each resume is ranked by its best-matching chunk, and the matching chunks are returned as citations.
4. **RAG answers (`/ask/`):** a LangChain chain (`prompt | LLM | parser`) answers using only the retrieved chunks, via a Groq-hosted LLM.
5. **LangGraph flow (`/ask_graph/`):** retrieve, grade the chunks, then answer, rewrite the query and retry (up to 2 retries, widening the search each time), or return a "not found" message. The response includes a `trace` of the steps taken.
6. **Job description matching:** paste a job description and get the closest resumes.
7. **Chat-style search:** the last three queries are combined as context.
8. **Delete support:** removing a resume also removes its chunks from the search index.
9. **Retrieval evaluation:** a script that measures precision, recall, F1, hit rate and MRR against keyword and random baselines (see [Evaluation](#evaluation)).
10. **React frontend (optional):** a simple UI for uploading and searching.
---
 
## Architecture
 
```mermaid
flowchart TD
    A["Upload .pdf / .docx"] --> B["FastAPI backend"]
    B --> C["Extract and clean text"]
    C --> D[("PostgreSQL: filename, content, timestamp")]
    D --> E["Split into chunks"]
    E --> F["Sentence Transformer embeddings"]
    F --> G["FAISS index in memory"]
    D -. rebuild at startup and after deletes .-> E
    Q["Search / question"] --> B
    B --> G
    G --> R["Top chunks with cosine scores"]
    R --> S["Search results with citations"]
    R --> T["LLM answer with sources"]
```
 
### The LangGraph flow (`/ask_graph/`)
 
```mermaid
flowchart TD
    S((start)) --> R["retrieve"]
    R --> G["grade: are the chunks relevant?"]
    G -- yes --> A["generate answer"]
    G -- "no, retries left" --> W["rewrite query"]
    W --> R
    G -- "no, out of retries or nothing indexed" --> N["not found"]
    A --> E((end))
    N --> E
```
 
PostgreSQL is the source of truth. The FAISS index lives in memory and is rebuilt from PostgreSQL when the server starts and after a delete, so the two cannot drift apart. Scores are cosine similarity, so **higher means a closer match**.
 
---
 
## Tech Stack
 
**Backend**
 
- Python 3.10
- FastAPI, Uvicorn
- PostgreSQL with `psycopg2`
- Sentence Transformers (`all-MiniLM-L6-v2`, pre-trained, 384 dimensions)
- FAISS (`faiss-cpu`)
- LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-groq`)
- LangGraph
- Groq API for the LLM (model set in `.env`)
- PyPDF2, python-docx, python-dotenv
**Frontend (optional)**
 
- React 18, plain CSS, Fetch API
---
 
## Project structure
 
```
fastapi-app/
  app.py                    FastAPI app, search, RAG chain and endpoints
  rag_graph.py              LangGraph flow (retrieve, grade, rewrite, generate)
  requirements.txt          Pinned dependencies
  .env.example              Template for the settings you must provide
  eval/
    make_eval_data.py       Creates the synthetic resumes and labeled queries
    evaluate_retrieval.py   Computes retrieval metrics against baselines
    eval_queries.json       The labeled queries
resume-frontend/            Optional React UI
```
 
---
 
## Setup
 
### 1. Prerequisites
 
- Python 3.10
- PostgreSQL 14 or newer (tested with 17)
- A Groq API key from console.groq.com (needed only for `/ask/` and `/ask_graph/`)
- Node.js LTS (only for the frontend)
### 2. Create the database
 
```bash
psql -U postgres -c "CREATE DATABASE resume_rag;"
```
 
The app creates the `resumes` table automatically on first start.
 
### 3. Configure the environment
 
Copy `fastapi-app/.env.example` to `fastapi-app/.env` and fill in your own values:
 
```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/resume_rag
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```
 
Use only letters and numbers in the database password, or URL-encode special characters. Never commit `.env` (it is in `.gitignore`). The model name is kept in `.env` so a retired model can be replaced without changing code. Restart the server after editing `.env`.
 
### 4. Install and run the backend
 
```bash
cd fastapi-app
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
 
pip install --upgrade pip
pip install -r requirements.txt
 
uvicorn app:app --reload
```
 
The API runs at `http://127.0.0.1:8000` and the interactive docs are at `http://127.0.0.1:8000/docs`. If port 8000 is taken, add `--port 8001`. The first start downloads the embedding model, so it needs internet access.
 
**Windows tip:** if installing PyTorch fails with a path-length error, create the virtual environment at a short path such as `C:\ragenv`.
 
### 5. Run the frontend (optional)
 
```bash
cd resume-frontend
npm install
npm start
```
 
The UI runs at `http://localhost:3000`. Check that the API base URL in `resume-frontend/src` matches the port the backend is running on.
 
---
 
## API Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, number of indexed chunks, and the LLM model name |
| POST | `/upload/` | Upload a `.pdf` or `.docx` resume |
| GET | `/resumes/{resume_id}` | Fetch a resume's stored text |
| DELETE | `/resumes/{resume_id}` | Delete a resume and its chunks |
| GET | `/semantic_search/?q=...&top_k=5` | Search resumes by meaning |
| POST | `/match_text/` | Match a job description against resumes |
| POST | `/chat/?user_query=...&top_k=5` | Search using the last three queries as context |
| POST | `/ask/?question=...&top_k=5` | RAG answer with source chunks |
| POST | `/ask_graph/?question=...&top_k=3` | LangGraph RAG answer with a step-by-step trace |
 
### Examples
 
On Windows PowerShell, use `curl.exe` instead of `curl`.
 
```bash
# Upload
curl -X POST -F "file=@resume.pdf" http://127.0.0.1:8000/upload/
 
# Search
curl "http://127.0.0.1:8000/semantic_search/?q=python%20developer&top_k=3"
 
# Ask a question (RAG)
curl -X POST "http://127.0.0.1:8000/ask/?question=Which%20resume%20has%20Terraform%20experience&top_k=5"
 
# Ask with the LangGraph flow
curl -X POST "http://127.0.0.1:8000/ask_graph/?question=Which%20resume%20has%20Terraform%20experience&top_k=3"
```
 
An `/ask_graph/` response looks like this:
 
```json
{
  "question": "Which resume has Terraform experience?",
  "answer": "...",
  "search_query_used": "Which resume has Terraform experience?",
  "attempts": 1,
  "trace": [
    "retrieve(query='...', k=3) -> 3 chunks",
    "grade -> YES",
    "generate"
  ],
  "sources": [{"resume_id": 1, "filename": "...", "score": 0.54, "snippet": "..."}]
}
```
 
---
 
## Database Schema
 
```sql
CREATE TABLE resumes (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    content     TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```
 
---
 
## Evaluation
 
Retrieval quality is measured with the scripts in `fastapi-app/eval/`.
 
**Test data.** `make_eval_data.py` creates 24 fictional resumes (one role each, such as DevOps engineer, data scientist, recruiter, accountant) and 31 queries, each labeled with the resumes that are correct answers. Six of the queries are marked "hard": everyday phrasing with very little word overlap with the resumes (for example, "someone who can stop our website from going down at night"), so they test meaning and not shared keywords.
 
**Metrics.** For k = 1, 3 and 5: precision@k, recall@k, F1@k and hit rate@k, plus MRR@5 (how high the first correct resume ranks, counted as 0 if it is outside the top 5). The script prints two reference points next to them: the exact expected score of a random ranking, and a plain keyword (TF-IDF) ranking over the same resumes. Hard queries are also summarized separately.
 
**Reproduce it.** Use a separate database so your own resumes are untouched (PowerShell, from `fastapi-app`):
 
```powershell
psql -U postgres -c "CREATE DATABASE resume_rag_eval;"
 
# start a second server on the evaluation database
$env:DATABASE_URL = "postgresql://postgres:your_password@localhost:5432/resume_rag_eval"
python -m uvicorn app:app --port 8003
 
# in another window, from fastapi-app/eval
python make_eval_data.py --upload http://127.0.0.1:8003
python evaluate_retrieval.py eval_queries.json http://127.0.0.1:8003 --tfidf-dir sample_resumes
```
 
The script prints a per-query table, the metrics with both baselines, the queries where no correct resume appeared in the top 3, and a Markdown table. Per-query results are saved to `retrieval_eval_results.csv`.
 
**How to read the results.** The resumes and queries are synthetic and were written together, so the numbers show whether the search separates distinct roles, not how it would perform on real resumes. The hard queries are the meaningful test: on easy queries even a keyword search does well.
 
---
 
## Limitations
 
- **Embeddings measure meaning, not keywords.** A question like "which resume has Excel experience" can score a database-skills chunk almost as high as the chunk that lists Excel. The LLM stage reads the retrieved text and filters this out, but retrieval scores alone do not separate the results sharply. Hybrid search (vector plus keyword) would help.
- **PDF text extraction is imperfect.** PyPDF2 sometimes splits words or interleaves columns, which adds noise to chunks and citations.
- **Resume text is sent to a third-party LLM** (Groq) when `/ask/` or `/ask_graph/` is used. Only use sample or your own resumes.
- **Chat memory is a shared in-memory list.** It is lost on restart and is not per user.
- **The FAISS index is in memory** and rebuilt from PostgreSQL at startup, which is fine for small collections but not for very large ones.
- **No authentication.** Resumes contain personal data, so run it locally or add access control before exposing it.
- **The evaluation uses synthetic resumes written alongside the queries,** so it is not a measure of real-world quality. There are no automated unit tests yet.
---
 
## Roadmap
 
- Hybrid search using PostgreSQL full-text search alongside vectors.
- Better PDF extraction.
- Per-user chat sessions.
- Evaluate on real resumes with independently written queries.
- A `docker-compose.yml` running the API and PostgreSQL together.
---
 
## Security notes
 
- Keep API keys and database passwords in `.env` only. Never commit them or include them in screenshots.
- Use a separate Groq key per project so one can be deleted without affecting the others.
- Resumes contain personal data. Take screenshots from the synthetic evaluation database, not from real CVs.
---
 
## Author
 
Developed by Hemanth R
 
## License
 
This project is licensed under the **MIT License**.
 
