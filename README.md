\# Resume Search \& Chat (FastAPI + PostgreSQL + FAISS + LangChain + LangGraph)



A resume search and question-answering service. Upload resumes (PDF/DOCX), store them in PostgreSQL, split them into chunks, embed the chunks with a Sentence Transformer, and search them by meaning with FAISS. Ask questions in plain English and get an answer grounded in the retrieved resume text, with the source chunks returned alongside it.



Two question-answering endpoints are provided:



\- `/ask/` is a simple RAG chain: retrieve the best chunks, then generate an answer.

\- `/ask\_graph/` runs the same task as a LangGraph flow that grades the retrieved chunks and retries with a rewritten query when they are not relevant.



\---



\## Features



1\. \*\*Resume upload:\*\* accepts `.pdf` and `.docx` files and extracts the text (PyPDF2, python-docx).

2\. \*\*PostgreSQL storage:\*\* keeps the filename, full text, and upload timestamp for every resume.

3\. \*\*Chunked semantic search:\*\* resumes are split into \~700 character chunks (LangChain text splitter), embedded with `all-MiniLM-L6-v2` (pre-trained), and searched in FAISS using cosine similarity. Each resume is ranked by its best-matching chunk, and the matching chunks are returned as citations.

4\. \*\*RAG answers (`/ask/`):\*\* a LangChain chain (`prompt | LLM | parser`) answers using only the retrieved chunks, via a Groq-hosted LLM.

5\. \*\*LangGraph flow (`/ask\_graph/`):\*\* retrieve, grade the chunks, then answer, rewrite the query and retry (up to 2 retries, widening the search each time), or return a "not found" message. The response includes a `trace` of the steps taken.

6\. \*\*Job description matching:\*\* paste a job description and get the closest resumes.

7\. \*\*Chat-style search:\*\* the last three queries are combined as context.

8\. \*\*Delete support:\*\* removing a resume also removes its chunks from the search index.

9\. \*\*React frontend (optional):\*\* a simple UI for uploading and searching.



\---



\## Architecture



```mermaid

flowchart TD

&#x20;   A\[Upload .pdf / .docx] --> B\[FastAPI backend]

&#x20;   B --> C\[Extract and clean text]

&#x20;   C --> D\[(PostgreSQL: filename, content, timestamp)]

&#x20;   D --> E\[Split into chunks]

&#x20;   E --> F\[Sentence Transformer embeddings]

&#x20;   F --> G\[FAISS index in memory]

&#x20;   D -. rebuild at startup and after deletes .-> E

&#x20;   Q\[Search / question] --> B

&#x20;   B --> G

&#x20;   G --> R\[Top chunks with cosine scores]

&#x20;   R --> S\[Search results with citations]

&#x20;   R --> T\[LLM answer with sources]

```



\### The LangGraph flow (`/ask\_graph/`)



```mermaid

flowchart TD

&#x20;   S((start)) --> R\[retrieve]

&#x20;   R --> G\[grade: are the chunks relevant?]

&#x20;   G -- yes --> A\[generate answer]

&#x20;   G -- no, retries left --> W\[rewrite query]

&#x20;   W --> R

&#x20;   G -- no, out of retries or nothing indexed --> N\[not found]

&#x20;   A --> E((end))

&#x20;   N --> E

```



PostgreSQL is the source of truth. The FAISS index lives in memory and is rebuilt from PostgreSQL when the server starts and after a delete, so the two cannot drift apart. Scores are cosine similarity, so \*\*higher means a closer match\*\*.



\---



\## Tech Stack



\*\*Backend\*\*

\- Python 3.10

\- FastAPI, Uvicorn

\- PostgreSQL with `psycopg2`

\- Sentence Transformers (`all-MiniLM-L6-v2`, pre-trained, 384 dimensions)

\- FAISS (`faiss-cpu`)

\- LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-groq`)

\- LangGraph

\- Groq API for the LLM (model set in `.env`)

\- PyPDF2, python-docx, python-dotenv



\*\*Frontend (optional)\*\*

\- React 18, plain CSS, Fetch API



\---



\## Setup



\### 1. Prerequisites



\- Python 3.10

\- PostgreSQL 14 or newer (tested with 17)

\- A Groq API key from console.groq.com (needed only for `/ask/` and `/ask\_graph/`)

\- Node.js LTS (only for the frontend)



\### 2. Create the database



```bash

psql -U postgres -c "CREATE DATABASE resume\_rag;"

```



The app creates the `resumes` table automatically on first start.



\### 3. Configure the environment



Copy `fastapi-app/.env.example` to `fastapi-app/.env` and fill in your own values:



```

DATABASE\_URL=postgresql://postgres:your\_password@localhost:5432/resume\_rag

GROQ\_API\_KEY=your\_groq\_key

GROQ\_MODEL=openai/gpt-oss-120b

```



Use only letters and numbers in the database password, or URL-encode special characters. Never commit `.env` (it is in `.gitignore`). The model name is kept in `.env` so a retired model can be replaced without changing code. Restart the server after editing `.env`.



\### 4. Install and run the backend



```bash

cd fastapi-app

py -3.10 -m venv .venv

.venv\\Scripts\\activate          # Windows

\# source .venv/bin/activate     # Linux/Mac



pip install --upgrade pip

pip install -r requirements.txt



uvicorn app:app --reload

```



The API runs at `http://127.0.0.1:8000` and the interactive docs are at `http://127.0.0.1:8000/docs`. If port 8000 is taken, add `--port 8001`. The first start downloads the embedding model, so it needs internet access.



\*\*Windows tip:\*\* if installing PyTorch fails with a path-length error, create the virtual environment at a short path such as `C:\\ragenv`.



\### 5. Run the frontend (optional)



```bash

cd resume-frontend

npm install

npm start

```



The UI runs at `http://localhost:3000`. Check that the API base URL in `resume-frontend/src` matches the port the backend is running on.



\---



\## API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | `/` | Health check, number of indexed chunks, and the LLM model name |

| POST | `/upload/` | Upload a `.pdf` or `.docx` resume |

| GET | `/resumes/{resume\_id}` | Fetch a resume's stored text |

| DELETE | `/resumes/{resume\_id}` | Delete a resume and its chunks |

| GET | `/semantic\_search/?q=...\&top\_k=5` | Search resumes by meaning |

| POST | `/match\_text/` | Match a job description against resumes |

| POST | `/chat/?user\_query=...\&top\_k=5` | Search using the last three queries as context |

| POST | `/ask/?question=...\&top\_k=5` | RAG answer with source chunks |

| POST | `/ask\_graph/?question=...\&top\_k=3` | LangGraph RAG answer with a step-by-step trace |



\### Examples



On Windows PowerShell, use `curl.exe` instead of `curl`.



```bash

\# Upload

curl -X POST -F "file=@resume.pdf" http://127.0.0.1:8000/upload/



\# Search

curl "http://127.0.0.1:8000/semantic\_search/?q=python%20developer\&top\_k=3"



\# Ask a question (RAG)

curl -X POST "http://127.0.0.1:8000/ask/?question=Which%20resume%20has%20Terraform%20experience\&top\_k=5"



\# Ask with the LangGraph flow

curl -X POST "http://127.0.0.1:8000/ask\_graph/?question=Which%20resume%20has%20Terraform%20experience\&top\_k=3"

```



An `/ask\_graph/` response looks like this:



```json

{

&#x20; "question": "Which resume has Terraform experience?",

&#x20; "answer": "...",

&#x20; "search\_query\_used": "Which resume has Terraform experience?",

&#x20; "attempts": 1,

&#x20; "trace": \[

&#x20;   "retrieve(query='...', k=3) -> 3 chunks",

&#x20;   "grade -> YES",

&#x20;   "generate"

&#x20; ],

&#x20; "sources": \[{"resume\_id": 1, "filename": "...", "score": 0.54, "snippet": "..."}]

}

```



\---



\## Database Schema



```sql

CREATE TABLE resumes (

&#x20;   id          SERIAL PRIMARY KEY,

&#x20;   filename    TEXT NOT NULL,

&#x20;   content     TEXT NOT NULL,

&#x20;   uploaded\_at TIMESTAMPTZ NOT NULL DEFAULT NOW()

);

```



\---



\## Limitations



\- \*\*Embeddings measure meaning, not keywords.\*\* A question like "which resume has Excel experience" can score a database-skills chunk almost as high as the chunk that lists Excel. The LLM stage reads the retrieved text and filters this out, but retrieval scores alone do not separate the results sharply. Hybrid search (vector plus keyword) would help.

\- \*\*PDF text extraction is imperfect.\*\* PyPDF2 sometimes splits words or interleaves columns, which adds noise to chunks and citations.

\- \*\*Resume text is sent to a third-party LLM\*\* (Groq) when `/ask/` or `/ask\_graph/` is used. Only use sample or your own resumes.

\- \*\*Chat memory is a shared in-memory list.\*\* It is lost on restart and is not per user.

\- \*\*The FAISS index is in memory\*\* and rebuilt from PostgreSQL at startup, which is fine for small collections but not for very large ones.

\- \*\*No authentication.\*\* Resumes contain personal data, so run it locally or add access control before exposing it.

\- \*\*No automated evaluation or tests yet.\*\*



\---



\## Roadmap



\- Hybrid search using PostgreSQL full-text search alongside vectors.

\- Better PDF extraction.

\- Per-user chat sessions.

\- Automated tests and a small evaluation set for retrieval quality.

\- A `docker-compose.yml` running the API and PostgreSQL together.



\---



\## Author



Developed by Hemanth R



\## License



This project is licensed under the \*\*MIT License\*\*.

