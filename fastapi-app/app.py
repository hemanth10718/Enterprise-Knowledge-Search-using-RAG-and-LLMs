from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, io, re
from typing import List, Dict, Any, Optional

# PDF/DOCX readers
import PyPDF2
import docx

# Vector search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# PostgreSQL
import psycopg2
from dotenv import load_dotenv

# -------------------------
# Config
# -------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to the .env file.")

app = FastAPI(title="Resume RAG - PostgreSQL + FAISS + Chat Memory")

# Allow React (localhost:3000) to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Embedding model + in-memory FAISS index
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = model.get_sentence_embedding_dimension()

# IndexFlatIP + normalized vectors = cosine similarity (higher is better)
index = faiss.IndexFlatIP(DIM)
index_ids: List[int] = []  # resume ids, aligned with the vectors in the index


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


# -------------------------
# PostgreSQL helpers
# -------------------------
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resumes (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def insert_resume(filename: str, content: str) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO resumes (filename, content) VALUES (%s, %s) RETURNING id",
                (filename, content),
            )
            rid = cur.fetchone()[0]
        conn.commit()
        return rid
    finally:
        conn.close()


def get_resume_by_id(resume_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, content, uploaded_at FROM resumes WHERE id = %s",
                (resume_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "content": row[2],
        "uploaded_at": row[3].isoformat(),
    }


def get_resumes_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, content FROM resumes WHERE id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "filename": r[1], "content": r[2]} for r in rows]


def fetch_all_resumes() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, content FROM resumes ORDER BY id")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "filename": r[1], "content": r[2]} for r in rows]


def delete_resume_row(resume_id: int) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM resumes WHERE id = %s", (resume_id,))
            affected = cur.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()


# -------------------------
# Index management
# -------------------------
def rebuild_index():
    """Rebuild the FAISS index from PostgreSQL (used at startup and after deletes)."""
    global index, index_ids
    rows = fetch_all_resumes()
    index = faiss.IndexFlatIP(DIM)
    index_ids = []
    if rows:
        vecs = embed_texts([r["content"] for r in rows])
        index.add(vecs)
        index_ids = [r["id"] for r in rows]


def add_resume_to_index(text: str, resume_id: int):
    index.add(embed_texts([text]))
    index_ids.append(resume_id)


def extract_citations(resume_text: str, query: str, max_snippets: int = 2) -> List[str]:
    # Whole-word matching, so "ai" does not match inside words like "maintain"
    terms = [t for t in re.findall(r"[a-z0-9+#.]+", query.lower()) if len(t) > 2]
    sentences = re.split(r"(?<=[.!?]) +", resume_text)
    matches = []
    for s in sentences:
        low = s.lower()
        if any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in terms):
            matches.append(s.strip())
        if len(matches) >= max_snippets:
            break
    return matches if matches else [resume_text[:200] + "..."]


def search_index(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    scores, positions = index.search(embed_texts([query]), k)
    hits = [(int(p), float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]
    docs = {d["id"]: d for d in get_resumes_by_ids([index_ids[p] for p, _ in hits])}

    results = []
    for rank, (pos, score) in enumerate(hits, start=1):
        rid = index_ids[pos]
        doc = docs.get(rid)
        if not doc:
            continue
        results.append(
            {
                "rank": rank,
                "score": round(score, 4),  # cosine similarity, higher is better
                "resume_id": rid,
                "filename": doc["filename"],
                "citations": extract_citations(doc["content"], query),
            }
        )
    return results


# -------------------------
# Text extraction
# -------------------------
def extract_pdf_bytes(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])


def extract_docx_bytes(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in document.paragraphs])


def clean_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()


# -------------------------
# Startup
# -------------------------
init_db()
rebuild_index()

# -------------------------
# Conversation memory (simple, shared, in-memory)
# -------------------------
chat_history: List[str] = []


@app.post("/chat/")
def chat(user_query: str, top_k: int = 5):
    user_query = user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query required.")

    chat_history.append(user_query)
    context = " ".join(chat_history[-3:])  # last 3 queries as context

    return {
        "conversation": chat_history,
        "latest_query": user_query,
        "context": context,
        "results": search_index(context, top_k),
    }


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Resume RAG service running", "resumes_indexed": int(index.ntotal)}


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    content_bytes = await file.read()
    fname = file.filename or "uploaded_file"

    try:
        if fname.lower().endswith(".pdf"):
            raw_text = extract_pdf_bytes(content_bytes)
        elif fname.lower().endswith(".docx"):
            raw_text = extract_docx_bytes(content_bytes)
        else:
            raise HTTPException(status_code=400, detail="Only .pdf and .docx files supported")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text extracted. Possibly scanned/image PDF.")

    cleaned = clean_text(raw_text)
    resume_id = insert_resume(fname, cleaned)
    add_resume_to_index(cleaned, resume_id)

    return JSONResponse(
        {
            "id": resume_id,
            "filename": fname,
            "chars": len(cleaned),
            "cleaned_preview": cleaned[:800],
        }
    )


@app.get("/resumes/{resume_id}")
def get_resume(resume_id: int):
    r = get_resume_by_id(resume_id)
    if not r:
        raise HTTPException(status_code=404, detail="Resume not found")
    return r


@app.get("/semantic_search/")
def semantic_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' required.")
    return {"query": q, "results": search_index(q.strip(), top_k)}


@app.post("/match_text/")
def match_text(jd: Dict[str, str], top_k: int = 5):
    jd_text = jd.get("jd_text", "").strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="Job description text is required.")
    return {"jd": jd_text, "results": search_index(jd_text, top_k)}


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int):
    affected = delete_resume_row(resume_id)
    if affected == 0:
        raise HTTPException(status_code=404, detail="Resume not found")
    rebuild_index()  # remove the deleted resume's vector too
    return {"deleted": affected}