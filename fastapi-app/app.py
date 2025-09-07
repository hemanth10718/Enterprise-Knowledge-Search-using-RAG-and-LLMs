from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, os, io, datetime, pickle, re
from typing import List, Dict, Any

# PDF/DOCX readers
import PyPDF2
import docx

# FAISS
import faiss
import numpy as np

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "resumes.db")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss.index")
FAISS_META_FILE  = os.path.join(BASE_DIR, "faiss_meta.pkl")

app = FastAPI(title="Resume RAG - SQLite + FAISS + Chat Memory")

# Allow React (localhost:3000) to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DB initialization
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------
# FAISS Helpers
# -------------------------
dimension = 384  # embedding size

def save_faiss(index, metadata):
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(metadata, f)

def load_faiss():
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_META_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        index = faiss.IndexFlatL2(dimension)
        return index, []

faiss_index, faiss_meta = load_faiss()

def embed_text(text: str) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))  # deterministic
    return np.random.rand(1, dimension).astype("float32")

def add_resume_to_faiss(text: str, resume_id: int, filename: str):
    global faiss_index, faiss_meta
    vec = embed_text(text)
    faiss_index.add(vec)
    faiss_meta.append({"id": resume_id, "filename": filename, "content": text})
    save_faiss(faiss_index, faiss_meta)

def extract_citations(resume_text: str, query: str, max_snippets: int = 2) -> List[str]:
    q_terms = query.lower().split()
    sentences = re.split(r'(?<=[.!?]) +', resume_text)
    matches = []
    for s in sentences:
        for qt in q_terms:
            if qt in s.lower():
                matches.append(s.strip())
                break
        if len(matches) >= max_snippets:
            break
    return matches if matches else [resume_text[:200] + "..."]

def search_faiss(query: str, top_k=5):
    if faiss_index.ntotal == 0:
        return []
    vec = embed_text(query)
    D, I = faiss_index.search(vec, top_k)
    results = []
    for pos, idx in enumerate(I[0]):
        if idx == -1 or idx >= len(faiss_meta):
            continue
        meta = faiss_meta[idx]
        results.append({
            "rank": pos + 1,
            "score": float(D[0][pos]),
            "resume_id": meta.get("id"),
            "filename": meta.get("filename"),
            "citations": extract_citations(meta.get("content", ""), query)
        })
    return results

# -------------------------
# Text extraction
# -------------------------
def extract_pdf_bytes(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_docx_bytes(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()

# -------------------------
# DB helpers
# -------------------------
def insert_resume(filename: str, content: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO resumes (filename, content, uploaded_at) VALUES (?, ?, ?)",
        (filename, content, datetime.datetime.utcnow().isoformat() + "Z"),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def get_resume_by_id(resume_id: int) -> Dict[str, Any] | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename, content, uploaded_at FROM resumes WHERE id = ?", (resume_id,))
    row = cur.fetchone()
    conn.close()
    if not row: return None
    return {"id": row[0], "filename": row[1], "content": row[2], "uploaded_at": row[3]}

# -------------------------
# Conversation Memory
# -------------------------
chat_history = []  # store queries

@app.post("/chat/")
def chat(user_query: str, top_k: int = 5):
    global chat_history
    user_query = user_query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query required.")

    # Add query to chat history
    chat_history.append(user_query)

    # Use last 3 queries as context
    context = " ".join(chat_history[-3:])

    # Run semantic search with context
    results = search_faiss(context, top_k)

    return {
        "conversation": chat_history,
        "latest_query": user_query,
        "context": context,
        "results": results
    }

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Resume RAG service running", "db": DB_PATH}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    content_bytes = await file.read()
    fname = file.filename or "uploaded_file"

    if fname.lower().endswith(".pdf"):
        raw_text = extract_pdf_bytes(content_bytes)
    elif fname.lower().endswith(".docx"):
        raw_text = extract_docx_bytes(content_bytes)
    else:
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files supported")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text extracted. Possibly scanned/ image PDF.")

    cleaned = clean_text(raw_text)
    resume_id = insert_resume(fname, cleaned)
    add_resume_to_faiss(cleaned, resume_id, fname)

    return JSONResponse({
        "id": resume_id,
        "filename": fname,
        "chars": len(cleaned),
        "cleaned_preview": cleaned[:800]
    })

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
    return {"query": q, "results": search_faiss(q.strip(), top_k)}

@app.post("/match_text/")
def match_text(jd: Dict[str, str], top_k: int = 5):
    jd_text = jd.get("jd_text", "").strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="Job description text is required.")
    return {"jd": jd_text, "results": search_faiss(jd_text, top_k)}

@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return {"deleted": affected}
