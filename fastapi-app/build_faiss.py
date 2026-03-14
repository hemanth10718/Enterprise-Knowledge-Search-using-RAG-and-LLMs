# build_faiss.py
import sqlite3
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "resumes.db")
INDEX_PATH = os.path.join(BASE, "faiss.index")
META_PATH = os.path.join(BASE, "faiss_meta.pkl")

# 1) load model (will download on first run)
MODEL_NAME = "all-MiniLM-L6-v2"   # compact & good
model = SentenceTransformer(MODEL_NAME)
dim = model.get_sentence_embedding_dimension()
print("Embedding dim:", dim)

def read_resumes():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename, content FROM resumes")
    rows = cur.fetchall()
    conn.close()
    return rows

def build_index(rows):
    if len(rows) == 0:
        print("No resumes found in DB.")
        return

    # We'll embed each entire resume text (simple). For better results you can chunk later.
    texts = [r[2] for r in rows]
    print("Encoding", len(texts), "documents (this may take a moment)...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize vectors to unit length (so we can use inner product = cosine sim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    embeddings = embeddings.astype("float32")

    # Build FAISS index (IndexFlatIP for cosine with normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("Added vectors to FAISS. ntotal =", index.ntotal)

    # Build metadata list aligned with vector order
    meta = []
    for i, r in enumerate(rows):
        rid, fname, content = r
        snippet = (content[:400] + "...") if len(content) > 400 else content
        meta.append({"resume_id": rid, "filename": fname, "snippet": snippet})

    # Save index and meta
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print("Saved:", INDEX_PATH)
    print("Saved:", META_PATH)

if __name__ == "__main__":
    rows = read_resumes()
    build_index(rows)
