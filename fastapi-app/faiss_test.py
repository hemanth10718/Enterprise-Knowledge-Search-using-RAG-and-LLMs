import faiss
import numpy as np
import pickle
import os

INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.pkl"

# --- Load or initialize FAISS index ---
def load_index(dimension=5):
    if os.path.exists(INDEX_FILE):
        print("🔹 Loading existing FAISS index...")
        index = faiss.read_index(INDEX_FILE)
    else:
        print("🔹 Creating new FAISS index...")
        index = faiss.IndexFlatL2(dimension)
    return index

# --- Load or initialize metadata ---
def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "rb") as f:
            return pickle.load(f)
    return []

# --- Save index and metadata ---
def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

# --- Add embeddings ---
def add_resumes(embeddings, meta_info):
    index = load_index(dimension=embeddings.shape[1])
    meta = load_meta()

    # Add new embeddings
    index.add(embeddings)
    meta.extend(meta_info)

    # Save
    save_index(index, meta)
    print("✅ Added to FAISS:", len(meta_info), "resumes")

# --- Search resumes ---
def search_resume(query_vector, k=3):
    index = load_index(dimension=query_vector.shape[1])
    meta = load_meta()

    distances, indices = index.search(query_vector, k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):  # Valid index
            results.append({
                "resume": meta[idx],
                "distance": float(dist)
            })
    return results
