Resume Search & Chat (FastAPI + React + FAISS)

An end-to-end AI-powered Resume Search & Chat System that allows uploading resumes / documents (PDF/DOCX), indexing them using FAISS vector search, and performing semantic queries via a chat-like UI.

# Features:

1. Resume Upload: Upload .pdf or .docx resumes.

2. Text Cleaning: Extract text from resumes using PyPDF2/docx.

3. Vector Indexing (FAISS): Store embeddings for semantic search.

4. Semantic Search: Query resumes by skills (e.g., Python Developer).

5. Citation Snippets: Show snippets of where the query matched inside resumes.

6. Hugging Face Transformers: Trained and deployed pre-trained machine learning models.

7. Chat UI: Search resumes in a chat-like interface with query memory.

8. PostgreSQL Metadata Store: Keep track of resumes, filenames, and timestamps.

9. React Frontend: Simple UI for uploading, searching, and chatting.

10. End-to-End Fullstack: Combines FastAPI backend with React frontend.


      # Architecture
[Resume Upload (.pdf/.docx)]
        ↓
  [FastAPI Backend]
        ↓
[Extract & Clean Text]
        ↓
 [FAISS Vector Index] ←→ [SQLite Metadata DB]
        ↓
   [Semantic Search API]
        ↓
   [React Frontend Chat UI]


# Tech Stack

Backend (FastAPI)
->Python 3.10+
->FastAPI, Uvicorn
->FAISS (Facebook AI Similarity Search)
->SQLite (metadata DB)
->PyPDF2, python-docx
Frontend (React)
->React 18+
->Plain CSS (custom styling)
->Fetch API for backend calls

# Python Version
use python 3.10 version (py -3.10 --version) Python 3.10.11

# Installation & Setup

Backend (FastAPI)
cd fastapi-app
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

pip install --upgrade pip wheel setuptools (creating new virtual environment)
pip install "transformers==4.41.2" "sentence-transformers==2.7.0" faiss-cpu fastapi uvicorn

pip install -r requirements.txt

# Start FastAPI server
uvicorn app:app --reload


API will be live at:
 http://127.0.0.1:8000

# Frontend (React)
cd resume-frontend
npm install
npm start


Frontend will be live at:
 http://localhost:3000

# SQLite Commands

sqlite3 resumes.db 
.tables
.schema resumes
SELECT id, filename, uploaded_at FROM resumes LIMIT 5;

SELECT id, filename, substr(content, 1, 100) as snippet
   ...> FROM resumes
   ...> WHERE content LIKE '%python%'
   ...> LIMIT 3;

SELECT COUNT(*) FROM resumes;

Sentence Transformers calls hugging face transformers using Tensorflow + keras

# Embeddings

Embeddings using all-MiniLM-L6-v2
FAISS index and metadata

 # API Endpoints
Upload Resume
POST /upload/


# Upload .pdf or .docx

Extract → clean → store in DB + FAISS

# Semantic Search
GET /semantic_search/?q=Python Developer&top_k=5


Finds top resumes matching query

Returns rank, score, filename, snippets

# Chat Interface
POST /chat/?user_query=AI engineer&top_k=3


Adds query to conversation memory

Returns relevant resumes + citations

# Resume Retrieval
GET /resumes/{resume_id}
Fetch resume content by ID

# Author
Developed by Hemanth R

## License
This project is licensed under the **MIT License**.  
© 2025 Hemanth. All rights reserved.



