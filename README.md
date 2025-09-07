Resume Search & Chat (FastAPI + React + FAISS)

An end-to-end AI-powered Resume Search & Chat System that allows uploading resumes (PDF/DOCX), indexing them using FAISS vector search, and performing semantic queries via a chat-like UI.

# Features:

1. Resume Upload: Upload .pdf or .docx resumes.

2.Text Cleaning: Extract text from resumes using PyPDF2/docx.

3. Vector Indexing (FAISS): Store embeddings for semantic search.

4. Semantic Search: Query resumes by skills (e.g., Python Developer).

5. Citation Snippets: Show snippets of where the query matched inside resumes.

6. Chat UI: Search resumes in a chat-like interface with query memory.

7. SQLite Metadata Store: Keep track of resumes, filenames, and timestamps.

8. React Frontend: Simple UI for uploading, searching, and chatting.

9. End-to-End Fullstack: Combines FastAPI backend with React frontend.


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

# Installation & Setup

Backend (FastAPI)
cd fastapi-app
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

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
Developed by Hemanth R, AI/ML engineer

## License
This project is licensed under the **MIT License**.  
© 2025 Hemanth. All rights reserved.



