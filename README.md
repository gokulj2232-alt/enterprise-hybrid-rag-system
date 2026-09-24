# AI-Powered Enterprise Hybrid RAG System

An AI-powered enterprise document question-answering system that combines **semantic search, keyword search, hybrid retrieval, cross-encoder reranking, context compression, evidence extraction, and local LLM generation** to provide grounded answers from a private knowledge base.

The system also supports **user-uploaded documents** through an isolated document index, allowing questions to be answered from newly uploaded TXT, CSV, Excel, or PDF files without modifying the main enterprise knowledge base.

---

## 🚀 Project Overview

Traditional keyword search can miss documents when the wording of the query differs from the wording in the document.

This project combines multiple retrieval techniques:

```text
User Question
      ↓
Query Processing
      ↓
Multi-Query Generation
      ↓
┌─────────────────────────────┐
│ Semantic Search             │
│ FAISS + Sentence Embeddings │
└─────────────────────────────┘
              +
┌─────────────────────────────┐
│ Keyword Search              │
│ BM25                        │
└─────────────────────────────┘
              ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Relevance Filtering
      ↓
Context Compression
      ↓
Evidence Extraction
      ↓
Answerability Validation
      ↓
Qwen LLM
      ↓
Grounding Validation
      ↓
Final Answer + Sources
```

The system is designed to answer questions using the available documents and avoid unsupported answers when relevant evidence cannot be found.

---

# ✨ Key Features

* Enterprise-scale document retrieval
* Semantic vector search using FAISS
* Keyword retrieval using BM25
* Hybrid retrieval
* Multi-query search
* Cross-encoder reranking
* Relevance filtering
* Context compression
* Evidence extraction
* Answerability validation
* Grounding validation
* Local LLM answer generation
* Conversation memory
* RAG evaluation components
* FastAPI REST API
* Streamlit web interface
* TXT, CSV, Excel and PDF document upload
* Isolated uploaded-document vector index
* Docker deployment
* Unknown-question fallback
* Source-aware answers

---

# 📊 Knowledge Base

The primary knowledge base is based on an upgraded 50,000-document dataset.

### Dataset

```text
NLP_50K_Upgraded.xlsx
```

### Original dataset fields

* `document_id`
* `category`
* `title`
* `content`
* `keyword`

### Processed knowledge base

After preprocessing and duplicate removal, the system contains approximately:

```text
49,321 documents
77,658 chunks
77,658 FAISS vectors
384-dimensional embeddings
```

The main FAISS index and chunk database were verified to contain the same number of records.

---

# 🧠 Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

Each document chunk is converted into a:

```text
384-dimensional vector
```

These vectors are stored in a FAISS index for efficient semantic retrieval.

---

# 🔎 Hybrid Retrieval

The retrieval system combines two complementary approaches.

## 1. Semantic Search

FAISS searches for documents that are semantically similar to the user's question.

This helps retrieve relevant documents even when the query and document use different wording.

## 2. Keyword Search

BM25 identifies documents containing important query terms.

This provides strong lexical matching.

## 3. Hybrid Search

The results from semantic and keyword retrieval are combined into a hybrid ranking.

```text
Semantic Retrieval
        +
Keyword Retrieval
        ↓
Hybrid Score
        ↓
Candidate Documents
```

---

# 🎯 Cross-Encoder Reranking

Retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates the relationship between:

```text
Question ↔ Retrieved Document
```

and improves the ordering of candidate results before generation.

---

# 🧹 Context Processing

Before sending information to the language model, the pipeline performs several processing stages:

1. Relevance filtering
2. Context compression
3. Evidence extraction
4. Answerability validation

This reduces irrelevant information and helps keep generated answers grounded in retrieved evidence.

---

# 🤖 Large Language Model

The system uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The LLM generates answers using the retrieved document context.

The generation pipeline is designed around document-grounded question answering rather than unrestricted general knowledge generation.

If sufficient supporting information is not available, the system can return:

```text
I don't know based on the provided documents.
```

---

# 📄 Uploaded Document RAG

The application supports uploading new documents through the Streamlit interface.

Supported formats:

* TXT
* CSV
* XLSX
* PDF

Uploaded documents are processed separately from the main enterprise knowledge base.

```text
Uploaded File
     ↓
Document Processing
     ↓
Text Cleaning
     ↓
Chunking
     ↓
Embedding
     ↓
Separate FAISS Index
     +
Separate BM25 Data
     ↓
Uploaded-Document RAG
```

### Important architecture decision

Uploaded documents **do not modify the main enterprise FAISS index**.

The uploaded-document system uses:

```text
data/uploads/index/
```

This keeps the original enterprise knowledge base isolated and protects the main production index.

---

# 🌐 FastAPI

The backend API is implemented using FastAPI.

### API

```text
http://localhost:8000
```

### Health Check

```http
GET /health
```

### Question Endpoint

```http
POST /ask
```

Example request:

```json
{
  "question": "What is retrieval augmented generation?",
  "top_k": 5,
  "conversation": [],
  "uploaded_dataset": false
}
```

Example response structure:

```json
{
  "question": "What is retrieval augmented generation?",
  "answer": "...",
  "sources": [],
  "processing_time_seconds": 0,
  "conversation_messages_received": 0,
  "uploaded_dataset": false
}
```

---

# 🖥️ Streamlit Application

The user interface is built with Streamlit.

```text
http://localhost:8501
```

The application provides:

* Question answering
* Document upload
* Uploaded-document mode
* Source-aware responses
* Knowledge-base querying
* Unknown-answer fallback

---

# 🐳 Docker Deployment

The complete application can be run using Docker Compose.

### Services

```text
enterprise-rag-api
enterprise-rag-streamlit
```

### Architecture

```text
Browser
   │
   ▼
Streamlit
:8501
   │
   ▼
FastAPI
:8000
   │
   ▼
Enterprise Hybrid RAG Pipeline
   │
   ├── FAISS
   ├── BM25
   ├── Reranker
   ├── Context Processing
   └── Qwen LLM
```

---

# ▶️ Running the Project

## 1. Start Docker Desktop

Make sure Docker Desktop is running.

## 2. Open PowerShell

Navigate to the project:

```powershell
cd D:\semantic_search_project
```

## 3. Start the services

```powershell
docker compose up -d
```

## 4. Check containers

```powershell
docker ps
```

You should see:

```text
enterprise-rag-api
enterprise-rag-streamlit
```

## 5. Open Streamlit

Open:

```text
http://localhost:8501
```

## 6. Check API

```text
http://localhost:8000
```

---

# 🧪 Testing the API

Example PowerShell request:

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/ask `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"question":"What is retrieval augmented generation?"}'
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

# 🧪 Validation Performed

The system has been tested with:

### Enterprise knowledge base

Question:

```text
What is retrieval augmented generation?
```

The system successfully generated a document-grounded answer.

### Uploaded document

Question:

```text
What is the main purpose of this document?
```

The system successfully answered using the uploaded document.

### Unsupported question

Question:

```text
What is the capital of France?
```

The system returned:

```text
I don't know based on the provided documents.
```

This confirms the document-grounded fallback behavior.

### Main FAISS integrity

```text
Chunks:       77,658
FAISS vectors:77,658
Dimension:    384
```

The main chunk data and FAISS index are aligned.

---

# 📁 Project Structure

```text
semantic_search_project/
│
├── data/
│   ├── archive/
│   ├── uploads/
│   │   └── index/
│   ├── chunks.jsonl
│   ├── chunk_metadata.jsonl
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   ├── NLP_50K_Preprocessed.xlsx
│   └── NLP_50K_Upgraded.xlsx
│
├── legacy_backup/
│
├── src/
│   ├── api.py
│   ├── app.py
│   ├── answerability.py
│   ├── evidence_extractor.py
│   ├── grounding_validator.py
│   ├── llm.py
│   │
│   ├── chunking/
│   ├── dataset_generator/
│   ├── document_processing/
│   ├── embeddings/
│   ├── evaluation/
│   ├── memory/
│   ├── preprocessing/
│   ├── rag/
│   ├── search/
│   ├── uploads/
│   └── vector_store/
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── README.md
├── requirements-docker.txt
└── requirements.txt
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* OpenPyXL

### NLP / Embeddings

* Sentence Transformers
* `all-MiniLM-L6-v2`

### Retrieval

* FAISS
* BM25
* Rank-BM25

### Reranking

* Cross Encoder
* `cross-encoder/ms-marco-MiniLM-L-6-v2`

### LLM

* Qwen
* `Qwen/Qwen2.5-0.5B-Instruct`

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

### Deployment

* Docker
* Docker Compose
* WSL 2

---

# 🎯 Project Objective

The objective of this project is to build a practical enterprise-style RAG system capable of retrieving relevant information from a large private document collection and generating grounded answers using a local language model.

The architecture demonstrates a complete retrieval-augmented generation workflow rather than a simple chatbot implementation.

---

# 🔮 Future Improvements

Possible future enhancements include:

* Larger instruction-tuned LLM
* GPU acceleration
* Advanced query routing
* Better evaluation datasets
* Authentication and authorization
* Document management dashboard
* User-specific knowledge bases
* Streaming responses
* Monitoring and logging
* Production cloud deployment
* Automated document ingestion pipelines

---

# 📌 Project Status

**Core development: Complete**

The system has been implemented, tested, containerized, and validated with both the primary enterprise knowledge base and uploaded documents.

The project is now in the final documentation and release-preparation stage.
