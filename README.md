# Hybrid RAG Engine (Embeddings + BM25 + HyDE + Fusion)

A fully functional, end-to-end Retrieval-Augmented Generation (RAG) system combining dense embeddings, BM25 lexical retrieval, HyDE synthetic expansion, and multi-query fusion for highly accurate document-grounded responses.

This project implements:
- A complete ingestion pipeline (batch + single PDF)
- Chunking and embedding generation
- Hybrid retrieval (Embeddings + BM25 + HyDE + Fusion)
- A FastAPI backend with production-style startup lifecycle
- GPU-accelerated inference (optional)
- Ready-to-use `/chat` endpoint for answering user queries

---

## Features

### 🔹 Retrieval System
- **Dense Embeddings** using SentenceTransformers
- **BM25 Lexical Retrieval**
- **HyDE (Hypothetical Document Embeddings)** synthetic query expansion
- **MultiQuery Fusion** for multi-perspective retrieval
- **Vectorstore + Docstore** architecture

### 🔹 API Layer (FastAPI)
- `/chat` – query endpoint returning RAG-generated answers  
- `/health` – liveness check  
- `/ready` – readiness indicator  
- Clean startup lifecycle with heavy loading handled asynchronously

### 🔹 Ingestion Tools
- **Batch ingestion** for multiple PDFs  
- **Single-file ingestion** (for quick testing)  
- Automatic chunking, embedding, and vectorstore updates

---

## Project Structure
```bash
ultimate-rag-project/
│
├── app/
│ ├── main.py # FastAPI application
│ ├── chains.py # Defines final_rag_chain
│ ├── retrievers.py # Hybrid/BM25/HyDE logic
│ └── schemas.py # API request/response models
│
├── data/ # Your source PDFs
├── docstore/ # Document store
├── vectorstore/ # Embedding index
│
├── scripts/
│ ├── ingest.py # Batch ingestion
│ └── add_single.py # Single PDF ingestion
│
├── requirements.txt
└── README.md
```

---

## Ingest a Single PDF

Place your PDF inside the `data/` directory:

data/
my_file.pdf

bash
Copy code

Run:

```bash
python scripts/add_single.py data/my_file.pdf
```

This will:

Parse the PDF

Chunk it

Generate embeddings

Update the vectorstore/docstore

Run the API :
```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open Swagger UI:

http://localhost:8000/docs


Test:

POST /chat
{
  "query": "Summarize the document"
}

## Tech Stack

Python 3.10+

FastAPI

LangChain

SentenceTransformers

BM25 (rank-bm25)

Uvicorn

Optional: CUDA + PyTorch for GPU acceleration

## What I Learned

Retrieval engineering beyond basic vector search

Hybrid RAG design and multi-step retrieval fusion

Handling asynchronous & synchronous chain invocation

API lifecycle management

Clean modular RAG architecture

## Future Enhancements

Optional multimodal ingestion

Webpage ingestion pipeline

Streaming responses

Ranking/ensemble experiments

Contact

If you're working on RAG systems, retrieval engineering, or production ML, feel free to reach out.
I will be happy to help!
