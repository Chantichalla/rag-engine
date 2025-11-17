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

