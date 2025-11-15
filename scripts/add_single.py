#!/usr/bin/env python3
import sys
import os
import json
import hashlib
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
DB_DIR = Path(os.getenv("DB_DIR", str(ROOT / "vectorstore")))
DOCSTORE_DIR = Path(os.getenv("DOCSTORE_DIR", str(ROOT / "docstore")))
CHECKPOINT_FILE = Path(os.getenv("CHECKPOINT_FILE", str(ROOT / "ingest_checkpoint.json")))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("EMBED_DEVICE", "cuda")

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 40

def sha256_hex_key(k) -> str:
    if isinstance(k, str):
        raw = k.encode("utf-8")
    else:
        raw = json.dumps(k, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_checkpoint(cp: dict):
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))

def add_single(paths):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()

    # initialize embeddings & stores
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    byte_store = LocalFileStore(root_path=str(DOCSTORE_DIR))
    store = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda k: sha256_hex_key(k),
        value_serializer=lambda v: pickle.dumps(v),
        value_deserializer=lambda b: pickle.loads(b),
    )

    vectorstore = Chroma(
        collection_name="full-documents",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

    retriever = ParentDocumentRetriever(
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        vectorstore=vectorstore,
        docstore=store,
    )

    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            print(f"Skipping (not found): {p}")
            continue
        h = file_hash(p)
        if checkpoint.get(str(p)) == h:
            print(f"Already processed: {p}")
            continue

        print(f"Processing single file: {p}")
        loader = DirectoryLoader(str(p.parent), glob=p.name, loader_cls=PyPDFLoader, show_progress=False)
        docs = loader.load()
        if not docs:
            print("No docs found by loader for", p)
            checkpoint[str(p)] = h
            save_checkpoint(checkpoint)
            continue

        retriever.add_documents(docs)

        # persist vectorstore if API exists (best-effort)
        try:
            if hasattr(vectorstore, "persist") and callable(vectorstore.persist):
                vectorstore.persist()
        except Exception:
            pass

        checkpoint[str(p)] = h
        save_checkpoint(checkpoint)
        print("Added and checkpointed:", p)

def main():
    if len(sys.argv) < 2:
        print("Usage (CMD): python scripts\\add_single.py data\\newfile.pdf")
        sys.exit(1)
    add_single(sys.argv[1:])

if __name__ == "__main__":
    main()
