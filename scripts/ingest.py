import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import EncoderBackedStore
from langchain_classic.storage import LocalFileStore
import pickle
import json
from langchain_classic.storage import InMemoryStore
#from langchain_classic.storage import LocalFileStore
#import pickle
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

#-- Load --
load_dotenv()

data_path = 'data/'
DB_path = "vectorstore/"
EMBED_MODEL_NAME = "BAAI/bge-m3" 
doc_store_path = 'docstore/'

def main():
    print("--- Starting Ingestion")
# --- Step 1: Load Documents ---
    print(f"Loading documents from {data_path}...")
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    if not documents:
        print("No documents found. Please add a PDF (like the RAG wiki page) to the 'data/' folder.")
        return
    print(f"loaded {len(documents)} documents")
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} 

    print(f"Initializing embedding model: {EMBED_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("embedding model initialized")

    #parent document
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap = 200)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

    # 1. Create the "dumb" byte store that saves to disk
    byte_store = LocalFileStore(root_path=doc_store_path)
    
    # 2. Create the "smart" store that wraps it, using pickle to encode
    store = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda k: json.dumps(k),
        value_serializer=lambda v: pickle.dumps(v),
        value_deserializer=lambda b: pickle.loads(b),
    )
    vectorstore = Chroma(
        collection_name="full-documents",
        embedding_function=embeddings,
        persist_directory=DB_path
    )

    print("--- initializing Parent Docucment ---")

    #this retriever connnects everything

    retriever = ParentDocumentRetriever(
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        vectorstore=vectorstore,
        docstore=store,
    )

    # We feed the retriever in small batches to avoid the ChromaDB error
    batch_size = 100
    total_docs = len(documents)
    
    print(f"Adding {total_docs} documents in batches of {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        print(f"--- Processing batch {i//batch_size + 1} / {total_docs//batch_size + 1} ---")
        retriever.add_documents(batch_docs)
    
    
    print("-----------------------------------------")
    print(" Ingestion complete!")
    print(f"Vector store created at: {DB_path}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

