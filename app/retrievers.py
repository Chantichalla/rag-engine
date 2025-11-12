import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_classic.storage import LocalFileStore
from langchain_classic.storage import InMemoryStore
#import pickle
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Advance imports
#--hyDE imports
# ... (all your existing imports) ...
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_classic.chains import LLMChain

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
 #-- RAG_Fusion imports
 # ... (all your existing imports) ...
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever
)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
# --- Load Environment Variables ---
load_dotenv()

# --- 1. Define Constants ---
DB_PATH = "vectorstore/"
DATA_PATH = "data/"
EMBED_MODEL_NAME = "BAAI/bge-m3"

# --- 2. Function to Load Core Components (THE FIX) ---
def get_base_retriever():
    print("--Loading the existing embeddings---")

    model_kwargs = {"device":"cuda"}
    encode_kwargs = {"normalize_embeddings":True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs = encode_kwargs
    )
    
    # 1. Load the persistent vector store from disk
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # 2. Create a NEW, EMPTY docstore for our server
    store = InMemoryStore()
    
    # 3. --- THIS IS THE FIX ---
    #    Load the raw PDFs from /data (just like ingest.py)
    print(f"Loading documents from {DATA_PATH} to populate docstore...")
    loader = DirectoryLoader(
        DATA_PATH, # <-- This was the bug, it's now fixed
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

    # 4. Initialize the retriever
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, # Use the one from disk
        docstore=store,         # Use the new empty one
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    # 5. --- THIS IS THE FIX ---
    #    Repopulate the in-memory docstore with the parent docs.
    if docs:
        print("Populating in-memory docstore...")
        base_retriever.add_documents(docs, add_to_vectorstore=False) # Don't re-add to vector store
    else:
        print("Warning: No documents found in /data, docstore is empty.")

    print("Base retriever loaded.")
    return base_retriever, vectorstore

# --- 3. Define Expert Retrievers ---

def get_hybrid_retriever(vectorstore): 
    print("Initializing hybrid retriever...")
    
    print(f"Loading documents from {DATA_PATH} for BM25...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    all_docs = loader.load()

    if not all_docs:
        print("Warning: No documents found in /data. Skipping BM25.")
        return None 

    print(f"Initializing BM25Retriever with {len(all_docs)} documents...")
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        print("Warning: COHERE_API_KEY not found. Skipping re-ranker.")
        return ensemble_retriever 

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-english-v3.0",
        top_n=5
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    
    print("✅ Hybrid + Re-ranker retriever initialized.")
    return compression_retriever

def get_hyde_retriever(base_embeddings):
    print("Initializing HyDE retriever...")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    hyde_prompt_template = """
    Please write a short, concise document that answers the following question.
    This document will be used to find the most relevant real documents.
    Question: {question}
    Document:
    """
    hyde_prompt = PromptTemplate(
        template=hyde_prompt_template,
        input_variables=["question"]
    )
    
    llm_chain = LLMChain(llm=llm, prompt=hyde_prompt)

    hyde_embedder = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, 
        base_embeddings=base_embeddings
    )
    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=hyde_embedder,
    )
    
    hyde_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ HyDE retriever initialized.")
    return hyde_retriever

def get_rag_fusion_retriever(vectorstore):
    print("Initializing RAG-Fusion (MultiQuery) retriever...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    fusion_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )
    
    print("---fusion retriever initialized---")
    return fusion_retriever

# --- 4. Main Factory Function ---
def load_retrieval_components():
    base_retriever, vectorstore = get_base_retriever()

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    base_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    hybrid_retriever = get_hybrid_retriever(vectorstore)
    hyde_retriever = get_hyde_retriever(base_embeddings)
    fusion_retriever = get_rag_fusion_retriever(vectorstore)
    
    print("--- All retrieval components loaded ---")
    
    return {
        "base": base_retriever,
        "hybrid": hybrid_retriever,
        "hyde": hyde_retriever,
        "rag_fusion": fusion_retriever
    }