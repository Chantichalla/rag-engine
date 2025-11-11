import os
from dotenv import load_dotenv

from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Advance imports
#--hyDE imports
# ... (all your existing imports) ...
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
 #-- RAG_Fusion imports
 # ... (all your existing imports) ...
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever
)

from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CohereRerank

load_dotenv()

DB_PATH = "vectorstore/"
EMBED_MODEL_NAME = "BAAI/bge-m3"

def get_base_retriever():

    print("--Loading the existing embeddings---")

    # embed model
    model_kwargs = {"device":"cuda"}
    encode_kwargs = {"normalize_embeddings":True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs = encode_kwargs
    )
    # Vector database
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    store = InMemoryStore()
    
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore.as_retriever(search_kwargs={"k":5}),
        docstore=store,
    )

    print("Base retriever loaded.")
    return base_retriever, vectorstore

# Now we need to build the function that takes this base rtv
def get_hybrid_retrievers (base_retriever, vectorstore):
    
    # Let's get all documents from the vector store.
    all_documents = vectorstore.get().get("documents",[])
    if not all_documents:
        print("Warning: No documents found in vector store for BM25. Skipping hybrid search.")
        return base_retriever # Fall back to the base retriever
    from langchain_core.documents import Document

    bm25_docs = [Document(page_content=doc) for doc in all_documents ]

    print(f"Initializing BM25Retriever with {len(bm25_docs)} documents...")
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[base_retriever,bm25_retriever],
        weights=[0.5,0.5],
    )
    # --- Step 4: Initialize Re-ranker ---
    # We need to get our API key
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        print("Warning: COHERE_API_KEY not found. Skipping re-ranker.")
        return ensemble_retriever # Fall back to just the hybrid search
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        top_n=5
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    print(" Hybrid + Re-ranker retriever initialized.")
    return compression_retriever

def load_retrieval_documents():

    base_retriever, vectorstore = get_base_retriever()

    # We need the standalone BGE embedding model for HyDE
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    base_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    hybrid_retriever = get_hybrid_retrievers(base_retriever,vectorstore)
    hyde_retriever = hyde_retriever_fun(base_embeddings)
    fusiion_retriever = get_fusion(vectorstore)

    return {
        "base":base_retriever,
        "hybrid":hybrid_retriever,
        "hyde":hyde_retriever,
        "rag-fusion":fusiion_retriever
    }

# ---hyDE---

def hyde_retriever_fun(base_embeddings):
    print("Initializing HyDE retriever...")

    llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)
    hyde_embedder = HypotheticalDocumentEmbedder(
        llm=llm,
        base_embeddings=base_embeddings,
        prompt_key = "web_search"
    )
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=hyde_embedder,
    )
    hyde_retriever = vectorstore.as_retriever(search_kwargs = {"k":5})

    return hyde_retriever

# --- RAG-Fusion
def get_fusion(vectorstore):
    # creating a multi query method

    llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)

    # intialization of multiquery retriever
    fusion_retriever = MultiQueryRetriever(
        retriever=vectorstore.as_retriever(search_kwargs={"k":5}),
        llm=llm,
    )
    print("---fusion retriever initialized---")
    return fusion_retriever



