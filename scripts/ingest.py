import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
# THIS IS THE CORRECT PATH FOR THIS VERSION:
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

#-- Load --
load_dotenv()

data_path = 'data/'
DB_path = "vectorstore/"
EMBED_MODEL_NAME = "BAAI/bge-m3" 

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

    store = InMemoryStore()

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

    retriever.add_documents(documents)
    print("-----------------------------------------")
    print("✅ Ingestion complete!")
    print(f"Vector store created at: {DB_path}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()




