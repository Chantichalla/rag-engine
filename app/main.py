from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
# This is the "master chain" we built in Phase 5
from app.chains import final_rag_chain

# Initialize our FastAPI app
app = FastAPI(
    title="Ultimate RAG Project API",
    description="An advanced, multi-retriever RAG system."
)

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the server is running."""
    return {"message": "Welcome to the Ultimate RAG API!"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    This is the main chat endpoint.
    It receives a query and returns a RAG-generated answer.
    """
    print(f"--- Received query: {request.query} ---")
    
    # 1. We call our "master chain"
    # We use 'ainvoke' for asynchronous execution, which FastAPI loves
    response = await final_rag_chain.ainvoke({"query": request.query})
    
    # 2. We return the response in the correct Pydantic shape
    return ChatResponse(answer=response)