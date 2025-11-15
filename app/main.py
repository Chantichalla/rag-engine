from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.chains import final_rag_chain


app = FastAPI(
    title="Ultimate RAG Project API",
    description="An advanced, multi-retriever RAG system."
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ultimate RAG API!"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    It receives a query and returns a RAG-generated answer.
    """
    print(f"--- Received query: {request.query} ---")
    
    #we use ainvoke which is asnychronous invoke
    response = await final_rag_chain.ainvoke({"query": request.query})
    
    return ChatResponse(answer=response)