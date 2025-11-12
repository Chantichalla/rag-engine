from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# It imports the retrievers and the *finished* router chain
from app.retrievers import load_retrieval_components
from app.router import router_chain

# --- 1. Load all our retriever tools ---
print("Loading retrieval components...")
retrievers = load_retrieval_components()
print("All components loaded.")

# --- 2. Define the Final RAG LLM & Prompt ---
# This is the ONLY ChatGroq in this file
final_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

rag_prompt_template = """
You are an expert Q&A assistant.
Use the following context to answer the user's question.
If you don't know the answer, just say "I am sorry, I do not have information on that topic."
Do not make up an answer. Be concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

rag_prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "query"]
)

# --- 3. Build the Routing Logic ---
def route_to_retriever(input_dict):
    tool_name = input_dict.get("routing_decision", {}).get("tool", "base")
    query = input_dict.get("query", "")
    
    print(f"--- Routing to: {tool_name} ---")
    
    if tool_name == "hybrid" and retrievers["hybrid"]:
        return retrievers["hybrid"].invoke(query)
    elif tool_name == "hyde":
        return retrievers["hyde"].invoke(query)
    elif tool_name == "rag_fusion":
        return retrievers["rag_fusion"].invoke(query)
    else:
        return retrievers["base"].invoke(query)

# --- 4. Build the Full End-to-End Chain ---
# This chain correctly pipes to the router_chain
chain_with_routing_decision = RunnablePassthrough.assign(
    routing_decision=router_chain,
    query=lambda x: x["query"]
)

context_retrieval_chain = RunnableLambda(route_to_retriever) | (lambda docs: "\n---\n".join([d.page_content for d in docs]))

final_rag_chain = chain_with_routing_decision | {
    "context": context_retrieval_chain,
    "query": (lambda x: x["query"])
} | rag_prompt | final_llm | StrOutputParser()