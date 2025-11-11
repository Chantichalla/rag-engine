from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.retrievers import load_retrieval_documents
from app.router import router_chain

retrievers = load_retrieval_documents()

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
{question}

ANSWER:
"""

rag_prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context","question"]
)

def decisiom_maker(input_dict):
    # This takes output from router and brings the correct answer
    tool_name = input_dict.get("routing_decision", {}).get("tool", "base")
    query = input_dict.get("query", "")

    print(f"--- Routing to: {tool_name} ---") # For debugging

    if tool_name == "hybrid":
        return retrievers["hybrid"].invoke(query)
    elif tool_name == "hyde":
        return retrievers["hyde"].invoke(query)
    elif tool_name == "rag_fusion":
        return retrievers["rag_fusion"].invoke(query)
    else:
        return retrievers["base"].invoke(query)
    
# --- 4. Build the Full End-to-End Chain ---
# Step 1: Create a "passthrough" for the query.
# This chain will take a query (string) and output a dictionary:
# {"query": "the user's query"}
chain_with_query = RunnablePassthrough.assign(
    query=lambda x: x["query"]
)
# Step 2: Add the routing decision.
# This chain will take the dict from Step 1 and add the router's choice:
# {"query": "...", "routing_decision": {"tool": "hybrid"}}
chain_with_routing_decision = chain_with_query | {
    "routing_decision": (lambda x: router_chain.invoke({"query": x["query"]})),
    "query": (lambda x: x["query"])
}

# Step 3: Run the chosen retriever.
# This chain will take the dict from Step 2, run our custom `route_to_retriever`
# function, and format the output.

final_rag_chain = chain_with_routing_decision | {
    "context": (lambda x: decisiom_maker(x)) | (lambda docs: "\n---\n".join([d.page_content for d in docs])),
    "query": (lambda x:x["query"])
    }| rag_prompt | final_llm | StrOutputParser()

# `full_rag_chain` is the single, runnable object our API will call.