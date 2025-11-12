from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.json import JsonOutputParser

# --- 1. Define the Router LLM ---
# This is where the ChatGroq model is defined.
# It will load the GROQ_API_KEY from your .env file.
router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# --- 2. Define the Router Prompt ---
router_prompt_template = """
You are an expert at routing a user's question to the best retrieval tool.
Based on the query, choose ONE tool that is best suited to answer it.

Here are the available tools:

1.  **rag_fusion**: 
    - Good for broad, vague, or exploratory questions.
    - Use this when the user is asking a general question (e.g., "What about RAG?").
    - This tool expands the query into multiple sub-queries.

2.  **hyde**:
    - Good for ambiguous questions or when the query is short.
    - Use this when the user's query lacks detail.
    - This tool generates a hypothetical answer to find similar documents.

3.  **hybrid**:
    - Good for specific, technical, or keyword-heavy questions.
    - Use this for finding exact terms, function names, or acronyms (e.g., "What is BM25?").
    - This tool combines keyword search with vector search and re-ranks the results.

4.  **base**:
    - A general-purpose retriever.
    - Use this as a default for any clear, straightforward question that doesn't fit the other categories.

**User Query:**
{query}

**Instructions:**
Return a JSON object with a *single* key "tool" and the value being the *exact name* of the tool you chose.
Example: {{"tool": "hybrid"}}
"""

router_prompt = PromptTemplate(
    template=router_prompt_template,
    input_variables=[ "query"]# This only has one variable
)

# --- 3. Define the Router Chain ---
router_chain = router_prompt | router_llm | JsonOutputParser()