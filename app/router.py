from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.json import JsonOutputParser


router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

router_prompt_template = """
You are an expert at routing a user's question to the best retrieval tool.
Based on the query, choose ONE tool that is best suited to answer it.

Here are the available tools:

1.  **rag_fusion**: 
    - **USE THIS FOR:** Broad, vague, or exploratory questions.
    - **KEYWORDS:** "what is", "how does", "explain", "overview", "compare"
    - **EXAMPLE:** "What is RAG-Fusion?", "Compare RAG vs. Fine-tuning", "What are diffusion models?"

2.  **hyde**:
    - **USE THIS FOR:** Vague questions where the user's query is very short or ambiguous.
    - **EXAMPLE:** "Tell me about AI", "What about transformers?"

3.  **hybrid**:
    - **USE THIS FOR:** Very specific, keyword-heavy, or technical lookups.
    - **KEYWORDS:** Acronyms, function names, specific numbers, error messages.
    - **EXAMPLE:** "What is the `chunk_size` for `ParentDocumentRetriever`?", "What is SPLADE?", "What is ResNet-50?"

4.  **base**:
    - A general-purpose retriever. Use this as a default if the query is clear and simple but doesn't fit other categories.
**User Query:**
{query}

**Instructions:**
Return a JSON object with a *single* key "tool" and the value being the *exact name* of the tool you chose.
Example: {{"tool": "hybrid"}}
"""

router_prompt = PromptTemplate(
    template=router_prompt_template,
    input_variables=[ "query"]
)

router_chain = router_prompt | router_llm | JsonOutputParser()