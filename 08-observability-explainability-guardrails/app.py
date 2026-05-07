import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_groq import ChatGroq
# from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import logfire
from langsmith import traceable
from langsmith import Client as LangSmithClient 

########## Environment Variable Configuration ##########

# Activing the parallelism tokenization
os.environ['TOKENIZERS_PARALLELISM'] = 'True'
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logfire.critical("GROQ_API_KEY was not defined in the environment or .env!") # Using logfire

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langchain_api_key_env = os.getenv("LANGCHAIN_API_KEY")
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")

try:
    logfire.configure() 
    print("Log - Logfire setted.") 
except Exception as e:
     print(f"Log - Warning: Failed to configure Logfire automatically.: {e}")



if not langsmith_api_key or not langchain_api_key_env:
    print("LANGSMITH_API_KEY and/or LANGCHAIN_API_KEY aren't defineds. LangSmith tracing may not working correctly.")

if not LOGFIRE_API_KEY:
    print("LOGFIRE_API_KEY not defined. Logs to Pydantic LogFire Cloud will not work (unless another OTEL exporter is configured).")

# Checking LANGCHAIN_TRACING_V2
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "true":
    print("The LANGCHAIN_TRACING_V2 environment variable is not set to 'true'. Automatic tracing from LangGraph to LangSmith MAY be disabled.")

# Checking LANGCHAIN_API_KEY specifically for tracing
if not langchain_api_key_env:
     print("The LANGCHAIN_API_KEY environment variable is not set. LangGraph tracing for LangSmith will NOT work.")

# RAG path
VECTORSTORE_PATH = "faiss_index"

########## Auxiliary Functions ##########
@st.cache_resource
def load_llm_final_answer():
    print("LOG - Loading LLM Groq...")
    
    try:
        llm = ChatGroq(api_key = groq_api_key, model = "openai/gpt-oss-120b", temperature = 0.1) 
        
        logfire.info("LLM Groq (resposta final) carregado com sucesso.")
        
        return llm
    
    except Exception as e:
        logfire.error("Error Loading final LLM", error = str(e), exc_info = True)
        

@st.cache_resource
def load_retriever():
    print("LOG - Loading Retriever RAG...")
    
    if not os.path.exists(VECTORSTORE_PATH):
        logfire.error("FAISS index was not found", path = VECTORSTORE_PATH)
        
        print(f"FAISS was not found at '{VECTORSTORE_PATH}'. Run 'setup_rag.py'.")
    try:
        model_name = "BAAI/bge-base-en"
        encode_kwargs = {'normalize_embeddings': True} 
        embedding_model = HuggingFaceEmbeddings(
            model_name = model_name, 
            model_kwargs={'device': 'cpu'},
            encode_kwargs = encode_kwargs
        )
        
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization = True)
        retriever = vector_store.as_retriever(search_kwargs = {'k': 5})
        
        logfire.info("Retriever RAG loaded sucessfully.", path = VECTORSTORE_PATH)
        
        return retriever
    
    except Exception as e:
        logfire.error("Error loading Retriever RAG", path = VECTORSTORE_PATH, error = str(e), exc_info = True) 
        print(f"Erro loading Retriever RAG: {e}")


########## Functions for Graph Nodes in LangGraph ##########
class GraphState(TypedDict):
    query: str
    source_decision: Literal["RAG", "WEB", ""]
    rag_context: str | None
    web_results: str | None
    final_answer: str | None


@traceable(run_type = "llm", name = "Node_RouteQuery") # LangSmith Decorator
def route_query_node(state: GraphState) -> dict:
    """
    Analyst searchs and decides the source (RAG or WEB).
    Update 'source_decision' in state.
    """

    # Extracts the query from state
    query = state["query"]

    # Logfire span to grouping nodes logs
    span = logfire.span("Running node: Query routing", query = query)
    
    # Inside the Span
    with span:

        # **REFINE PROMPT WITH EXAMPLES (FEW-SHOT)**
        prompt = f"""Your task is to classify a user's query to direct it to the best source of information. The sources are:
            1. **RAG**: Internal knowledge base with technical support documents, specific procedures, our system settings, internal guides. Use RAG for questions about 'how to do X in our system', 'what is the configuration for Y', 'internal documentation on Z'.
            2. **WEB**: General internet search for information about third-party software (e.g., Anaconda, Python, Excel), technology news, generic errors not documented internally, very recent information, or anything that is not specific to our internal documents.

            Examples:
            - Query: "How do I configure the internal email server?" -> Answer: RAG
            - Query: "What is the latest version of Streamlit?" -> Answer: WEB
            - Query: "What is the procedure to reset the ABC system password?" -> Answer: RAG
            - Query: "How do I install the Python interpreter on Windows 11?" -> Answer: WEB
            - Query: "how to install Anaconda Python" -> Response: WEB

            Now, classify the following query:
            User Query: '{query}'
            Based on the query, which is the most appropriate source? Answer ONLY with the word 'RAG' or the word 'WEB'."""

        
        try:

            # **CREATING A DEDICATED LLM FOR ROUTING**
            router_llm = ChatGroq(api_key = groq_api_key,
                                  model = "openai/gpt-oss-120b",
                                  temperature = 0.0)

            response = router_llm.invoke(prompt)
            raw_decision = response.content

            decision = raw_decision.strip().upper().replace("'", "").replace('"', '')

            if decision == "RAG":
                final_decision = "RAG"
            elif decision == "WEB":
                 final_decision = "WEB"
            else:
                logfire.warn("Unexpected decision by the router, using the web as a fallback.", raw_decision = raw_decision, query = query, decision_parsed = decision)
                final_decision = "WEB"

            logfire.info("Routing decision finalized.", raw_decision = raw_decision, final_decision = final_decision)

            return {"source_decision": final_decision}

        except Exception as e:
            logfire.error("Error in the routing node, using WEB as fallback..", query = query, error = str(e), exc_info = True)
            return {"source_decision": "WEB"}


@traceable(run_type = "retriever", name = "Node_RetrieveRAG") # LangSmith Decorator
def retrieve_rag_node(state: GraphState) -> dict:
    query = state["query"]
    
    span = logfire.span("Running node: Retrieve RAG", query = query)
    with span:
        try:
            local_retriever = load_retriever()
            results = local_retriever.invoke(query)
            
            # Concatenating the content of each returned document into a single context string.
            context = "\n\n".join([doc.page_content for doc in results])
            
            if not context:
                logfire.info("No RAG context found.")
                return {"rag_context": "No relevant internal documents were found."}
            
            else:
                logfire.info("RAG context was found.", context_length = len(context))
                return {"rag_context": context}
        
        except Exception as e:
            logfire.error("Error in RAG node", query = query, error = str(e), exc_info = True)
            return {"rag_context": f"Error searching internal documents.: {e}"}


@traceable(run_type = "tool", name = "Node_SearchWeb") # LangSmith Decorator
def search_web_node(state: GraphState) -> dict:
    query = state["query"]
    span = logfire.span("Executando Nó: Search Web", query = query)
    
    with span:
        try:
            web_search_tool = DuckDuckGoSearchRun()
            results = web_search_tool.run(query)

            if not results:
                logfire.info("No results were found in the web search.")
                return {"web_results": "No results were found in the web search."}
            
            else:
                logfire.info("Web search results found.", results_length = len(results))
                return {"web_results": results}
        
        except Exception as e:
            logfire.error("Error in Web Search node", query = query, error = str(e), exc_info = True)
            return {"web_results": f"Error while performing a web search.: {e}"}


@traceable(run_type = "llm", name = "Node_GenerateAnswer") # LangSmith Decorator
def generate_answer_node(state: GraphState) -> dict:
    query = state["query"]
    span = logfire.span("Executando Nó: Geração da Resposta", query = query)
    
    with span:
        rag_context = state.get("rag_context")
        web_results = state.get("web_results")
        context_provided = ""
        source_used = "None"

        rag_useful = rag_context != "No relevant internal documents were found."
        web_useful = web_results != "No results were found in the web search."

        if rag_useful:
            context_provided = f"Context of internal documents:\n{rag_context}"
            source_used = "RAG"
            logfire.info("Using RAG context to generate a response.")
        elif web_useful:
            context_provided = f"Web search results:\n{web_results}"
            source_used = "WEB"
            logfire.info("Using web results to generate a response.")
        else:
            context_provided = "No additional information was found in the available sources."
            logfire.info("No useful context found to generate a response.")

        logfire.info('Source(s) for generation', source_used = source_used, rag_context_present = rag_useful, web_results_present = web_useful)

        prompt = f"""You are a helpful and concise technical support assistant. Answer the user's question clearly, using ONLY the information provided in the context below. 
        If the context is not helpful or relevant to the question, state that you did not find specific information about it in the available sources. DO NOT invent answers.

        User query: {query}

        Provided context: {context_provided}

        Concise answer:"""

        # Execution block
        try:

            # Runing the function
            llm_resposta_final = load_llm_final_answer()
            
            # Running the LLM
            response = llm_resposta_final.invoke(prompt)

            # Extract answer contents
            final_answer = response.content

            # LogFire
            logfire.info("Generated final answer", source_used = source_used, answer_length = len(final_answer))
            
            return {"final_answer": final_answer}
       
        except Exception as e:
            logfire.error("Error in the response generation node.", query = query, source_used = source_used, error = str(e), exc_info = True)
            return {"final_answer": f"Sorry, a technical error occurred while trying to generate the final response.: {e}"}


def decide_source_edge(state: GraphState) -> Literal["retrieve_rag_node", "search_web_node"]:
    decision = state["source_decision"]
    logfire.debug("Conditional edge: Deciding next node", current_decision = decision) # LogFire debug level

    if decision == "RAG":
        return "retrieve_rag_node"
    else: # Including "WEB" and any fallback
        return "search_web_node"
########## Function to Compile the Graph and Define Routing Rules ##########

########## Streamlit Configuration ##########