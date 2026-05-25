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
@st.cache_resource
def compile_graph():
    span = logfire.span("Compiling LangGraph graph")
    
    with span:
        print("LOG - Compiling LangGraph graph...")
        
        try:
            
            # Setting the nodes
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("route_query_node", route_query_node)
            graph_builder.add_node("retrieve_rag_node", retrieve_rag_node)
            graph_builder.add_node("search_web_node", search_web_node)
            graph_builder.add_node("generate_answer_node", generate_answer_node)
            graph_builder.set_entry_point("route_query_node")

            # Defining the condition for routing
            graph_builder.add_conditional_edges("route_query_node", decide_source_edge, {
                "retrieve_rag_node": "retrieve_rag_node",
                "search_web_node": "search_web_node",
            })

            # Adding sequential edges
            graph_builder.add_edge("retrieve_rag_node", "generate_answer_node")
            graph_builder.add_edge("search_web_node", "generate_answer_node")
            graph_builder.add_edge("generate_answer_node", END)

            # Compling the graph
            app = graph_builder.compile()
            print("LOG - Graph compiled successfully.!")
            logfire.info("Graph compiled successfully.")
            return app
        
        except Exception as e:
            
            print(f"LOG - Serious error compiling the graph.: {e}")
            logfire.critical("Error compiling the graph", error = str(e), exc_info = True)
            raise e


########## Streamlit Configuration ##########
st.title("🤖 Technical Support Assistant") 

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you as your technical assistant today?"}]

# Messages history
for message in st.session_state.messages:
    
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "source" in message:
            
            st.caption(f"Searched source: {message['source']}")


llm_final = load_llm_final_answer()
retriever_rag = load_retriever()
compiled_app = compile_graph()

# User input via chat_input
if user_query := st.chat_input("Your question about technical support..."):
    # Creates a span in LogFire to track the entire request, storing the user's query
    span_chat = logfire.span("Processing user query via Chat Interface", query = user_query) # Span for the entire request

    # Opens the span context to group the query processing operations
    with span_chat:

        # Registers in LogFire that a new query was received via chat_input
        logfire.info("New query received from chat input.")

        # Adds the user's message to the chat history in the session and displays it
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Displays the user's message in the chat with the "user" role
        with st.chat_message("user"):
            st.markdown(user_query)

        # Starts the block to display the assistant's response in the chat
        with st.chat_message("assistant"):

            # Uses st.status to provide detailed loading feedback
            with st.status("Thinking... 🧠", expanded = False) as status:

                # Starts the try block to catch errors in the agent's processing
                try:
                    # Displays text indicating the analysis of the question and the decision about the source
                    st.write("Analyzing your question and deciding on the best source...")

                    # Prepares the input dictionary to invoke the agent graph
                    inputs = {"query": user_query}

                    # Executes the compiled LangGraph with the provided inputs
                    final_state = compiled_app.invoke(inputs)

                    # Updates the status indicating which source is being consulted
                    st.write(f"Consulting {final_state.get('source_decision', 'source unkonw')}...") # Updates the status

                    # Extracts the final answer from the state returned by the graph
                    final_answer = final_state.get("final_answer", "Unable to generate an answer.")

                    # Extracts the decision from the source used
                    source = final_state.get('source_decision', 'N/A')

                    # Updates the status to complete, indicating that the answer is ready
                    status.update(label = "Answer ready!", state = "complete", expanded = False) # Updates status to complete

                # In case of an error in invoking the graph, catches the exception
                except Exception as e:

                    # Logs the error to LogFire, including the query and traceback
                    logfire.error("Error invoking the main LangGraph graph from the Chat Interface", query = user_query, error = str(e), exc_info = True)

                    # Displays an error message in the user interface with details of the exception
                    st.error(f"An unexpected error occurred. Detail: {e}")

                    # Defines the displayed response as an error message
                    final_answer = f"Sorry, a technical error occurred: {e}"

                    # Defines the source as "Error" for logging
                    source = "Error"

                    # Updates the status indicating a processing error
                    status.update(label = "Processing error", state = "error", expanded = True)

            # Displays the final response generated by the assistant in the chat
            st.markdown(final_answer)

            # Adds the assistant's response to the chat history, including the consulted source
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "source": source # Stores the source along with the message
            })

# Instructions can go to an expander or modal instead of the sidebar for a cleaner look
with st.expander("ℹ️ Instructions and Notes"):
    st.write("""
        - Ask specific questions about your technical query.
        - The system will automatically decide the best source (internal documents or web).
        - Generative AI can make mistakes. ALWAYS validate critical information.
        - Logs and Traces are sent to **Pydantic LogFire** and **LangSmith**.
    """)

if st.button("DSA Support"):

st.write("Questions? Send an email to: suporte@datascienceacademy.com.br")