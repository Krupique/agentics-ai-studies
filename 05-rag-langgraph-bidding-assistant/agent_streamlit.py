import os
import time
import numpy as np
import warnings

import streamlit as st
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# avoid streamlit/pytorch compatibility problems (kept from your original)
import torch

warnings.filterwarnings("ignore")
torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

########## Etapa 1 - Streamlit Interface Setting ##########
st.set_page_config(page_title="Agent for Bidding Process", page_icon=":100:", layout="centered")

st.sidebar.title("Instructions")
st.sidebar.write("""
- Type specific questions about the bidding process to get detailed answers.
- The AI ​​assistant will use the RAG database to generate customized answers.
- Supplementary documents, contracts, and procedures can be used to improve the RAG system (which in this case must be recreated with each new document).
- Generative AI makes mistakes. ALWAYS validate the answers.
""")

if st.sidebar.button("Support"):
    st.sidebar.write("Send an email to: krupck@outlook.com")

st.title("Project - Bidding Process")
st.title("Re-Ranking, Agentic RAG with LangGraph and LM Studio for Bidding Process Assistant")

########## Etapa 2 - LLM Model and RAG Retrive Process Setting ##########
llm = ChatOpenAI(
    model_name="hermes-3-llama-3.2-3b@q6_k",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.3,
    max_tokens=256
)

# exploratory LLM used to generate diverse candidate answers
llm_exploratory = ChatOpenAI(
    model_name="hermes-3-llama-3.2-3b@q6_k",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.9,
    max_tokens=256
)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

vector_db = Chroma(
    persist_directory="rag/chroma_db",
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever()


########## Etapa 3 - Tools for the AI Agent Setting ##########
search = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=5)
web_search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Search updated information on internet about public digging"
)


########## Etapa 4 - AI Agent components Setting ##########
class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0

def agent_decision_step(state: AgentState) -> AgentState:
    q = state.query.lower()
    if any(p in q for p in ["explain", "summarize", "define", "concept", "general", "what is"]):
        state.next_step = "generate"
    elif any(p in q for p in ["search on web", "news", "updated", "recently", "latest informations"]):
        state.next_step = "use_web"
    else:
        state.next_step = "retrieve"
    return state

def use_web_tool(state: AgentState) -> AgentState:
    """
        Directly calls the DuckDuckGo tool and formats the output.
        Adjust if your DuckDuckGo wrapper version uses a different method (e.g., .run vs. .invoke).
    """
    try:
        result = search.run(state.query) 
    except Exception as e:
        result = f"Error to execute the search: {e}"
    # normalize result to string
    if isinstance(result, dict):
        # try common keys
        out = result.get("text") or result.get("output") or str(result)
    else:
        out = str(result)
    state.ranked_response = out or "No information was obtained through web search."
    state.confidence_score = 0.5
    return state

def retrieve_info(state: AgentState) -> AgentState:
    docs = retriever.invoke(state.query)
    # normalize: some retrievers return {"documents": [...]}
    if isinstance(docs, dict) and "documents" in docs:
        docs = docs["documents"]
    state.retrieved_info = docs or []
    return state

def generate_multiple_answers(state: AgentState) -> AgentState:
    # prepare concatenated context
    docs = state.retrieved_info or []
    context = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    max_chars = 4000
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n...[context truncated]..."

    # generate diverse candidates with exploratory LLM
    responses = []
    for _ in range(5):
        try:
            res = (prompt | llm_exploratory | StrOutputParser()).invoke({"input": state.query, "context": context})
        except Exception:
            # fallback to simpler chain if invoke signature differs
            res = (prompt | llm_exploratory | StrOutputParser()).run({"input": state.query, "context": context})
        responses.append(res)
    state.possible_responses = responses
    return state

def _safe_extract_text(obj):
    if isinstance(obj, dict):
        return obj.get("answer") or obj.get("text") or str(obj)
    return str(obj)

def evaluate_similarity(state: AgentState) -> AgentState:
    retrieved_texts = [getattr(d, "page_content", str(d)) for d in state.retrieved_info] if state.retrieved_info else []
    responses = state.possible_responses or []
    response_texts = [_safe_extract_text(r) for r in responses]

    if not retrieved_texts or not response_texts:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    # embed once per unique text
    retrieved_embeddings = np.array(embedding_model.embed_documents(retrieved_texts))
    response_embeddings = np.array(embedding_model.embed_documents(response_texts))

    # normalize
    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1e-10
        return x / n

    re_norm = norm_rows(retrieved_embeddings)
    resp_norm = norm_rows(response_embeddings)

    # similarity matrix R x M
    sim_matrix = resp_norm.dot(re_norm.T)
    similarities = sim_matrix.mean(axis=1).tolist()
    state.similarity_scores = similarities
    return state

########## Etapa 5 - LangGraph Agent Execution Flow Setting ##########


########## Etapa 6 - Web App with Streamlit Setting ##########