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


########## Etapa 4 - AI Agent components Setting ##########


########## Etapa 5 - LangGraph Agent Execution Flow Setting ##########


########## Etapa 6 - Web App with Streamlit Setting ##########