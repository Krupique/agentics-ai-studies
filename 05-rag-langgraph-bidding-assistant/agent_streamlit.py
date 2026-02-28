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
st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="centered")

st.sidebar.title("Instructions")
st.sidebar.write("""
- Type specific questions about the bidding process to get detailed answers.
- The AI ​​assistant will use the RAG database to generate customized answers.
- Supplementary documents, contracts, and procedures can be used to improve the RAG system (which in this case must be recreated with each new document).
- Generative AI makes mistakes. ALWAYS validate the answers.
""")


########## Etapa 2 - LLM Model and RAG Retrive Process Setting ##########


########## Etapa 3 - Tools for the AI Agent Setting ##########


########## Etapa 4 - AI Agent components Setting ##########


########## Etapa 5 - LangGraph Agent Execution Flow Setting ##########


########## Etapa 6 - Web App with Streamlit Setting ##########