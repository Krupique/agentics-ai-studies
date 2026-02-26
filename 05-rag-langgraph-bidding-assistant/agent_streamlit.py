import os
import time
import numpy as np
import warnings

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


########## Etapa 2 - LLM Model and RAG Retrive Process Setting ##########


########## Etapa 3 - Tools for the AI Agent Setting ##########


########## Etapa 4 - AI Agent components Setting ##########


########## Etapa 5 - LangGraph Agent Execution Flow Setting ##########


########## Etapa 6 - Web App with Streamlit Setting ##########