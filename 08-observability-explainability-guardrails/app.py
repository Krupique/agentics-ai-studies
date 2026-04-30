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
        


########## Functions for Graph Nodes in LangGraph ##########

########## Function to Compile the Graph and Define Routing Rules ##########

########## Streamlit Configuration ##########