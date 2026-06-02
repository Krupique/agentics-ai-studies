import os
import streamlit as st
import sqlite3 
import operator
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

DB_FILE = "dsa_crm_database.db" 
st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="wide")

st.title("Data Science Academy - Projeto 7")
st.title("🤖 Gerenciamento de Memória e Contexto - Sistema Multi-Agentes de IA com LangGraph Para Automação do CRM e Consulta a Banco de Dados")

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")

if not groq_api_key or not openai_api_key:
    st.warning("⚠️ Set both the Groq and OpenAI keys in the sidebar to continue.")
    st.stop()


class AgentState(TypedDict):
    # Declaring the 'messages' field as a list of BaseMessage, aggregated by the sum operator.
    messages: Annotated[List[BaseMessage], operator.add]