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

########## Auxiliary Functions ##########

########## Functions for Graph Nodes in LangGraph ##########

########## Function to Compile the Graph and Define Routing Rules ##########

########## Streamlit Configuration ##########