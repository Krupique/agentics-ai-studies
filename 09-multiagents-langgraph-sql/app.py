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


@tool
def query_crm_database(sql_query: str) -> str:
    # Docstring that describes the behavior and use of the tool
    """
    Executes a SELECT query ONLY on the SQLite CRM database and returns the results.
    We use this tool to obtain information about customers or interactions.
    Available tables:
        1. tb_clients (columns: customer_id, name, email, phone, company, status, created_at)

        - status can be 'Lead', 'Active', 'Inactive', 'Prospect'

        2. tb_interactions (columns: interaction_id, customer_id, interaction_date, type, notes)

        - type can be 'Email', 'Call', 'Meeting', 'Note'

        Important: Provide ONLY `SELECT` SQL queries. Do not use `UPDATE`, `DELETE`, `INSERT`, or `DROP`.

        Example of a valid SQL query:

        'SELECT name, email FROM tb_clients WHERE status = 'Active';' 
        'SELECT i.interaction_date, i.type, i.notes FROM tb_interactions i JOIN tb_clients c ON i.customer_id = c.customer_id WHERE c.name = \\'João Silva\\' ORDER BY i.interaction_date DESC;'
    """

    print(f"--- Tool query_crm_database receiving SQL: {sql_query} ---")

    if not sql_query.strip().upper().startswith("SELECT"):
        print("!!! SECURITY ERROR: Attempting to execute non-SELECT SQL !!!")
        return "Error: This tool can only execute SELECT queries."

    conn = None
    try:

        if not os.path.exists(DB_FILE):
             return f"Error: Database file '{DB_FILE}' not found. Run the script 'create_crm_db.py' first."

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

        if not results:
            return "No results were found for the query."
        else:
            column_names = [description[0] for description in cursor.description]
            header = " | ".join(column_names)
            rows_str = [" | ".join(map(str, row)) for row in results]
            max_results = 15
            output = f"Results of the query ({len(results)} found)):\n{header}\n" + "\n".join(rows_str[:max_results])

            if len(results) > max_results:
                output += f"\n... (more {len(results) - max_results} results omitted)"

            return output

    except sqlite3.Error as e:
        print(f"!!! SQL Error: {e} while executing '{sql_query}' !!!")
        return f"Erro ao executar a consulta SQL: {e}. Verifique a sintaxe da sua consulta e os nomes das tabelas/colunas."
   
    except Exception as e:
        print(f"!!! Unexpected ERROR in the tool: {e} !!!")
        return f"An unexpected error occurred in the database tool: {e}"
    
    finally:
        if conn:
            conn.close()


# Tools list
tools = [query_crm_database]

# Creates the object for the tool node.
tool_node = ToolNode(tools) 


# Define the function that creates a "runnable" agent from an LLM and a system prompt.
def create_runnable_agent(llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name = "messages"),
        ]
    )
    agent_runnable = prompt | llm.bind_tools(tools)
    return agent_runnable