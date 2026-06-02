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


# Defines the function of the Groq agent node responsible for interacting with the CRM.
def groq_agent_node(state: AgentState):
    print("\n *** Running the Groq Node (CRM) *** \n")
    try:
        # Initializes the LLM Groq with the model, temperature, and API key.
        llm_groq = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=groq_api_key) 
        
        # Sets the system prompt with instructions for using the CRM tool
        system_prompt = """You are a CRM assistant named Groq (model Llama3).
            Your main function is to answer questions about clients and interactions by querying the CRM database.
            Use the 'query_crm_database' tool by providing a valid SQL SELECT query to retrieve the requested information.
            See the tool description for the database schema (tables: tb_clients, tb_interactions and their columns).
            Be direct and base your answers on the data returned by the tool. If the tool returns an error, inform the user.
            Do not invent information if it is not in the database.
        """
        
        agent_runnable = create_runnable_agent(llm_groq, system_prompt)
        print("Runnable Groq (CRM) created. Invoking...")
        response = agent_runnable.invoke({"messages": state['messages']})
        print(f"Node Groq (CRM) Retrieved Response: Type = {type(response)}, Content = '{response.content[:50]}...'")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Node Groq (CRM) is calling the tool: {response.tool_calls}")
        
        return {"messages": [response]}

    except Exception as e:
        print(f"!!! Groq Node Error (CRM): {e} !!!")
        print(f"An error occurred while conecting the Groq API: {e}")
        error_msg = AIMessage(content = f"[GROQ INTERNAL ERROR]: It was not possible to process with Groq. Detail: {e}", name = "ErrorGroq")
        return {"messages": [error_msg]}
    

# Defines the function of the OpenAI agent node responsible for interacting with the CRM.
def openai_agent_node(state: AgentState):
    print("\n--- Running the OpenAI Node (CRM) ---")
    try:
        llm_openai = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        
        system_prompt = """You are an experienced CRM assistant called OpenAI (GPT model).
            Your goal is to assist the user with information from the CRM database.
            Use the 'query_crm_database' tool to execute SQL SELECT queries and retrieve data about customers or interactions.
            Refer to the tool's description to understand the database schema (tables: tb_clients, tb_interactions; relevant columns such as name, email, status, interaction_date, type, notes).
            Formulate precise SQL SELECT queries based on the user's question.
            Present the results clearly. If you encounter a tool error, report it.
            If the information is not available, indicate this clearly.
        """
        
        agent_runnable = create_runnable_agent(llm_openai, system_prompt)
        print("Runnable OpenAI (CRM) created. Invoking...")
        response = agent_runnable.invoke({"messages": state['messages']})
        print(f"Node OpenAI (CRM) Retrieved Response: Type={type(response)}, Content='{response.content[:50]}...'")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Node OpenAI (CRM) is calling the tool: {response.tool_calls}")

        return {"messages": [response]}

    except Exception as e:
        print(f"!!! OpenAI Node Error (CRM): {e} !!!")
        print(f"An error occurred while conecting the OpenAI API: {e}")
        error_msg = AIMessage(content=f"[GROQ INTERNAL ERROR]: It was not possible to process with OpenAI. Detail: {e}", name="ErrorOpenAI")
        return {"messages": [error_msg]}
    

# Function for the routing node
# The routing logic will be in the next function
# Even though it doesn't have processing logic, it acts as an explicit routing node, making it clear in the graph where the central decision occurs
# In agent flow graphs, it's good practice to have explicit nodes that act as hubs or routers, even if they don't modify the state
def route_junction_node(state: AgentState) -> dict:
    print("--- Routing Junction Node (No State Change) ---")
    return {}


# Define the function responsible for deciding where the router should send the next message
def router_logic(state: AgentState) -> str:
    print("\n--- Routing Logic Function (Deciding Next Step) ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None

    if not last_message:
        print("Logical Decision: No messages in the state, ending.")
        return "__end__"

    print(f"Router analyzing last message: Type={type(last_message).__name__}, Content='{last_message.content[:80]}...'")

    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("Logical Decision: Last AI message has 'tool_calls'. Routing to Tools.")
        return "tools"

    if isinstance(last_message, AIMessage):
        print("Logical Decision: Final AI response received (without tool_calls). Ending the current loop.")
        return "__end__"

    if isinstance(last_message, HumanMessage):
        user_input_current = last_message.content.lower()
        print(f"Analyzing last human message for mentions: '{user_input_current}'")

        if "@openai" in user_input_current:
            print("Logical Decision: Routing to OpenAI (explicit mention in the last message)")
            return "openai_agent"

        elif "@groq" in user_input_current:
            print("Logical Decision: Routing to Groq (explicit mention in the last message)")
            return "groq_agent"

    if isinstance(last_message, ToolMessage):
        print("Logical Decision: Tool result received, routing to an agent (via alternation)...")

    ai_message_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
    print(f"Current AI message count for alternation: {ai_message_count}")

    if ai_message_count % 2 == 0:
        print(f"Logical Decision: Routing to Groq (default/alternating)")
        return "groq_agent"

    else:
        print(f"Logical Decision: Routing to OpenAI (default/alternating)")
        return "openai_agent"
    

# Define the function responsible for compiling the agent's state and transition graph.
def compile_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("openai_agent", openai_agent_node)
    workflow.add_node("groq_agent", groq_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("router", route_junction_node)

    # Connecting the START entry point to the routing node
    workflow.add_edge(START, "router")
    
    # Configuring conditional edges exiting the router based on routing logic.
    workflow.add_conditional_edges(
        "router",
        router_logic,
        {
            "tools": "tools",
            "groq_agent": "groq_agent",
            "openai_agent": "openai_agent",
            "__end__": END
        },
    )
    
    workflow.add_edge("openai_agent", "router")
    workflow.add_edge("groq_agent", "router")
    workflow.add_edge("tools", "router")

    # Compiling the workflow into an executable application.
    app = workflow.compile()

    print("Graph compiled successfully!")
    return app


##### Interface with Streamlit #####
if "app" not in st.session_state:
    if not os.path.exists(DB_FILE):
        st.error(f"Error: The database file '{DB_FILE}' was not found.")
        st.info("Please run the 'create_crm_db.py' script in the same directory to create the database, and then reload this page.")
        st.stop()

    st.write("Initializing the graph for the first time...")

    try:
        st.session_state.app = compile_graph()

        st.session_state.thread_id = "streamlit_thread_crm"
        
        # If the chat history does not exist in the session, it initializes with a welcome message.
        st.session_state.chat_history = [AIMessage(content="Hello! I'm your CRM assistant. Ask me about clients or interactions (e.g., 'Which clients are active?', 'Show oão Smith's interactions').")]
        
        # Displays a success message after initializing the graph.
        st.success("CRM graph initialized.")

    except Exception as e:
        # In case of a critical error in the graph construction, display an error message and the exception.
        st.error(f"Critical error when building the CRM graph.: {e}")
        st.exception(e)

        st.stop()


# Sidebar
st.sidebar.title("Memory")
with st.sidebar.expander("📜 View Full Conversation History", expanded=False):

    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):            
            role = "ai" if isinstance(msg, AIMessage) else ("tool" if isinstance(msg, ToolMessage) else "user")
            
            sender_display = "User"            
            if role == "ai":
                # Count how many AI messages occurred before this one.
                ai_message_index = sum(1 for m in st.session_state.chat_history[:i] if isinstance(m, AIMessage))
                
                # Check if there is explicit mention of the Groq router.
                is_groq_explicit = "@groq" in msg.content.lower()
                
                # Check if there is explicit mention of the OpenAI router.
                is_openai_explicit = "@openai" in msg.content.lower()
                
                # Gets the custom message name, if it exists.
                msg_name = getattr(msg, 'name', None)
                
                # Defines the sender's display based on mention and toggle conditions
                if is_groq_explicit or (not is_openai_explicit and ai_message_index % 2 == 0 and not msg_name):
                    sender_display = "AI (Groq/Llama3)"
                elif is_openai_explicit or (not is_groq_explicit and ai_message_index % 2 != 0 and not msg_name):
                    sender_display = "AI (OpenAI/GPT)"
                elif msg_name:
                    sender_display = f"AI ({msg_name})"
                else:
                    sender_display = "AI (Assistente)"
            
            # If the message is about a tool, adjust the sender's name to 'Tool'.
            elif role == "tool":
                # Get the tool name or use the pattern.
                tool_name = getattr(msg, 'name', 'dsa_query_crm_database')
                sender_display = f"Ferramenta ({tool_name})"
            
            # Renders the sender header.
            st.markdown(f"**{sender_display}:**")
            
            # Displays the message content in a read-only text field.
            st.text_area(label=f"msg_{i}", value=msg.content, height=100, disabled=True, label_visibility="collapsed")
            
            # If the message is from AI and contains tool calls, it is displayed in JSON format.
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                st.write("*Chamada(s) de Ferramenta:*")
                st.json([{'name': tc.get('name', 'N/A'), 'args': tc.get('args', {})} for tc in msg.tool_calls])
            
            # If the message is from a tool and has a caller ID, display the caption with the ID.
            if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id'):
                st.caption(f"ID da Chamada: {msg.tool_call_id}")
            
            st.divider()    
    else:
        st.write("No messages in the history yet.")


st.markdown("### Active Chat")
container_chat = st.container(height = 500)

with container_chat:
    for i, msg in enumerate(st.session_state.chat_history):
        role = "ai" if isinstance(msg, AIMessage) else ("tool" if isinstance(msg, ToolMessage) else "user")
        avatar_icon = "👤"
        sender_name = "User"
        message_role_for_streamlit = "user"

        if role == "ai":
            message_role_for_streamlit = "assistant"
            
            ai_message_index = sum(1 for m in st.session_state.chat_history[:i] if isinstance(m, AIMessage))
            is_groq_explicit = "@groq" in msg.content.lower()
            is_openai_explicit = "@openai" in msg.content.lower()
            
            msg_name = getattr(msg, 'name', None)
            
            # If it's a Groq or an alternating pattern with an even index and no name, set the Groq avatar.
            if is_groq_explicit or (not is_openai_explicit and ai_message_index % 2 == 0 and not msg_name):
                    avatar_icon = "🦙"
                    sender_name = "Groq (Llama3)"
            
            # If it's OpenAI or an odd-indexed, nameless toggle, set the OpenAI avatar.
            elif is_openai_explicit or (not is_groq_explicit and ai_message_index % 2 != 0 and not msg_name):
                    avatar_icon = "🤔"
                    sender_name = "OpenAI (GPT)"
            
            # If a custom name is included in the message, use the system avatar.
            elif msg_name:
                    avatar_icon = "⚠️"
                    sender_name = f"System ({msg_name})"
            
            # Otherwise, it uses a generic assistant avatar.
            else:
                    avatar_icon = "🤖"
                    sender_name = "Assistant"
        
        # If the message is from Tool, adjust role, avatar, and name.
        elif role == "tool":
                message_role_for_streamlit = "assistant"
                avatar_icon = "🛠️"
                sender_name = "Tool"


        with st.chat_message(message_role_for_streamlit, avatar=avatar_icon):
            if role == "tool":
                tool_name = getattr(msg, 'name', 'query_crm_database')
                print(f"**Tool result ({tool_name})**:")
                print(f"{msg.content}")
                print(f"Call ID: {msg.tool_call_id}")
                
            # If the message is from AI, it displays the content and calls to the Tool if available.
            elif role == "ai":
                print(f"**{sender_name}:**")
                if getattr(msg, 'tool_calls', None):
                        print(f"*Calling Tool(s):*")
                        print([{'name': tc.get('name', 'N/A'), 'args': tc.get('args', {})} for tc in msg.tool_calls])
                print(msg.content)
                
            # If it's a message from the User, it displays the text directly.
            else:
                print(msg.content)