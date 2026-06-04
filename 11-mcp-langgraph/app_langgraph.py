import os
import asyncio
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from typing import Annotated
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt

load_dotenv()

# Creates an MCP client configured for "math" and "customers" servers.
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["mcp_server_math.py"],
            "transport": "stdio",
        },
        "customers": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)

# Define an asynchronous function to create and compile the agent's state graph.
async def create_graph(math_session, clientes_session):
    
    # Instantiates the Google Generative AI LLM model with an API key.
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash",
        temperature = 0,
        api_key = os.getenv("GOOGLE_API_KEY")
    )
    
    # Loading and concatenating tools
    math_tools = await load_mcp_tools(math_session)
    cust_tools = await load_mcp_tools(clientes_session)
    tools = math_tools + cust_tools
    
    # Links the tools to the LLM for use during execution.
    llm_with_tool = llm.bind_tools(tools)
    
    # Loads the system prompt defined on the math server.
    system_prompt = await load_mcp_prompt(math_session, "system_prompt")
    
    # Create a chat prompt template including the system prompt.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt[0].content),
        MessagesPlaceholder("messages")
    ])
    
    # Combine the prompt template with the LLM that has the tools.
    chat_llm = prompt_template | llm_with_tool

    # Defines the graph's state type, adding messages to the state.
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Function that represents the chat node in the graph.
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Building the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    
    graph_builder.add_edge(START, "chat_node")    
    graph_builder.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
    graph_builder.add_edge("tool_node", "chat_node")
    
    graph = graph_builder.compile(checkpointer = MemorySaver())
    return graph

# Defines the main function that initiates the agent interaction loop.
async def main():

    # Configuration where we can pass configurable parameters to the agent.
    config = {"configurable": {"thread_id": 1234}}
    
    # Open MCP sessions for both servers.
    async with client.session("math") as math_session, client.session("customers") as cust_session:
        
        agent = await create_graph(math_session, cust_session)
        print("\nStarting the AI ​​System...\n")
        while True:
            message = input("\nUser: ")
            response = await agent.ainvoke({"messages": message}, config=config)            
            print("AI System: " + response["messages"][-1].content)

if __name__ == "__main__":
    # Executes the main function using asyncio
    asyncio.run(main())
