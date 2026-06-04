import os
import asyncio
from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict

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

# MCP Client
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


async def create_graph(math_session, customers_session):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Load tools
    math_tools = await load_mcp_tools(math_session)
    customer_tools = await load_mcp_tools(customers_session)

    tools = math_tools + customer_tools

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Load MCP system prompt
    system_prompt = await load_mcp_prompt(
        math_session,
        "system_prompt"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt[0].content),
            MessagesPlaceholder("messages"),
        ]
    )

    chat_llm = prompt_template | llm_with_tools

    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    def chat_node(state: State):

        response = chat_llm.invoke(
            {"messages": state["messages"]}
        )

        return {
            "messages": [response]
        }

    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges(
        "chat_node",
        tools_condition,
        {
            "tools": "tool_node",
            "__end__": END,
        },
    )

    graph_builder.add_edge("tool_node", "chat_node")

    graph = graph_builder.compile(
        checkpointer=MemorySaver()
    )

    return graph


def extract_text(message):
    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(item["text"])

        return "".join(parts)
    return str(content)

async def main():
    config = {
        "configurable": {
            "thread_id": "1234"
        }
    }

    async with (
        client.session("math") as math_session,
        client.session("customers") as cust_session
    ):

        agent = await create_graph(
            math_session,
            cust_session
        )

        print("\nStarting AI System...\n")
        while True:
            user_message = input("\nUser: ")
            if user_message.lower() in ["exit", "quit"]:
                break
            try:
                response = await agent.ainvoke(
                    {"messages": user_message},
                    config=config,
                )
                last_message = response["messages"][-1]
                answer = extract_text(last_message)
                print(f"\nAI System: {answer}")

            except Exception as e:
                print("\nERROR:")
                print(type(e).__name__)
                print(e)

if __name__ == "__main__":
    asyncio.run(main())