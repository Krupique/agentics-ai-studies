import os

from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.tools.websearch import WebSearchTools
from agno.tools.yfinance import YFinanceTools


# ============================================================
# Environment variables
# ============================================================

load_dotenv()

print("[INFO] Environment variables loaded.")


# ============================================================
# Check API key
# ============================================================

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Add it to your .env file."
    )


# ============================================================
# Financial Agent
# ============================================================

financial_agent = Agent(
    name="Financial Agent",
    role="Financial analyst specialized in stocks and company fundamentals.",
    
    model=Groq(
        id="openai/gpt-oss-120b"
    ),

    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_analyst_recommendations=True,
            enable_stock_fundamentals=True,
        )
    ],

    instructions=[
        "Analyze companies and stocks from a financial perspective.",
        "Use YFinance whenever financial data is required.",
        "Generate tables when comparing financial information.",
        "Clearly distinguish facts from your own analysis.",
        "Generate the final answer in English.",
    ],

    markdown=True,
)


# ============================================================
# Web Search Agent
# ============================================================

search_agent = Agent(
    name="Web Search Agent",
    role="Financial news and web research specialist.",

    model=Groq(
        id="openai/gpt-oss-120b"
    ),

    tools=[
        WebSearchTools()
    ],

    instructions=[
        "Search the web whenever current information is required.",
        "Always provide the sources used.",
        "Prefer reliable and authoritative sources.",
        "Prioritize recent financial news.",
        "Generate the final answer in English.",
    ],

    markdown=True,
)


# ============================================================
# Financial Research Team
# ============================================================

research_team = Team(
    name="Financial Research Team",

    model=Groq(
        id="openai/gpt-oss-120b"
    ),

    members=[
        financial_agent,
        search_agent,
    ],

    instructions=[
        "You are a senior financial research team.",
        "Delegate financial analysis to the Financial Agent.",
        "Delegate current news and web research to the Web Search Agent.",
        "Use both agents when the question requires financial data and current news.",
        "Always include the sources used.",
        "Use tables when comparing companies, analysts, or financial metrics.",
        "Clearly separate factual information from analysis.",
        "Generate the final answer in English.",
    ],

    markdown=True,
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    research_team.print_response(
        """
        Summarize the recommendations from analysts about
        Netflix (NFLX) investments and share the latest
        information and news.

        Include:
        1. Current stock information
        2. Analyst recommendations
        3. Important financial fundamentals
        4. Recent relevant news
        5. A comparison table
        6. Sources
        """,
        stream=True,
    )