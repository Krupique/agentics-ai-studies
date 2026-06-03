import asyncio
import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from agents import Agent, Runner, trace
from agents import set_default_openai_key
from firecrawl import FirecrawlApp
from agents.tool import function_tool

load_dotenv()

st.set_page_config(page_title="Henrique Secchi", page_icon=":100:", layout="wide")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")


with st.sidebar:

    st.title("APIs Settings")
    
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.openai_api_key,
        type="password"
    )
    
    firecrawl_api_key = st.text_input(
        "Firecrawl API Key", 
        value=st.session_state.firecrawl_api_key,
        type="password"
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        set_default_openai_key(openai_api_key)
    
    if firecrawl_api_key:
        st.session_state.firecrawl_api_key = firecrawl_api_key

st.title("📘 AI Agents for Deep Research with OpenAI Agents SDK and Firecrawl")

research_topic = st.text_input("Enter the Topic to Search:", placeholder = "For example: What is Hi-Fi audio and what devices should be used?")


@function_tool
async def deep_research(query: str, max_depth: int, time_limit: int, max_urls: int) -> Dict[str, Any]:
    try:
        firecrawl_app = FirecrawlApp(api_key = st.session_state.firecrawl_api_key)
        # Parameters for in-depth research
        params = {
            "maxDepth": max_depth,
            "timeLimit": time_limit,
            "maxUrls": max_urls
        }
        
        # Callback function to display progress activities in the interface.
        def on_activity(activity):
            st.write(f"[{activity['type']}] {activity['message']}")
        
        # Displays a loading indicator while the search is in progress.
        with st.spinner("Running deep research..."):
            results = firecrawl_app.deep_research( # https://docs.firecrawl.dev/features/search?search=deep+research
                query = query,
                params = params,
                on_activity = on_activity
            )
        
        # Returns structured results with final analysis and sources found.
        return {
            "success": True,
            "final_analysis": results['data']['finalAnalysis'],
            "sources_count": len(results['data']['sources']),
            "sources": results['data']['sources']
        }

    # Handles errors and displays error messages to the user.
    except Exception as e:
        st.error(f"Deep research error: {str(e)}")
        return {"error": str(e), "success": False}
    

# Configuration of the agent responsible for the initial search.
research_agent = Agent(
    name = "research_agent",
    instructions = """You are a research assistant who can conduct web research on any topic.
        When presented with a research topic or question:

        1. Use the deep_research tool to gather comprehensive information.

        Always use these parameters:

        * max_depth: 3 (for moderate depth)
        * time_limit: 180 (3 minutes)
        * max_urls: 5 (sufficient sources)

        2. The tool will search the web, analyze various sources, and provide a summary.
        3. Review the research results and organize them into a well-structured report.
        4. Include appropriate citations for all sources.
        5. Highlight key findings and insights.
    """,
    tools = [deep_research]
)

# Configuration of the agent responsible for improving the report.
elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""
        You are a content enhancement specialist, specializing in research writing.

        When a research report is presented:

        1. Analyze the structure and content of the report
        2. Improve the report by:
            - Adding more detailed explanations of complex concepts
            - Including relevant examples, case studies, and real-world applications
            - Expanding on key points with additional context and nuance
            - Adding descriptions of visual elements (graphs, diagrams, infographics)
            - Incorporating the latest trends and future forecasts
            - Suggesting practical implications for different stakeholders
        3. Maintaining academic rigor and factual accuracy
        4. Preserving the original structure while making it more comprehensive
        5. Ensuring that all additions are relevant and valuable to the topic
    """
)

# Main function that orchestrates the flow of research and development.
async def run_research_process(topic: str):
    
    # Loading indicator for initial search.
    with st.spinner("Conduzindo a pesquisa inicial..."):
        research_result = await Runner.run(research_agent, topic)
        initial_report = research_result.final_output
    
    # Expandable to display the initial report to the user.
    with st.expander("Visualizar o Relatório da Pesquisa"):
        st.markdown(initial_report)
    
    # Loading indicator during the report enhancement phase.
    with st.spinner("Improving the report with additional information..."):
        
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}
        
        INITIAL RESEARCH REPORT:
        {initial_report}
        
        Supplement this research report with additional information, examples, case studies, and more in-depth insights, while maintaining its academic rigor and factual accuracy..
        """
        
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report = elaboration_result.final_output
    
    return enhanced_report