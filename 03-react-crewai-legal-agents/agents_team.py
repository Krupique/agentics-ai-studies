import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import BaseTool
from typing import Type

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

from pydantic import BaseModel


print('\nStarting the work of the Agents Team based on generative AI\n')
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("There was no API key found.")

OPENAI_API_KEY=api_key
print('\nAgents API keys were successfully loaded!\n')

class SearchInput(BaseModel):
    query: str

class DuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Executes web search using DuckDuckGo"
    args_schema: Type[BaseModel] = SearchInput 

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)

class WikipediaTool(BaseTool):
    name: str = "Wikipedia Search"
    description: str = "Search contents on Wikipedia"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)


# Instantiate tools
search_tool = DuckDuckGoTool()
wikipedia_tool = WikipediaTool()

llm_researcher = LLM(model="gpt-4o", temperature=0.7)
llm_legal_assistant = LLM(model="gpt-4o", temperature=0.2)
llm_reporter = LLM(model="gpt-4o", temperature=0.5)


# Creating the Researcher Agent
researcher = Agent(
    role='Senior Researcher Analyst',
    goal='Do complete searches and collect relevant data',
    backstory="""Você é um analista de pesquisa sênior com 15 anos de experiência em análise de dados. 
    You are a senior researcher analyst with fifteen years of experience in data analysis and market reasearch. 
    You have a postgraduation in Data Science and you are a specialist in recognize patterns and trend analysis. 
    You are great in:
    1. Collect and validate informations from different sources
    2. Identify trends and emergents patterns
    3. Critical evaluation of the quality and relevancy of the data
    4. Provide context and insights for complex topics""",
    verbose=True,
    tools=[search_tool, wikipedia_tool],  
    allow_delegation=True,
    llm=llm_researcher
)

# Creating the Legal Asistent Agent
legal_assistent = Agent(
    role='Legal Assistent',
    goal='Process and transform the data in useful format',
    backstory="""You are a senior legal assistent with experience in the law area,
    you are specialized in the legal standards of the USA. 
    Your scpecialities include:
    1. Preparing and organizing legal documents, such as contracts and petitions
    2. Doing legal researches to support lawyers in specific cases
    3. Following procedural deadlines and to managing the calendar of the office tasks  
    4. Rewrite legal opions under lawyers supervision
    You always guarantee the data integrity when you prepare them to analysis.""",
    verbose=True,
    allow_delegation=True,
    llm=llm_legal_assistant 
)

# Creating the Reporter Agent
reporter = Agent(
    role="Report creator",
    goal="Create comprehensive and details reports",
    backstory="""You are a specialist in creating reports and technical editor with an attentive to details.
    Your strenghts include:
    1. Creating objectives reports, clear and comprehensive
    2. Translate complex data into comprehensive insights
    3. Structure informations more clear as possible
    4. Keep the consistency and the patterns on the reports
    You excel in becoming the complex informations into comprehensive informations for all stakeholders.""",
    verbose=True,
    allow_delegation=True,
    llm=llm_reporter 
)

# Executing the agents team
def execute_agents_team(topic):
    try:
        tasks = [
            Task(
                description=f"""Search about {topic} in a detailed way. 
                Collect the data and relevant informations from reliable sources.
                Provide a summary from the most relevant discoveries""",
                agent=researcher,
                expected_output="A structured summary with the main discoveries about the searched topic."
            ),
            Task(
                description=f"""
                Clean and process the collected data about {topic}.
                Identify and process any missing value or anomalies, if they exist.
                Prepare the data to analyse and report creation.
                """,
                agent=legal_assistent,
                expected_output="Processed data and organized, ready to final report generation."
            ),
            Task(
                description=f"""
                Create a comprehensive report about {topic} using processed data.
                Include relevant insights.
                Guarantee that the report be clear and actionable""",
                agent=reporter,
                expected_output="A detailed and well structered report, having relevant insights about the topic."
            )
        ]

        # Create the Agent Team (Crew)
        legal_team = Crew(agents=[researcher, legal_assistent, reporter],
                             tasks=tasks,
                             verbose=True)

        print("\nStarting the agent team execution...\n")
        # Initialize the execution and saving the result
        result = legal_team.kickoff()
        print("\nExecution finished!\n")
        return result

    except Exception as e:
        return f"There was an error: {str(e)}"

if __name__ == "__main__":
    # Defining the topic
    topic = "Pretrial detention and post-conviction incarceration pending appeal in U.S. law"
    print('\nDefinied topic. The agent team will start working!\n')
    # Extract the result
    result = execute_agents_team(topic)
    print("\nResult:", result)