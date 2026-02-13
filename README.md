# agentics-ai-studies
This is a repository to compile my Agentic AI studies

---
## 1st Project: ReAct for Financial Market Analysis (GROQ api)
ReAct (Reasoning + Action): It's a core paradigm in AI Agent design that combines the generated text from an LLM with external tools, such as searches, apis, sql databases, and so on.<br/>
1. The aim of this project is building an Agent that get stocks data from yFinance API, news about a specific company using DuckDuckGo and combining LLMs to analyse the data.<br/>
2. It's the simpler project that uses Agents. I just get the API key from GROQ and I'm using the class Agent from the phi package to coordinate al this operation.<br/>

* Ideas: Creating a Agent System to analyze FIIs. A specific type of investments from Brazil;<br/> 

---
## 2nd Project: ReAct and RAG for Legal Assistent (OPENAI api)
ReAct (Reasoning + Action): It's a core paradigm in AI Agent design that combines the generated text from an LLM with external tools, such as searches, apis, sql databases, and so on.<br/>
RAG(Retrieval-Augmented Generation): It's a technique that lets an AI look up factual information from an external database before answering a question.<br/>
1. The goal of this project is creating an agent that collect information from different sources and answer complete questions about the company. It's a great project because you can use your company's local documents to provide context to the LLM
   1. I use local documents (RAG), news from internet (DuckDuckGo) and Stock values (yFinance);
   2. Then I compile all of these informations into a very structured prompt;
   3. The prompt is submited to GPT via API; 

---
# 3rd Project: ReAct and Crew AI agents (OPENAI api)
ReAct (Reasoning + Action): It's a core paradigm in AI Agent design that combines the generated text from an LLM with external tools, such as searches, apis, sql databases, and so on.<br/>
CrewAI: It's an open-source framework designed to orchestrate role-playing, collaborative AI agents
1. This project uses a client and server technology with FastAPI where the client ask for the server and its triggers the agents team that works together to generate the best answer.
2. I'm using three agents, each one has a diferent role. The first one is responsible to work as a researcher, the second is going to work as a legal assistent and the third one is going to work as a report creator. Each agent has got your tool to work. For instance, the researcher can use search tool and wikipedia tool. It will allow it searching recent news and information from external sources.
3. Every agents are different instances and each one has your own API key and work independent from the others.
4. The class Agent is from the package CREWAI.

---
# 4th Project: LangGraph Ollama
