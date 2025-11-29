# Smoke-test key LangChain / LangGraph entry points that MLPatrol relies on.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

print("✅ LangChain / LangGraph imports succeeded!")
