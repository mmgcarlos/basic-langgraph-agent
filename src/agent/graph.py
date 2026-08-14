from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
import os

# 1. Define the tool
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    """Search the web for the latest information."""
    return tavily.get_search_context(query=query, search_depth="basic")

# 2. Set up the model and bind tools
tools = [search_web]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools(tools)

# 3. Define the agent node
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# 4. Define routing logic
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
