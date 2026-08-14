import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from .tools import search_web, get_current_time

# 1. Define tools
tools = [search_web, get_current_time]
tool_node = ToolNode(tools)

# 2. Setup LLM
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.environ.get("OPENAI_API_KEY")
).bind_tools(tools)

# 3. Agent node
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# 4. Routing logic
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Build graph with memory
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 6. Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
