import os
import sqlite3
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from .tools import search_web, get_current_time
from .memory import get_db_path

# 1. Define tools
tools = [search_web, get_current_time]
tool_node = ToolNode(tools)

# 2. Setup LLM
def get_model():
    """Get the appropriate model based on environment variables."""
    use_ollama = os.environ.get("USE_OLLAMA", "false").lower() == "true"
    
    if use_ollama:
        # Use Ollama for testing/local development
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        ).bind_tools(tools)
    else:
        # Use Gemini for production
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.3,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        ).bind_tools(tools)

model = get_model()

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
db_path = get_db_path()
conn = sqlite3.connect(db_path)
sqlMemory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=sqlMemory)

# 7. Convenience functions
def invoke_agent(query: str, thread_id: str = "default") -> str:
    """Invoke the agent with a query and thread ID."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [("user", query)]},
        config
    )
    
    # Extract response text
    last_message = result["messages"][-1]
    if isinstance(last_message.content, list):
        return " ".join([
            part.get('text', '') 
            for part in last_message.content 
            if part.get('type') == 'text'
        ])
    return last_message.content

def reset_memory():
    """Reset all memory (careful!)."""
    import os
    db_path = os.environ.get("CHECKPOINT_DB", "checkpoints.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"🧹 Reset memory: {db_path}")
    else:
        print("ℹ️ No memory file found")
