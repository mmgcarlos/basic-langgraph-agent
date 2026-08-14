import os
import pytest
from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import graph
from src.agent.tools import search_web, get_current_time

def test_graph_import():
    """Test that the graph loads properly"""
    assert graph is not None

# /tests/test_agent.py
def test_basic_query():
    """Test agent responds to simple queries"""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"messages": [("user", "What is the capital of France?")]},
        config
    )
    last_message = result["messages"][-1]
    
    # Extract text content (handles both string and list formats)
    if isinstance(last_message.content, list):
        content_text = " ".join([part.get('text', '') for part in last_message.content if part.get('type') == 'text'])
    else:
        content_text = last_message.content
    
    assert "Paris" in content_text
    assert not last_message.tool_calls  # Shouldn't need tools
    
def test_tool_call():
    """Test agent uses tools when needed"""
    config = {"configurable": {"thread_id": "test-2"}}
    result = graph.invoke(
        {"messages": [("user", "What's the current date and time?")]},
        config
    )
    last_message = result["messages"][-1]
    assert "202" in last_message.content or "202" in str(last_message.content)  # Should contain year

def test_search_tool():
    """Test the search tool directly"""
    result = search_web("What is Python?")
    assert result is not None
    assert len(result) > 0

def test_time_tool():
    """Test the time tool directly"""
    result = get_current_time()
    assert result is not None
    assert len(result) > 0

@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="No API key")
def test_agent_with_gemini():
    """Test agent with actual Gemini (skipped if no key)"""
    config = {"configurable": {"thread_id": "test-3"}}
    result = graph.invoke(
        {"messages": [("user", "Hello, how are you?")]},
        config
    )
    assert result["messages"][-1].content is not None
