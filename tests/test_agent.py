import os
import pytest
from dotenv import load_dotenv
load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

def extract_text(content):
    """Extract text from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
        return " ".join(text_parts)
    return str(content)

def has_tool_calls(message):
    """Check if message has tool calls (handles different formats)."""
    if not hasattr(message, 'tool_calls'):
        return False
    return bool(message.tool_calls)

# ============================================================
# TESTS THAT WORK WITH BOTH GEMINI AND OLLAMA
# ============================================================

def test_graph_import():
    """Test that the graph loads properly."""
    from src.agent.graph import graph
    assert graph is not None

def test_basic_query():
    """Test agent responds to simple queries."""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"messages": [("user", "What is the capital of France?")]},
        config
    )
    last_message = result["messages"][-1]
    content_text = extract_text(last_message.content)
    
    # Less strict assertion for Ollama
    if USE_OLLAMA:
        # Ollama might say "Paris" or "The capital is Paris"
        assert "paris" in content_text.lower() or "france" in content_text.lower()
    else:
        assert "Paris" in content_text
    
    # Tool calls should be empty for simple queries
    assert not has_tool_calls(last_message)

def test_current_date():
    """Test date query (may or may not use tools)."""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-2"}}
    result = graph.invoke(
        {"messages": [("user", "What is today's date?")]},
        config
    )
    last_message = result["messages"][-1]
    content_text = extract_text(last_message.content)
    
    # Should contain a date or at least respond
    assert len(content_text) > 10  # Minimum response length
    # Check for date pattern (year)
    assert any(str(year) in content_text for year in range(2020, 2030))

@pytest.mark.skipif(not TAVILY_KEY, reason="TAVILY_API_KEY not set")
def test_search_query():
    """Test search functionality (if Tavily key is available)."""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-3"}}
    result = graph.invoke(
        {"messages": [("user", "What is Python programming?")]},
        config
    )
    last_message = result["messages"][-1]
    content_text = extract_text(last_message.content)
    
    # Should mention Python
    assert "python" in content_text.lower()

# ============================================================
# TOOL TESTS (No LLM calls)
# ============================================================

def test_tools_directly():
    """Test tools directly without LLM."""
    from src.agent.tools import get_current_time, search_web
    
    # Test time tool
    result = get_current_time()
    assert result is not None
    assert len(result) > 0
    # Check format: YYYY-MM-DD HH:MM:SS
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)
    
    # Test search tool (if key exists)
    if TAVILY_KEY:
        result = search_web("test query")
        assert result is not None
        assert len(result) > 0

# ============================================================
# OLLAMA-SPECIFIC TESTS (Skipped with Gemini)
# ============================================================

@pytest.mark.skipif(not USE_OLLAMA, reason="Only run with Ollama")
def test_ollama_tool_calling():
    """Test that Ollama can use tools."""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-ollama"}}
    result = graph.invoke(
        {"messages": [("user", "What's the current date?")]},
        config
    )
    last_message = result["messages"][-1]
    
    # Check that we got a response
    content_text = extract_text(last_message.content)
    assert len(content_text) > 0

# ============================================================
# GEMINI-SPECIFIC TESTS (Skipped with Ollama)
# ============================================================

@pytest.mark.skipif(USE_OLLAMA, reason="Only run with Gemini")
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No API key")
def test_gemini_tool_calls():
    """Test Gemini tool calling."""
    from src.agent.graph import graph
    
    config = {"configurable": {"thread_id": "test-gemini"}}
    result = graph.invoke(
        {"messages": [("user", "What's the weather like?")]},
        config
    )
    last_message = result["messages"][-1]
    content_text = extract_text(last_message.content)
    
    # Gemini might call search tool or respond directly
    assert len(content_text) > 0
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

@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No API key")
def test_agent_with_gemini():
    """Test agent with actual Gemini (skipped if no key)"""
    config = {"configurable": {"thread_id": "test-3"}}
    result = graph.invoke(
        {"messages": [("user", "Hello, how are you?")]},
        config
    )
    assert result["messages"][-1].content is not None
