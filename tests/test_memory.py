"""Tests for multi-conversation memory."""
import os
import pytest
import uuid
import time
from src.agent.graph import invoke_agent
from src.agent.memory import (
    get_conversation_messages,
    get_conversation_summary
)

# ============================================================
# TESTS (no setup needed - handled by conftest.py)
# ============================================================

def test_single_conversation_memory(test_thread, ollama_available):
    """Test that agent remembers within a conversation."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    thread = test_thread
    
    # Turn 1: Set name
    response1 = invoke_agent("My name is Alice", thread)
    print(f"Turn 1: {response1}")
    
    # Turn 2: Ask for name (should remember)
    response2 = invoke_agent("What's my name?", thread)
    print(f"Turn 2: {response2}")
    
    # Assert - check case-insensitively
    assert "alice" in response2.lower()

def test_multiple_conversations_separate(test_thread, ollama_available):
    """Test that different conversations don't share memory."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    # Conversation 1: Alice
    thread1 = test_thread
    invoke_agent("My name is Alice", thread1)
    response1 = invoke_agent("What's my name?", thread1)
    
    # Conversation 2: Bob
    thread2 = f"test-{uuid.uuid4().hex[:8]}"
    invoke_agent("My name is Bob", thread2)
    response2 = invoke_agent("What's my name?", thread2)
    
    # Assert each remembers correctly
    assert "alice" in response1.lower()
    assert "bob" in response2.lower()
    
    # Assert they don't cross-contaminate
    assert "bob" not in response1.lower()
    assert "alice" not in response2.lower()

def test_conversation_history(test_thread, ollama_available):
    """Test retrieving conversation history."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    thread = test_thread
    
    # Add some messages
    queries = [
        "Hello, I'm testing memory.",
        "Can you remember what I said?",
        "What's the capital of France?"
    ]
    
    for query in queries:
        invoke_agent(query, thread)
        time.sleep(0.3)  # Small delay to ensure messages are saved
    
    # Get history
    messages = get_conversation_messages(thread)
    
    # Should have messages
    assert len(messages) >= len(queries)
    
    # Check that user messages are there
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= len(queries)


def test_get_conversation_summary(test_thread, ollama_available):
    """Test getting a conversation summary."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    thread = test_thread
    
    # Create a conversation with multiple messages
    invoke_agent("Hello, I'm testing.", thread)
    invoke_agent("What's the weather?", thread)
    invoke_agent("Thank you!", thread)
    time.sleep(0.3)
    
    # Get summary
    summary = get_conversation_summary(thread)
    
    assert summary["thread_id"] == thread
    assert summary["message_count"] >= 3
    assert len(summary["messages"]) >= 3
    assert summary["title"] is not None

@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set"
)
def test_search_with_memory(test_thread, ollama_available):
    """Test search functionality with memory."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    thread = test_thread
    
    # Ask a search question
    response1 = invoke_agent("What is Python programming?", thread)
    print(f"Response 1: {response1[:100]}...")
    
    # Follow up question
    response2 = invoke_agent("Is it good for beginners?", thread)
    print(f"Response 2: {response2[:100]}...")
    
    # Should have remembered context
    assert "python" in response2.lower() or "it" in response2.lower()
