"""Tests for multi-conversation memory."""
import os
import pytest
import uuid
import time
from src.agent.graph import invoke_agent
from src.agent.memory import (
    list_conversations,
    get_conversation_messages,
    clear_conversation,
    get_conversation_summary,
    clear_all_conversations,
    get_conversation_count
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

def test_list_conversations(test_thread, ollama_available):
    """Test listing all conversations."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    # Clear existing data for clean test
    clear_all_conversations()
    
    # Create several conversations
    threads = []
    for i in range(3):
        new_thread = f"test-{uuid.uuid4().hex[:8]}"
        invoke_agent(f"Test message {i}", new_thread)
        threads.append(new_thread)
        time.sleep(0.3)
    
    # List all conversations
    conversations = list_conversations()
    
    # Should have at least our threads
    thread_ids = [c["thread_id"] for c in conversations]
    for thread in threads:
        assert thread in thread_ids, f"Thread {thread} not found in {thread_ids}"

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

def test_clear_conversation(test_thread, ollama_available):
    """Test clearing a specific conversation."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    # Clear existing data
    clear_all_conversations()
    
    # Create two conversations
    thread1 = test_thread
    thread2 = f"test-{uuid.uuid4().hex[:8]}"
    
    invoke_agent("Hello from thread 1", thread1)
    invoke_agent("Hello from thread 2", thread2)
    time.sleep(0.3)
    
    # Verify both exist
    conversations_before = list_conversations()
    thread_ids_before = [c["thread_id"] for c in conversations_before]
    assert thread1 in thread_ids_before
    assert thread2 in thread_ids_before
    
    # Clear one
    result = clear_conversation(thread1)
    assert result is True
    
    # Verify it's gone
    conversations_after = list_conversations()
    thread_ids_after = [c["thread_id"] for c in conversations_after]
    assert thread1 not in thread_ids_after
    assert thread2 in thread_ids_after

def test_clear_all_conversations(test_thread, ollama_available):
    """Test clearing all conversations."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    # Clear existing data
    clear_all_conversations()
    
    # Create several conversations
    threads = []
    for i in range(3):
        thread = f"test-{uuid.uuid4().hex[:8]}"
        invoke_agent(f"Message {i}", thread)
        threads.append(thread)
        time.sleep(0.3)
    
    # Verify they exist
    conversations_before = list_conversations()
    thread_ids_before = [c["thread_id"] for c in conversations_before]
    for thread in threads:
        assert thread in thread_ids_before
    
    # Clear all
    count = clear_all_conversations()
    assert count >= 3
    
    # Check all are gone
    conversations_after = list_conversations()
    thread_ids_after = [c["thread_id"] for c in conversations_after]
    for thread in threads:
        assert thread not in thread_ids_after

def test_memory_persistence(tmp_path, test_thread, ollama_available):
    """Test that memory persists across sessions."""
    if not ollama_available:
        pytest.skip("Ollama not running")
    
    # Use a temporary database for this test
    test_db = str(tmp_path / "test.db")
    os.environ["CHECKPOINT_DB"] = test_db
    
    try:
        thread = test_thread
        
        # Initialize database for this test
        from src.agent.memory import init_database
        init_database()
        
        # First session: Store memory
        invoke_agent("My favorite color is blue", thread)
        
        # Simulate restart by creating a new graph instance
        import importlib
        import sys
        
        # Clear the cached graph
        if "src.agent.graph" in sys.modules:
            importlib.reload(sys.modules["src.agent.graph"])
        
        # Import fresh
        from src.agent.graph import invoke_agent as invoke_agent_restart
        
        # Should still remember
        response = invoke_agent_restart("What's my favorite color?", thread)
        assert "blue" in response.lower()
    
    finally:
        # Cleanup
        os.environ.pop("CHECKPOINT_DB", None)
        # Remove temp database
        if os.path.exists(test_db):
            os.remove(test_db)

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
