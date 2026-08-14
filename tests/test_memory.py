"""Tests for multi-conversation memory."""
import os
import pytest
import uuid
from src.agent.graph import invoke_agent
from src.agent.memory import (
    list_conversations,
    get_conversation_messages,
    clear_conversation,
    get_conversation_summary
)

# Skip if no Ollama running
@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available."""
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except:
        return False

@pytest.fixture
def test_thread():
    """Generate a unique thread ID for testing."""
    return f"test-{uuid.uuid4().hex[:8]}"

def test_single_conversation_memory():
    """Test that agent remembers within a conversation."""
    thread = test_thread()
    
    # Turn 1: Set name
    response1 = invoke_agent("My name is Alice", thread)
    print(f"Turn 1: {response1}")
    
    # Turn 2: Ask for name (should remember)
    response2 = invoke_agent("What's my name?", thread)
    print(f"Turn 2: {response2}")
    
    # Assert
    assert "Alice" in response2 or "alice" in response2.lower()

def test_multiple_conversations_separate():
    """Test that different conversations don't share memory."""
    # Conversation 1: Alice
    thread1 = test_thread()
    invoke_agent("My name is Alice", thread1)
    response1 = invoke_agent("What's my name?", thread1)
    
    # Conversation 2: Bob
    thread2 = test_thread()
    invoke_agent("My name is Bob", thread2)
    response2 = invoke_agent("What's my name?", thread2)
    
    # Assert each remembers correctly
    assert "Alice" in response1
    assert "Bob" in response2
    
    # Assert they don't cross-contaminate
    assert "Bob" not in response1
    assert "Alice" not in response2

def test_conversation_history():
    """Test retrieving conversation history."""
    thread = test_thread()
    
    # Add some messages
    queries = [
        "Hello, I'm testing memory.",
        "Can you remember what I said?",
        "What's the capital of France?"
    ]
    
    for query in queries:
        invoke_agent(query, thread)
    
    # Get history
    messages = get_conversation_messages(thread)
    
    # Should have messages
    assert len(messages) >= len(queries)
    
    # Check that user messages are there
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= len(queries)

def test_list_conversations():
    """Test listing all conversations."""
    # Create several conversations
    threads = []
    for i in range(3):
        thread = test_thread()
        invoke_agent(f"Test message {i}", thread)
        threads.append(thread)
    
    # List all conversations
    conversations = list_conversations()
    
    # Should have at least our threads
    thread_ids = [c["thread_id"] for c in conversations]
    for thread in threads:
        assert thread in thread_ids

def test_get_conversation_summary():
    """Test getting a conversation summary."""
    thread = test_thread()
    
    # Create a conversation with multiple messages
    invoke_agent("Hello, I'm testing.", thread)
    invoke_agent("What's the weather?", thread)
    invoke_agent("Thank you!", thread)
    
    # Get summary
    summary = get_conversation_summary(thread)
    
    assert summary["thread_id"] == thread
    assert summary["message_count"] >= 3
    assert len(summary["messages"]) >= 3
    assert summary["title"] is not None

def test_clear_conversation():
    """Test clearing a specific conversation."""
    # Create two conversations
    thread1 = test_thread()
    thread2 = test_thread()
    
    invoke_agent("Hello", thread1)
    invoke_agent("Hello", thread2)
    
    # Clear one
    result = clear_conversation(thread1)
    assert result is True
    
    # Check it's gone
    messages1 = get_conversation_messages(thread1)
    messages2 = get_conversation_messages(thread2)
    
    assert len(messages1) == 0
    assert len(messages2) > 0

def test_memory_persistence(tmp_path):
    """Test that memory persists after agent restart."""
    # Use a temporary database for this test
    test_db = str(tmp_path / "test.db")
    os.environ["CHECKPOINT_DB"] = test_db
    
    try:
        # First session: Store memory
        thread = test_thread()
        invoke_agent("My favorite color is blue", thread)
        
        # Second session: New import should still remember
        import importlib
        import sys
        # Reimport graph module to simulate restart
        if "src.agent.graph" in sys.modules:
            importlib.reload(sys.modules["src.agent.graph"])
        
        from src.agent.graph import invoke_agent as invoke_agent_restart
        
        response = invoke_agent_restart("What's my favorite color?", thread)
        
        # Should still remember
        assert "blue" in response.lower()
    
    finally:
        # Cleanup
        os.environ.pop("CHECKPOINT_DB", None)
