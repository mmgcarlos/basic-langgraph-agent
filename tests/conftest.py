"""Pytest configuration and shared fixtures."""
import os
import pytest
import sqlite3
from src.agent.memory import init_database, get_db_path, clear_all_conversations

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """
    Initialize database once before all tests.
    This runs automatically for all tests.
    """
    print("\n📦 Setting up test database...")
    
    # Get database path
    db_path = get_db_path()
    print(f"📊 Database: {db_path}")
    
    # Initialize database (creates tables if they don't exist)
    init_database()
    
    # Clear all existing data for a clean test run
    clear_all_conversations()
    
    print("✅ Database initialized and ready for tests")
    
    # This runs after all tests
    yield
    
    # Optional: Clean up after all tests
    # print("\n🧹 Cleaning up test database...")
    # clear_all_conversations()
    # print("✅ Cleanup complete")

@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available (once per session)."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

@pytest.fixture
def test_thread():
    """Generate a unique thread ID for testing."""
    import uuid
    return f"test-{uuid.uuid4().hex[:8]}"
