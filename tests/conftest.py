"""Pytest configuration and shared fixtures."""
import os
import pytest

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
