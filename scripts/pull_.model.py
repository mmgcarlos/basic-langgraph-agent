#!/usr/bin/env python
"""Pull Ollama model before running tests."""
import os
import time
import requests
import sys

def pull_model():
    """Pull the Ollama model."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    
    print(f"📥 Pulling model: {model}")
    
    # Check if Ollama is running
    try:
        requests.get(f"{base_url}/api/tags", timeout=5)
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running!")
        print(f"   Please start Ollama: docker-compose up -d ollama")
        sys.exit(1)
    
    # Pull the model
    response = requests.post(
        f"{base_url}/api/pull",
        json={"name": model},
        stream=True,
        timeout=300
    )
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode('utf-8')
                    print(f"   {data}")
                except:
                    pass
        print(f"✅ Model {model} pulled successfully!")
    else:
        print(f"❌ Failed to pull model: {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    pull_model()
