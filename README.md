# 🧠 LangGraph Agent with Gemini & Ollama Support

A flexible, agentic AI assistant built with LangGraph that can use tools (web search, date/time) and supports both **Gemini** (production) and **Ollama** (local testing).

[![Deploy LangGraph Agent](https://github.com/YOUR_USERNAME/basic-langgraph-agent/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/basic-langgraph-agent/actions/workflows/deploy.yml)

## ✨ Features

- 🤖 **Agentic Workflow**: LangGraph-based agent with tool-calling capabilities
- 🔧 **Built-in Tools**: Web search (Tavily) and current date/time
- 🆓 **Free Testing**: Run tests locally with Ollama (no API quotas!)
- ☁️ **Production Ready**: Deploy with Gemini API
- 🔄 **GitHub Actions CI/CD**: Automated testing and deployment
- 🧪 **Comprehensive Tests**: Unit and integration tests included

## 🚀 Quick Start

### 1. Clone and Install
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/basic-langgraph-agent.git
cd basic-langgraph-agent
python -m venv venv
source venv/bin/activate
pip install -e .
\`\`\`

### 2. Set Up Environment Variables
\`\`\`bash
cp .env.example .env
# Edit .env with your API keys
\`\`\`

### 3. Run the Agent
\`\`\`python
from src.agent.graph import graph
result = graph.invoke({"messages": [("user", "What is the capital of France?")]})
print(result["messages"][-1].content)
\`\`\`

## 🧪 Testing

### With Ollama (Free, No Quota)
\`\`\`bash
docker-compose up -d ollama
python scripts/pull_model.py
USE_OLLAMA=true pytest tests/ -v
\`\`\`

### With Gemini (Uses API Quota)
\`\`\`bash
GOOGLE_API_KEY=your_key pytest tests/ -v
\`\`\`

## 📁 Project Structure

\`\`\`
basic-langgraph-agent/
├── .github/workflows/deploy.yml
├── src/agent/
│   ├── graph.py
│   └── tools.py
├── tests/
│   ├── test_agent.py
│   └── test_tools.py
├── docker-compose.yml
├── langgraph.json
├── pyproject.toml
└── README.md
\`\`\`

## 🤖 Supported Models

| Provider | Model | Use Case | Cost |
|----------|-------|----------|------|
| Gemini | gemini-2.0-flash | Production | Free (limited) |
| Ollama | llama3.1:8b | Testing | Free |
| Ollama | qwen2.5:7b | Testing | Free |

## 🚢 Deployment

Push to \`main\` branch - GitHub Actions automatically tests and deploys!

## 📝 License

MIT License

---
**Happy Building!** 🚀
EOF
