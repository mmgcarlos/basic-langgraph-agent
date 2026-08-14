import os
from tavily import TavilyClient

# Initialize Tavily client
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    """Search the web for current information."""
    try:
        result = tavily.get_search_context(
            query=query,
            search_depth="basic",
            max_tokens=1000
        )
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
