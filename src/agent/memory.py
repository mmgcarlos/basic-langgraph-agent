"""Memory management utilities for the agent."""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

def get_conversation_summary(thread_id: str) -> Dict[str, Any]:
    """Get a summary of a conversation."""
    messages = get_conversation_messages(thread_id)
    
    if not messages:
        return {"error": "Conversation not found"}
    
    # Get first user message as title
    title = "New Conversation"
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
            break
    
    return {
        "thread_id": thread_id,
        "title": title,
        "message_count": len(messages),
        "messages": messages
    }

def get_conversation_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Get all messages from a conversation."""
    from langchain_core.messages import HumanMessage, AIMessage
    from .graph import graph
    
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    if not state:
        return []
    
    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({
                "role": "user",
                "content": msg.content,
                "type": "human"
            })
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join([
                    part.get('text', '') 
                    for part in content 
                    if part.get('type') == 'text'
                ])
            messages.append({
                "role": "assistant",
                "content": content,
                "type": "ai",
                "tool_calls": bool(msg.tool_calls)})
            
__all__ = [
    'get_conversation_messages',
    'get_conversation_summary',
]
