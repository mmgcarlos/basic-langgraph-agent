"""Memory management utilities for the agent."""
import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

def get_db_path():
    """Get the database path with directory creation."""
    db_path = os.environ.get("CHECKPOINT_DB", "checkpoints.db")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return db_path
    
def get_connection():
    """Get SQLite connection."""
    return sqlite3.connect(get_db_path())

def init_database():
    """Initialize the database if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create checkpoints table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            state BLOB,
            PRIMARY KEY (thread_id, checkpoint_id)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_thread_id ON checkpoints(thread_id)
    """)
    
    conn.commit()
    conn.close()

def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations with metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            thread_id,
            COUNT(*) as message_count,
            MIN(created_at) as started_at,
            MAX(created_at) as last_updated
        FROM checkpoints 
        GROUP BY thread_id
        ORDER BY last_updated DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "thread_id": row[0],
            "message_count": row[1],
            "started_at": row[2],
            "last_updated": row[3]
        }
        for row in rows
    ]

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
                "tool_calls": bool(msg.tool_calls)
            })
    
    return messages

def clear_conversation(thread_id: str) -> bool:
    """Clear a specific conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?",
        (thread_id,)
    )
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_conversation_count() -> int:
    """Get total number of conversations."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
    count = cursor.fetchone()[0]
    conn.close()
    return count

# ============================================================
# CLEAR FUNCTIONS
# ============================================================

def clear_conversation(thread_id: str) -> bool:
    """Clear a specific conversation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?",
        (thread_id,)
    )
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_conversations() -> int:
    """Clear all conversations."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM checkpoints")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

__all__ = [
    'get_db_path',
    'init_database',
    'list_conversations',
    'get_conversation_messages',
    'get_conversation_summary',
    'get_conversation_count',
    'clear_conversation',
    'clear_all_conversations',
]
