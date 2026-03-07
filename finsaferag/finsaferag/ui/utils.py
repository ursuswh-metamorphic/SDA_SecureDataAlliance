"""
Utility functions for Streamlit UI
"""
from typing import Dict, Any, List
import json
from datetime import datetime


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_timestamp


def export_chat_history(messages: List[Dict[str, Any]], format: str = "json") -> str:
    """Export chat history to JSON or Markdown"""
    if format == "json":
        return json.dumps(messages, indent=2, ensure_ascii=False)
    
    elif format == "markdown":
        md_lines = ["# Chat History\n"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            timestamp = msg.get("timestamp", "")
            content = msg["content"]
            
            md_lines.append(f"## {role} ({timestamp})\n")
            md_lines.append(f"{content}\n")
            
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]
                if meta.get("privacy_applied"):
                    md_lines.append("\n**Privacy Stats:**\n")
                    stats = meta.get("privacy_stats", {})
                    md_lines.append(f"- PII Detected: {stats.get('pii_detected', 0)}\n")
                    md_lines.append(f"- Sentences Removed: {stats.get('sentences_removed', 0)}\n")
            
            md_lines.append("\n---\n\n")
        
        return "".join(md_lines)
    
    return ""


def calculate_chat_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from chat history"""
    stats = {
        "total_messages": len(messages),
        "user_messages": 0,
        "assistant_messages": 0,
        "total_pii_detected": 0,
        "total_sentences_removed": 0,
        "avg_response_time": 0.0,
        "privacy_applied_count": 0
    }
    
    response_times = []
    
    for msg in messages:
        if msg["role"] == "user":
            stats["user_messages"] += 1
        elif msg["role"] == "assistant":
            stats["assistant_messages"] += 1
            
            if msg.get("metadata"):
                meta = msg["metadata"]
                
                if meta.get("privacy_applied"):
                    stats["privacy_applied_count"] += 1
                
                if meta.get("privacy_stats"):
                    pstats = meta["privacy_stats"]
                    stats["total_pii_detected"] += pstats.get("pii_detected", 0)
                    stats["total_sentences_removed"] += pstats.get("sentences_removed", 0)
                
                if meta.get("response_time"):
                    response_times.append(meta["response_time"])
    
    if response_times:
        stats["avg_response_time"] = sum(response_times) / len(response_times)
    
    return stats

