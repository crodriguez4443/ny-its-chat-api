"""
conversation_logger.py - Conversation Logging Utility

Logs all user conversations to a JSONL file for analytics and auditing.
Each line in the file represents one exchange (user query + assistant response).
Conversations are grouped by session_id for easy analysis.
"""

import json
import os
from datetime import datetime
from typing import Optional

# Log file path
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "conversation_history.jsonl")


def ensure_log_directory():
    """Ensure the logs directory exists."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created logs directory: {LOG_DIR}")


def get_exchange_number(session_id: str) -> int:
    """
    Get the exchange number for this session by counting previous exchanges.

    Args:
        session_id: The session ID to count exchanges for

    Returns:
        The next exchange number (1-indexed)
    """
    if not os.path.exists(LOG_FILE):
        return 1

    exchange_count = 0
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('session_id') == session_id:
                        exchange_count += 1
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    except Exception as e:
        print(f"Warning: Error reading log file: {e}")
        return 1

    return exchange_count + 1


def log_conversation(
    session_id: str,
    user_role: str,
    user_query: str,
    assistant_response: str,
    conversation_context_length: int,
    chunks_retrieved: int,
    response_time_ms: int
) -> None:
    """
    Log a conversation exchange to the JSONL file.

    Args:
        session_id: Unique session identifier
        user_role: User's role (CONSULTANT, POLICY_MAKER, etc.)
        user_query: The user's question
        assistant_response: The LLM's response
        conversation_context_length: Number of messages in conversation history
        chunks_retrieved: Number of content chunks retrieved from search
        response_time_ms: Time taken to generate response (milliseconds)
    """
    try:
        # Ensure directory exists
        ensure_log_directory()

        # Get exchange number for this session
        exchange_number = get_exchange_number(session_id)

        # Create log entry
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "exchange_number": exchange_number,
            "user_role": user_role,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "conversation_context_length": conversation_context_length,
            "chunks_retrieved": chunks_retrieved,
            "response_time_ms": response_time_ms
        }

        # Append to log file (one JSON object per line)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        print(f"✓ Logged exchange {exchange_number} for session {session_id[:8]}...")

    except Exception as e:
        # Log errors but don't fail the request
        print(f"ERROR: Failed to log conversation: {e}")


def get_conversation_by_session(session_id: str) -> list:
    """
    Retrieve all exchanges for a specific session.

    Args:
        session_id: The session ID to retrieve

    Returns:
        List of exchange dictionaries, ordered by exchange_number
    """
    if not os.path.exists(LOG_FILE):
        return []

    exchanges = []
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('session_id') == session_id:
                        exchanges.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading conversation: {e}")
        return []

    # Sort by exchange number
    exchanges.sort(key=lambda x: x.get('exchange_number', 0))
    return exchanges


def get_total_exchanges() -> int:
    """
    Get the total number of exchanges logged.

    Returns:
        Total number of logged exchanges
    """
    if not os.path.exists(LOG_FILE):
        return 0

    count = 0
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Error counting exchanges: {e}")
        return 0

    return count


def get_unique_sessions() -> int:
    """
    Get the number of unique sessions logged.

    Returns:
        Number of unique session IDs
    """
    if not os.path.exists(LOG_FILE):
        return 0

    session_ids = set()
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    session_ids.add(entry.get('session_id'))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error counting sessions: {e}")
        return 0

    return len(session_ids)


if __name__ == "__main__":
    # Test the logger
    print("Testing conversation logger...")

    # Simulate logging
    test_session_id = "test-session-123"

    log_conversation(
        session_id=test_session_id,
        user_role="CONSULTANT",
        user_query="What are dynamic message signs?",
        assistant_response="Dynamic Message Signs (DMS) are electronic traffic signs...",
        conversation_context_length=2,
        chunks_retrieved=15,
        response_time_ms=3245
    )

    # Retrieve conversation
    print(f"\nRetrieving conversation for session: {test_session_id}")
    conversation = get_conversation_by_session(test_session_id)
    for exchange in conversation:
        print(f"Exchange {exchange['exchange_number']}: {exchange['user_query'][:50]}...")

    # Stats
    print(f"\nTotal exchanges logged: {get_total_exchanges()}")
    print(f"Unique sessions: {get_unique_sessions()}")
