#!/usr/bin/env python3
"""
Example 10 — Conversation Continuity
======================================
Shows how to continue a conversation across multiple query() calls using
session IDs and the --continue / --resume flags.
"""

import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ------------------------------------------------------------------
    # Step 1: Start a conversation, capture the session ID
    # ------------------------------------------------------------------
    print("=== Step 1: Initial question ===\n")

    response = client.query(
        "I'm going to ask you a series of questions about Python data structures. "
        "First question: What is a defaultdict? Reply in 2 sentences.",
    )
    print(f"Answer: {response.text}")
    session_id = response.session_id
    print(f"Session ID: {session_id}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Continue using --continue (most recent conversation)
    # ------------------------------------------------------------------
    print("=== Step 2: Continue (most recent) ===\n")

    response = client.query(
        "Second question: How is it different from a regular dict?",
        continue_conversation=True,
    )
    print(f"Answer: {response.text}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Resume a specific session by ID
    # ------------------------------------------------------------------
    if session_id:
        print("=== Step 3: Resume specific session ===\n")

        response = client.query(
            "Third question: When would I use OrderedDict instead?",
            resume=session_id,
        )
        print(f"Answer: {response.text}")
        print()

    # ------------------------------------------------------------------
    # Step 4: Fork — resume but create a new branch
    # ------------------------------------------------------------------
    if session_id:
        print("=== Step 4: Fork into new session ===\n")

        response = client.query(
            "Actually, let's switch topics. Tell me about Python sets instead.",
            resume=session_id,
            fork_session=True,
        )
        new_session = response.session_id
        print(f"Answer: {response.text}")
        print(f"Original session: {session_id}")
        print(f"Forked session:   {new_session}")
        print()

    # ------------------------------------------------------------------
    # Step 5: Use an explicit session_id for a fresh ID
    # ------------------------------------------------------------------
    import uuid

    custom_id = str(uuid.uuid4())
    print(f"=== Step 5: Custom session ID ({custom_id[:8]}...) ===\n")

    response = client.query(
        "Start of a new conversation. What's your favorite sorting algorithm? "
        "Reply in one sentence.",
        session_id=custom_id,
    )
    print(f"Answer: {response.text}")
    print(f"Session: {response.session_id}")


if __name__ == "__main__":
    main()
