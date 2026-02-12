#!/usr/bin/env python3
"""
Example 04 — Streaming
=======================
Stream Claude's response token-by-token as it generates, rather than
waiting for the full response.  Two approaches shown:

  1. stream_text()  — simple iterator of text chunks
  2. stream()       — full StreamEvent objects with metadata
"""

import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ------------------------------------------------------------------
    # Approach 1: stream_text() — just the text, nothing else
    # ------------------------------------------------------------------
    print("=== stream_text() — simple text chunks ===\n")

    for chunk in client.stream_text("Write a haiku about Python programming."):
        print(chunk, end="", flush=True)

    print("\n")

    # ------------------------------------------------------------------
    # Approach 2: stream() — full event objects with final summary
    # ------------------------------------------------------------------
    print("=== stream() — full events with metadata ===\n")

    gen = client.stream("Explain recursion in exactly three sentences.")
    response = None

    try:
        while True:
            event = next(gen)
            # Print text as it arrives
            if event.message:
                print(event.message, end="", flush=True)
    except StopIteration as stop:
        # The generator's return value is the final ClaudeResponse
        response = stop.value

    print("\n")

    if response:
        print(f"Session : {response.session_id}")
        print(f"Cost    : ${response.cost_usd}")
        print(f"Duration: {response.duration_ms}ms")
        print(f"Turns   : {response.num_turns}")


if __name__ == "__main__":
    main()
