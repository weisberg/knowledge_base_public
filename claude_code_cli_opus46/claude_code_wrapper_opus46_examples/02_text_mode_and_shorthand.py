#!/usr/bin/env python3
"""
Example 02 — Text Mode & Shorthand
====================================
Shows the difference between output formats and the `run()` one-liner.
"""

import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ---- run() returns just the text string ----
    print("=== run() shorthand ===")
    answer = client.run("Name the three primary colors. One sentence only.")
    print(answer)
    print()

    # ---- query() with TEXT format — no JSON parsing overhead ----
    print("=== query() with OutputFormat.TEXT ===")
    response = client.query(
        "What is 2 + 2? Reply with only the number.",
        output_format=claude.OutputFormat.TEXT,
    )
    print(f"Text: {response.text!r}")
    print(f"JSON: {response.json}")  # None — text mode doesn't produce JSON
    print()

    # ---- query() with default JSON format — richer metadata ----
    print("=== query() with OutputFormat.JSON (default) ===")
    response = client.query("What is the boiling point of water in Celsius? One sentence.")
    print(f"Text      : {response.text}")
    print(f"Session ID: {response.session_id}")
    print(f"Cost USD  : {response.cost_usd}")
    print(f"Duration  : {response.duration_ms}ms")
    print(f"Turns     : {response.num_turns}")
    print(f"Is error  : {response.is_error}")


if __name__ == "__main__":
    main()
