#!/usr/bin/env python3
"""
Example 01 — Hello World
=========================
The simplest possible test: ask Claude Code a question and print the answer.
Verifies that the CLI is installed, reachable, and can respond.
"""

import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ---- Preflight: is the CLI even installed? ----
    if not client.is_available():
        print("ERROR: 'claude' CLI not found on PATH.")
        print("Install it: https://code.claude.com/docs/en/quickstart")
        return

    print(f"CLI version: {client.version()}")
    print()

    # ---- Ask a simple factual question ----
    print("Asking: What is the capital of France?")
    print("-" * 40)

    response = client.query("What is the capital of France? Reply in one sentence.")

    print(f"Answer : {response.text}")
    print(f"Cost   : ${response.cost_usd}")
    print(f"Turns  : {response.num_turns}")
    print(f"Session: {response.session_id}")


if __name__ == "__main__":
    main()
