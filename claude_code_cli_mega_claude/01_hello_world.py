#!/usr/bin/env python3
"""
Example 01 — Hello World
=========================
The simplest possible test: can Claude Code answer a question?
Shows three ways to ask, from zero-config to fully configured.
"""

import claude_code_cli as claude


def main():
    # ---- Zero-config one-liner (module-level ask) ----
    print("=== claude.ask() — zero config ===")
    response = claude.ask("What is the capital of France? Reply in one sentence.")
    print(f"Answer: {response.text}")
    print()

    # ---- Configured client ----
    print("=== ClaudeCode client ===")
    client = claude.ClaudeCode()

    if not client.is_available():
        print("ERROR: 'claude' CLI not found on PATH.")
        return

    print(f"CLI version: {client.version()}")

    response = client.query("What is 7 * 8? Reply with just the number.")
    print(f"Answer : {response.text}")
    print(f"Cost   : ${response.cost_usd}")
    print(f"Turns  : {response.num_turns}")
    print(f"Session: {response.session_id}")
    print()

    # ---- run() shorthand — just the text ----
    print("=== run() shorthand ===")
    answer = client.run("Name the three primary colors. One sentence only.")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
