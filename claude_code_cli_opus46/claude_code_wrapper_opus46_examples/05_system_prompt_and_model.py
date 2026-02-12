#!/usr/bin/env python3
"""
Example 05 — System Prompts & Model Selection
===============================================
Customize Claude's persona via system prompts and choose which model to use.
"""

import claude_code_cli as claude


def main():
    # ------------------------------------------------------------------
    # Replace the entire system prompt — Claude becomes a specialist
    # ------------------------------------------------------------------
    print("=== Custom system prompt (replace) ===\n")

    specialist = claude.ClaudeCode(
        system_prompt=(
            "You are a concise Unix command-line expert. "
            "Reply only with the command and a one-line explanation. "
            "No markdown, no code fences."
        ),
    )

    answer = specialist.run("How do I find all .py files modified in the last 7 days?")
    print(f"Expert says: {answer}\n")

    # ------------------------------------------------------------------
    # Append to the default prompt — safer, keeps built-in capabilities
    # ------------------------------------------------------------------
    print("=== Append to default system prompt ===\n")

    polite = claude.ClaudeCode(
        append_system_prompt="Always end your response with 'Hope that helps!'",
    )

    answer = polite.run("What does the ls command do?")
    print(f"Polite Claude: {answer}\n")

    # ------------------------------------------------------------------
    # Model selection — alias vs full name
    # ------------------------------------------------------------------
    print("=== Model selection ===\n")

    # Using an alias
    fast = claude.ClaudeCode(model="sonnet")
    answer = fast.run("What is 17 * 23? Just the number.")
    print(f"Sonnet says: {answer}")

    # Per-call override
    client = claude.ClaudeCode(model="sonnet")
    answer = client.run(
        "What is the meaning of life? One sentence.",
        model="opus",  # override for this call only
    )
    print(f"Opus says: {answer}\n")

    # ------------------------------------------------------------------
    # Model with fallback
    # ------------------------------------------------------------------
    print("=== Model with fallback ===\n")

    resilient = claude.ClaudeCode(model="opus", fallback_model="sonnet")
    answer = resilient.run("Say hello in Japanese. One word only.")
    print(f"Resilient says: {answer}")


if __name__ == "__main__":
    main()
