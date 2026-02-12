#!/usr/bin/env python3
"""
Example 09 — Error Handling
=============================
Demonstrates how to catch and handle the various exceptions that can occur.
"""

import claude_code_cli as claude
from claude_code_cli import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)


def main():
    # ------------------------------------------------------------------
    # 1. Preflight check
    # ------------------------------------------------------------------
    print("=== Preflight check ===\n")

    client = claude.ClaudeCode()
    if client.is_available():
        print(f"Claude CLI is available: {client.version()}")
    else:
        print("Claude CLI is NOT available")
        return

    print()

    # ------------------------------------------------------------------
    # 2. Catch specific exceptions
    # ------------------------------------------------------------------
    print("=== Exception hierarchy demo ===\n")

    # Timeout — useful for long-running agentic tasks
    try:
        # Very short timeout to demonstrate the error path
        response = client.query(
            "Count from 1 to 1000, listing each number on its own line.",
            timeout=0.001,  # almost certainly too short
        )
        print(f"Surprisingly fast: {response.text[:50]}...")
    except ClaudeTimeoutError as e:
        print(f"[Caught ClaudeTimeoutError] {e}")
    except ClaudeProcessError as e:
        print(f"[Caught ClaudeProcessError] exit code {e.returncode}")
    except ClaudeCodeError as e:
        print(f"[Caught ClaudeCodeError] {e}")

    print()

    # ------------------------------------------------------------------
    # 3. Graceful pattern: try/except with fallback
    # ------------------------------------------------------------------
    print("=== Graceful fallback pattern ===\n")

    def ask_claude(prompt: str, timeout: float = 60.0) -> str:
        """Ask Claude with graceful error handling."""
        try:
            return client.run(prompt, timeout=timeout)
        except ClaudeTimeoutError:
            return f"[Timed out after {timeout}s — try increasing the timeout]"
        except ClaudeProcessError as e:
            return f"[CLI error (code {e.returncode}): {e.stderr[:200]}]"
        except ClaudeNotFoundError:
            return "[Claude CLI not installed]"
        except ClaudeCodeError as e:
            return f"[Unexpected error: {e}]"

    answer = ask_claude("What is 1 + 1? Just the number.")
    print(f"Answer: {answer}")

    print()

    # ------------------------------------------------------------------
    # 4. Inspecting error details
    # ------------------------------------------------------------------
    print("=== Error detail inspection ===\n")

    try:
        # Use a bogus binary path to force a NotFoundError
        bad_client = claude.ClaudeCode(claude_binary="/nonexistent/claude")
        bad_client.query("Hello")
    except ClaudeNotFoundError as e:
        print(f"Type    : {type(e).__name__}")
        print(f"Message : {e}")
        print(f"Code    : {e.returncode}")  # None for NotFound
        print(f"Stderr  : {e.stderr!r}")    # empty for NotFound

    print()

    # ------------------------------------------------------------------
    # 5. Max turns to prevent runaway agents
    # ------------------------------------------------------------------
    print("=== Max turns safety net ===\n")

    bounded = claude.ClaudeCode(max_turns=2)
    try:
        response = bounded.query(
            "Write a comprehensive 20-file Python project with tests. "
            "Create all files.",
        )
        print(f"Completed in {response.num_turns} turns")
        print(f"Response: {response.text[:100]}...")
    except ClaudeProcessError as e:
        print(f"Stopped (code {e.returncode}): {e.stderr[:200]}")
    except ClaudeCodeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
