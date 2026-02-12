#!/usr/bin/env python3
"""
Example 11 — Tool & Permission Configuration
===============================================
Control which tools Claude can use, set permission modes, and
configure safety boundaries for automated pipelines.
"""

import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ------------------------------------------------------------------
    # 1. Restrict to read-only tools (safe for analysis)
    # ------------------------------------------------------------------
    print("=== Read-only tools ===\n")

    reader = claude.ClaudeCode(
        tools=["Read", "Grep", "Glob"],  # no Bash, no Edit
    )
    # This will only let Claude read files, not modify anything
    answer = reader.run(
        "What programming language files exist in the current directory? "
        "Just list the filenames.",
        cwd=".",
    )
    print(f"Files: {answer}\n")

    # ------------------------------------------------------------------
    # 2. ToolSet helpers
    # ------------------------------------------------------------------
    print("=== ToolSet helpers ===\n")

    # All default tools
    all_tools = claude.ClaudeCode(tools=claude.ToolSet.default())
    print(f"Default tools CLI args: {claude.ToolSet.default().to_cli_args()}")

    # No tools at all — pure Q&A, no file access
    no_tools = claude.ClaudeCode(tools=claude.ToolSet.none())
    answer = no_tools.run("What is the Pythagorean theorem? One sentence.")
    print(f"No tools (pure Q&A): {answer}")
    print()

    # Specific tools with patterns
    git_only = claude.ToolSet(["Bash(git log:*)", "Bash(git diff:*)", "Read"])
    print(f"Git-only tools CLI args: {git_only.to_cli_args()}")
    print()

    # ------------------------------------------------------------------
    # 3. Allowed / disallowed tools
    # ------------------------------------------------------------------
    print("=== Allowed / Disallowed tools ===\n")

    # Allow git commands without prompting, but disallow Edit
    safe_client = claude.ClaudeCode(
        allowed_tools=["Bash(git log:*)", "Bash(git diff:*)", "Bash(git status:*)"],
        disallowed_tools=["Edit"],
    )
    print("Configured: git commands auto-approved, Edit blocked")
    print()

    # ------------------------------------------------------------------
    # 4. Permission modes
    # ------------------------------------------------------------------
    print("=== Permission modes ===\n")

    # Plan mode — Claude describes what it would do, but doesn't execute
    planner = claude.ClaudeCode(permission_mode=claude.PermissionMode.PLAN)
    answer = planner.run("How would you add type hints to the Python files in this directory?")
    print(f"Plan: {answer[:200]}...")
    print()

    # ------------------------------------------------------------------
    # 5. Skip permissions (for CI/CD — use with extreme caution!)
    # ------------------------------------------------------------------
    print("=== Dangerous: skip permissions ===\n")

    # ⚠️  This lets Claude run ANY command without asking
    # Only use in sandboxed environments (Docker, CI runners)
    print("ClaudeCode(dangerously_skip_permissions=True)")
    print("  → Would pass --dangerously-skip-permissions")
    print("  → NOT running this in the example for safety")
    print()

    # ------------------------------------------------------------------
    # 6. Max turns — prevent runaway agentic loops
    # ------------------------------------------------------------------
    print("=== Max turns ===\n")

    bounded = claude.ClaudeCode(max_turns=3)
    answer = bounded.run("What is 2+2? Reply with just the number.")
    print(f"Bounded (max 3 turns): {answer}")


if __name__ == "__main__":
    main()
