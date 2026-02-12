#!/usr/bin/env python3
"""
Example 12 — Custom Subagents
===============================
Define specialized subagents that Claude can delegate to during a task.
Each agent has its own system prompt, tool access, and optional model.
"""

import claude_code_cli as claude


def main():
    # ------------------------------------------------------------------
    # Define specialized agents
    # ------------------------------------------------------------------

    reviewer = claude.Agent(
        name="code-reviewer",
        description="Expert code reviewer. Invoked after code changes to check quality.",
        prompt=(
            "You are a senior code reviewer. Focus on:\n"
            "- Code correctness and edge cases\n"
            "- Security vulnerabilities\n"
            "- Performance considerations\n"
            "- Readability and maintainability\n"
            "Be specific about line numbers and provide fix suggestions."
        ),
        tools=["Read", "Grep", "Glob", "Bash"],
        model="sonnet",
    )

    test_writer = claude.Agent(
        name="test-writer",
        description="Test writing specialist. Creates thorough test suites.",
        prompt=(
            "You are a testing expert. Write comprehensive tests using pytest. "
            "Cover: happy paths, edge cases, error conditions, and boundary values. "
            "Use descriptive test names and docstrings."
        ),
        tools=["Read", "Edit", "Bash"],
        model="sonnet",
    )

    documenter = claude.Agent(
        name="documenter",
        description="Documentation specialist. Writes clear docstrings and README content.",
        prompt=(
            "You are a technical writer. Write clear, concise documentation. "
            "Use Google-style docstrings. Include usage examples."
        ),
        tools=["Read", "Edit"],
    )

    # ------------------------------------------------------------------
    # Use agents in a query
    # ------------------------------------------------------------------

    client = claude.ClaudeCode(
        agents=[reviewer, test_writer, documenter],
        max_turns=10,
    )

    print("=== Multi-agent task ===\n")

    # Claude will delegate to subagents as needed
    response = client.query(
        "Look at the Python files in this directory. "
        "Have the code-reviewer check them, then have the documenter "
        "suggest docstring improvements. Summarize the findings.",
        cwd=".",
    )

    print(f"Result:\n{response.text}")
    print(f"\nCost: ${response.cost_usd}")
    print(f"Turns: {response.num_turns}")

    # ------------------------------------------------------------------
    # Select a specific agent for a focused task
    # ------------------------------------------------------------------
    print("\n=== Single agent selection ===\n")

    focused = claude.ClaudeCode(agent="code-reviewer", agents=[reviewer])
    answer = focused.run(
        "Review the error handling in the current directory's Python files.",
        cwd=".",
    )
    print(f"Review:\n{answer}")


if __name__ == "__main__":
    main()
