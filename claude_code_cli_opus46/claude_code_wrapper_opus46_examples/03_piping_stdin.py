#!/usr/bin/env python3
"""
Example 03 — Piping Data via stdin
====================================
Equivalent of: cat file.py | claude -p "Explain this code"

Demonstrates passing file contents (or any string) to Claude through stdin.
"""

import claude_code_cli as claude


# A small Python snippet to analyze
SAMPLE_CODE = '''\
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''


def main():
    client = claude.ClaudeCode()

    # ---- Pipe code via stdin ----
    print("=== Piping code to Claude ===")
    print(f"Input:\n{SAMPLE_CODE}")
    print("-" * 40)

    response = client.query(
        "Explain what this function does in 2-3 sentences. "
        "Mention its time and space complexity.",
        stdin=SAMPLE_CODE,
    )
    print(f"Explanation:\n{response.text}\n")

    # ---- Pipe CSV data ----
    csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,Chicago\n"

    print("=== Piping CSV data ===")
    response = client.query(
        "How many people are in this CSV? What is the average age? Reply concisely.",
        stdin=csv_data,
    )
    print(f"Analysis: {response.text}\n")

    # ---- Pipe from an actual file ----
    print("=== Piping this script's own source ===")
    with open(__file__) as f:
        source = f.read()

    response = client.query(
        "How many functions are defined in this Python file? List their names.",
        stdin=source,
    )
    print(f"Functions found: {response.text}")


if __name__ == "__main__":
    main()
