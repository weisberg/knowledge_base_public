#!/usr/bin/env python3
"""
Example 14 — Advanced: ClaudeOptions, build_command, env, extra_args
=====================================================================
Showcases the power-user features adopted from the ClaudeOptions pattern:

  - Explicit ClaudeOptions objects (reusable, inspectable, validatable)
  - build_command() to see the exact argv without executing
  - extra_args escape hatch for future/unknown flags
  - env parameter for custom environment variables
  - Dual interface: flat kwargs AND explicit options, freely mixed
  - Option validation with ClaudeOptionError
"""

import json
import claude_code_cli as claude


def main():
    # ------------------------------------------------------------------
    # 1. ClaudeOptions — explicit, reusable configuration
    # ------------------------------------------------------------------
    print("=== ClaudeOptions — explicit config ===\n")

    # Build a reusable options object
    review_opts = claude.ClaudeOptions(
        model="sonnet",
        max_turns=10,
        tools=["Read", "Grep", "Glob", "Bash"],
        append_system_prompt="Focus on security issues. Be concise.",
        verbose=True,
    )

    # Validate before use (catches conflicts early)
    review_opts.validate()
    print(f"Options valid. Args: {review_opts.to_args()}")
    print()

    # ------------------------------------------------------------------
    # 2. build_command() — inspect without executing
    # ------------------------------------------------------------------
    print("=== build_command() — see the argv ===\n")

    client = claude.ClaudeCode(model="sonnet")

    cmd = client.build_command(
        "Review this codebase",
        options=review_opts,
    )
    print(f"Would run: {' '.join(cmd)}")
    print()

    # Per-call kwargs also work with build_command
    cmd2 = client.build_command("Explain this", model="opus", max_turns=3)
    print(f"With kwargs: {' '.join(cmd2)}")
    print()

    # ------------------------------------------------------------------
    # 3. Dual interface — kwargs + options, freely mixed
    # ------------------------------------------------------------------
    print("=== Dual interface ===\n")

    # Instance-level defaults via kwargs
    client = claude.ClaudeCode(model="sonnet", verbose=True)

    # Per-call override via kwargs
    cmd = client.build_command("test", model="opus")
    print(f"kwargs override:  {cmd}")

    # Per-call override via explicit options (highest priority)
    explicit = claude.ClaudeOptions(model="haiku", max_turns=1)
    cmd = client.build_command("test", model="opus", options=explicit)
    print(f"explicit options: {cmd}")
    # explicit options win → haiku, not opus
    print()

    # ------------------------------------------------------------------
    # 4. extra_args — escape hatch for future CLI flags
    # ------------------------------------------------------------------
    print("=== extra_args — future-proof ===\n")

    # If Claude Code adds a new flag tomorrow, you don't need a library update
    future_opts = claude.ClaudeOptions(
        model="sonnet",
        extra_args=["--some-future-flag", "value", "--another-flag"],
    )
    cmd = client.build_command("test", options=future_opts)
    print(f"With extra_args: {cmd}")
    assert "--some-future-flag" in cmd
    print()

    # ------------------------------------------------------------------
    # 5. env — custom environment variables
    # ------------------------------------------------------------------
    print("=== env — environment variables ===\n")

    # Pass API keys or config via environment
    client_with_env = claude.ClaudeCode(
        env={"ANTHROPIC_API_KEY": "sk-ant-..."},
    )
    print(f"Client with custom env: {client_with_env}")
    print()

    # ------------------------------------------------------------------
    # 6. Option validation — catch mistakes early
    # ------------------------------------------------------------------
    print("=== Option validation ===\n")

    # Mutual exclusion: system_prompt + system_prompt_file
    try:
        bad = claude.ClaudeOptions(
            system_prompt="Custom prompt",
            system_prompt_file="prompt.txt",
        )
        bad.validate()
    except claude.ClaudeOptionError as e:
        print(f"Caught: {e}")

    # fork_session without continue/resume
    try:
        bad2 = claude.ClaudeOptions(fork_session=True)
        bad2.validate()
    except claude.ClaudeOptionError as e:
        print(f"Caught: {e}")

    # Unknown kwargs
    try:
        client.build_command("test", nonexistent_flag="oops")
    except claude.ClaudeOptionError as e:
        print(f"Caught: {e}")

    print()

    # ------------------------------------------------------------------
    # 7. Options merge — layer configs
    # ------------------------------------------------------------------
    print("=== Options merge ===\n")

    base = claude.ClaudeOptions(
        model="sonnet",
        max_turns=10,
        verbose=True,
        tools=["Read", "Bash"],
    )

    security_layer = claude.ClaudeOptions(
        append_system_prompt="Flag any security vulnerabilities.",
        max_budget_usd=1.00,
    )

    combined = base.merge(security_layer)
    print(f"Base model:     {base.model}")
    print(f"Merged model:   {combined.model} (retained from base)")
    print(f"Merged budget:  ${combined.max_budget_usd} (from overlay)")
    print(f"Merged verbose: {combined.verbose} (retained)")
    print(f"Merged prompt:  {combined.append_system_prompt!r}")
    print()

    # ------------------------------------------------------------------
    # 8. max_budget_usd — cost cap
    # ------------------------------------------------------------------
    print("=== max_budget_usd ===\n")

    budget_opts = claude.ClaudeOptions(max_budget_usd=0.50)
    cmd = client.build_command("Expensive task", options=budget_opts)
    print(f"With budget cap: {cmd}")
    assert "--max-budget-usd" in cmd and "0.50" in cmd
    print()

    # ------------------------------------------------------------------
    # 9. PermissionMode — all six modes
    # ------------------------------------------------------------------
    print("=== PermissionMode — complete set ===\n")
    for mode in claude.PermissionMode:
        print(f"  {mode.name:22s} = {mode.value}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
