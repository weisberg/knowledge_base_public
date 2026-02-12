"""Custom exceptions for claude_code_cli.

This package wraps the Claude Code CLI (`claude`) in print/pipe mode (`claude -p`)
and provides higher-level helpers for JSON/streaming output and structured outputs.

Docs:
- CLI flags reference: https://code.claude.com/docs/en/cli-reference
- Headless / Agent SDK usage: https://code.claude.com/docs/en/headless
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


class ClaudeCodeError(Exception):
    """Base exception for all claude_code_cli errors."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        cmd: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.cmd = list(cmd) if cmd is not None else None


class ClaudeNotFoundError(ClaudeCodeError):
    """Raised when the `claude` CLI binary is not found on PATH."""

    def __init__(self, cli_path: str = "claude") -> None:
        super().__init__(
            f"Could not find Claude Code CLI executable '{cli_path}'. "
            "Install Claude Code: https://code.claude.com/docs/en/quickstart"
        )
        self.cli_path = cli_path


class ClaudeTimeoutError(ClaudeCodeError):
    """Raised when a CLI invocation exceeds the configured timeout."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"Claude Code CLI timed out after {timeout_s}s")
        self.timeout_s = timeout_s


class ClaudeProcessError(ClaudeCodeError):
    """Raised when the CLI process exits with a non-zero return code."""

    def __init__(
        self,
        *,
        returncode: int,
        stderr: str,
        stdout: str = "",
        cmd: Sequence[str] | None = None,
    ) -> None:
        cmd_str = " ".join(cmd) if cmd else "(unknown)"
        super().__init__(
            "Claude Code CLI exited with a non-zero status.\n"
            f"Exit code: {returncode}\n"
            f"Command: {cmd_str}\n"
            f"Stderr: {stderr}",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            cmd=cmd,
        )


class ClaudeResponseParseError(ClaudeCodeError):
    """Raised when output parsing fails (e.g., invalid JSON)."""


class ClaudeOptionError(ValueError):
    """Raised for invalid/unsupported option combinations (esp. in `-p` mode)."""


class ClaudeStructuredOutputError(ClaudeCodeError):
    """Raised when structured output is missing or cannot be parsed/validated."""
