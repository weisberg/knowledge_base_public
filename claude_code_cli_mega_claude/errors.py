"""Custom exceptions for claude_code_cli.

Hierarchy
---------
ClaudeCodeError (base)
├── ClaudeNotFoundError        — ``claude`` binary not on PATH
├── ClaudeTimeoutError         — subprocess exceeded timeout
├── ClaudeProcessError         — non-zero exit code
├── ClaudeResponseParseError   — JSON / structured-output parsing failure
└── ClaudeOptionError          — invalid flag combination (ValueError subclass)
"""

from __future__ import annotations

from typing import Any, Sequence


class ClaudeCodeError(Exception):
    """Base exception for all claude_code_cli errors."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


class ClaudeNotFoundError(ClaudeCodeError):
    """Raised when the ``claude`` CLI binary is not found on PATH."""

    def __init__(self, cli_path: str = "claude"):
        super().__init__(
            f"Could not find Claude Code CLI executable '{cli_path}'. "
            "Install it: https://code.claude.com/docs/en/quickstart"
        )
        self.cli_path = cli_path


class ClaudeTimeoutError(ClaudeCodeError):
    """Raised when a CLI invocation exceeds the configured timeout."""

    def __init__(self, timeout: float):
        super().__init__(f"Claude Code CLI timed out after {timeout}s")
        self.timeout = timeout


class ClaudeProcessError(ClaudeCodeError):
    """Raised when the CLI process exits with a non-zero return code."""

    def __init__(
        self,
        returncode: int,
        stderr: str,
        cmd: list[str] | None = None,
        stdout: str = "",
    ):
        cmd_str = " ".join(cmd) if cmd else "(unknown)"
        super().__init__(
            f"`claude` exited with code {returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Stderr: {stderr}",
            returncode=returncode,
            stderr=stderr,
        )
        self.cmd = cmd or []
        self.stdout = stdout


class ClaudeResponseParseError(ClaudeCodeError):
    """Raised when output parsing fails (invalid JSON, schema mismatch, etc.)."""

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


class ClaudeOptionError(ClaudeCodeError, ValueError):
    """Raised for invalid or conflicting CLI flag combinations.

    Also a :class:`ValueError` so it can be caught by generic validation handlers.
    """

    def __init__(self, message: str):
        # Bypass ClaudeCodeError's keyword args
        Exception.__init__(self, message)
        self.returncode = None
        self.stderr = ""
