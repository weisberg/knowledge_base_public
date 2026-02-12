"""Custom exceptions for claude_code_cli."""


class ClaudeCodeError(Exception):
    """Base exception for all claude_code_cli errors."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


class ClaudeNotFoundError(ClaudeCodeError):
    """Raised when the 'claude' CLI binary is not found on PATH."""

    def __init__(self):
        super().__init__(
            "The 'claude' CLI was not found on PATH. "
            "Install Claude Code: https://code.claude.com/docs/en/quickstart"
        )


class ClaudeTimeoutError(ClaudeCodeError):
    """Raised when a CLI invocation exceeds the configured timeout."""

    def __init__(self, timeout: float):
        super().__init__(f"Claude Code CLI timed out after {timeout}s")


class ClaudeProcessError(ClaudeCodeError):
    """Raised when the CLI process exits with a non-zero return code."""

    def __init__(self, returncode: int, stderr: str, cmd: list[str] | None = None):
        cmd_str = " ".join(cmd) if cmd else "(unknown)"
        super().__init__(
            f"Claude Code CLI exited with code {returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Stderr: {stderr}",
            returncode=returncode,
            stderr=stderr,
        )
        self.cmd = cmd
