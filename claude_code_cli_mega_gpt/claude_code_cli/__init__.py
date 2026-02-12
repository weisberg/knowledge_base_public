"""claude_code_cli — Python wrapper for the Claude Code CLI (`claude -p`).

Import style the user asked for:

    import claude_code_cli as claude

    client = claude.ClaudeCode()
    print(client.run("Explain this repo"))

This package focuses on print/headless usage (`-p`) and supports:
  - JSON output (`--output-format json`)
  - Streaming JSON output (`--output-format stream-json`)
  - Structured outputs via `--json-schema` and `query_model()`

Docs:
  - CLI reference: https://code.claude.com/docs/en/cli-reference
"""

from __future__ import annotations

from claude_code_cli.client import ClaudeCode, ClaudeOptions
from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeProcessError,
    ClaudeResponseParseError,
    ClaudeOptionError,
    ClaudeStructuredOutputError,
)
from claude_code_cli.models import (
    Agent,
    ClaudeResponse,
    InputFormat,
    ModelResponse,
    OutputFormat,
    PermissionMode,
    StreamEvent,
    TeammateMode,
    ToolSet,
)
from claude_code_cli.schema import (
    FieldMeta,
    extract_schema,
    schema,
    simplify_schema_for_claude,
)

# Backwards-friendly alias
Client = ClaudeCode


def ask(prompt: str, *, stdin: str | None = None, **kwargs):
    """One-shot helper: create a temporary client and return `ClaudeResponse`."""
    return ClaudeCode().query(prompt, stdin=stdin, **kwargs)


def run(prompt: str, *, stdin: str | None = None, **kwargs) -> str:
    """One-shot helper: create a temporary client and return only `.text`."""
    return ClaudeCode().run(prompt, stdin=stdin, **kwargs)
