"""
claude_code_cli — A Python wrapper for the Claude Code CLI.

Two styles of usage:

**Quick one-liner** (zero config)::

    import claude_code_cli as claude
    print(claude.ask("What is 2+2?").text)

**Configured client** (flat kwargs)::

    import claude_code_cli as claude
    client = claude.ClaudeCode(model="sonnet", max_turns=5)
    response = client.query("Explain this project")
    print(response.text)

**Explicit options** (full control)::

    opts = claude.ClaudeOptions(model="opus", tools=["Read", "Bash"])
    response = client.query("Audit this code", options=opts)

**Structured output** (typed Python objects)::

    from dataclasses import dataclass

    @dataclass
    class Review:
        summary: str
        issues: list[str]
        severity: str

    result = client.query_model("Review auth.py", Review)
    print(result.data.summary)   # typed Review instance
    print(result.cost_usd)       # API cost
"""

from claude_code_cli.client import ClaudeCode, ask
from claude_code_cli.options import (
    ClaudeOptions,
    Agent,
    ToolSet,
    OutputFormat,
    InputFormat,
    PermissionMode,
)
from claude_code_cli.models import (
    ClaudeResponse,
    ModelResponse,
    StreamEvent,
)
from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeProcessError,
    ClaudeResponseParseError,
    ClaudeOptionError,
)
from claude_code_cli.schema import (
    extract_schema,
    schema,
    FieldMeta,
)

__version__ = "0.2.0"
__all__ = [
    # Client
    "ClaudeCode",
    "ask",
    # Options
    "ClaudeOptions",
    "Agent",
    "ToolSet",
    "OutputFormat",
    "InputFormat",
    "PermissionMode",
    # Responses
    "ClaudeResponse",
    "ModelResponse",
    "StreamEvent",
    # Schema utilities
    "extract_schema",
    "schema",
    "FieldMeta",
    # Errors
    "ClaudeCodeError",
    "ClaudeNotFoundError",
    "ClaudeTimeoutError",
    "ClaudeProcessError",
    "ClaudeResponseParseError",
    "ClaudeOptionError",
]
