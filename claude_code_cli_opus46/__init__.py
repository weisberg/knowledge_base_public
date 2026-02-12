"""
claude_code_cli - A Python wrapper for the Claude Code CLI.

Usage:
    import claude_code_cli as claude

    # Basic query
    client = claude.ClaudeCode()
    response = client.query("Explain this project")
    print(response.text)

    # Structured output with a Python class
    from dataclasses import dataclass

    @dataclass
    class CodeReview:
        summary: str
        issues: list[str]
        severity: str

    result = client.query_model("Review auth.py for security issues", CodeReview)
    print(result.data.summary)   # typed CodeReview instance
    print(result.data.issues)
    print(result.cost_usd)       # API cost

    # With Pydantic
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        complexity: int
        suggestions: list[str]

    result = client.query_model("Analyze main.py", Analysis)
    print(result.data.summary)   # typed Analysis instance
"""

from claude_code_cli.client import ClaudeCode
from claude_code_cli.models import (
    ClaudeResponse,
    ModelResponse,
    StreamEvent,
    Agent,
    ToolSet,
    OutputFormat,
    InputFormat,
    PermissionMode,
)
from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeProcessError,
)
from claude_code_cli.schema import (
    extract_schema,
    schema,
    FieldMeta,
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "ClaudeCode",
    # Response models
    "ClaudeResponse",
    "ModelResponse",
    "StreamEvent",
    # Configuration
    "Agent",
    "ToolSet",
    "OutputFormat",
    "InputFormat",
    "PermissionMode",
    # Schema utilities
    "extract_schema",
    "schema",
    "FieldMeta",
    # Errors
    "ClaudeCodeError",
    "ClaudeNotFoundError",
    "ClaudeTimeoutError",
    "ClaudeProcessError",
]
