"""Data models for claude_code_cli."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Output format for print mode."""
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class InputFormat(str, Enum):
    """Input format for print mode."""
    TEXT = "text"
    STREAM_JSON = "stream-json"


class PermissionMode(str, Enum):
    """Permission modes for Claude Code."""
    DEFAULT = "default"
    PLAN = "plan"


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Defines a custom subagent for the --agents flag.

    Example:
        reviewer = Agent(
            name="reviewer",
            description="Reviews code for quality",
            prompt="You are a senior code reviewer.",
            tools=["Read", "Grep", "Glob", "Bash"],
            model="sonnet",
        )
    """

    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the agent definition dict (without the name key)."""
        d: dict[str, Any] = {
            "description": self.description,
            "prompt": self.prompt,
        }
        if self.tools is not None:
            d["tools"] = self.tools
        if self.model is not None:
            d["model"] = self.model
        return d


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolSet:
    """Convenience wrapper for specifying tools.

    Examples:
        ToolSet.default()           -> all default tools
        ToolSet.none()              -> no tools
        ToolSet(["Bash", "Read"])   -> specific tools
        ToolSet(["Bash(git log:*)", "Read"])  -> tools with patterns
    """

    tools: list[str]

    @classmethod
    def default(cls) -> "ToolSet":
        return cls(["default"])

    @classmethod
    def none(cls) -> "ToolSet":
        return cls([""])

    def to_cli_args(self) -> list[str]:
        """Return the CLI arguments for --tools."""
        return ["--tools", ",".join(self.tools)]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from stream-json output.

    Attributes:
        raw:  The original parsed JSON dict from one line of stream output.
        type: The event type if present (e.g. 'assistant', 'result', 'system').
    """

    raw: dict[str, Any]

    @property
    def type(self) -> str:
        return self.raw.get("type", "")

    @property
    def subtype(self) -> str:
        return self.raw.get("subtype", "")

    @property
    def message(self) -> str:
        """Best-effort extraction of the text content from this event."""
        # Handle the common content-block structure
        if "content_block" in self.raw:
            block = self.raw["content_block"]
            if isinstance(block, dict):
                return block.get("text", "")
        # Top-level message field
        if "message" in self.raw:
            return str(self.raw["message"])
        # Result type often has a "result" field
        if "result" in self.raw:
            return str(self.raw["result"])
        return ""

    @property
    def session_id(self) -> str | None:
        return self.raw.get("session_id")

    @property
    def cost_usd(self) -> float | None:
        return self.raw.get("cost_usd")

    @property
    def duration_ms(self) -> float | None:
        return self.raw.get("duration_ms")

    @property
    def num_turns(self) -> int | None:
        return self.raw.get("num_turns")

    @property
    def is_error(self) -> bool:
        return self.raw.get("is_error", False)

    def __repr__(self) -> str:
        return f"StreamEvent(type={self.type!r}, message={self.message[:80]!r})"


@dataclass
class ClaudeResponse:
    """Parsed response from a Claude Code CLI invocation.

    Attributes:
        text:        The plain-text output (available when output_format is TEXT).
        json:        The parsed JSON response (available when output_format is JSON).
        events:      List of StreamEvent objects (available when output_format is STREAM_JSON).
        returncode:  The process return code.
        stderr:      Any stderr output from the process.
        session_id:  The session ID if available (extracted from JSON/stream output).
        cost_usd:    Total cost in USD if available.
        duration_ms: Total duration in ms if available.
        num_turns:   Number of agentic turns if available.
        is_error:    Whether the response indicates an error.
    """

    text: str = ""
    json: dict[str, Any] | list[Any] | None = None
    events: list[StreamEvent] = field(default_factory=list)
    returncode: int = 0
    stderr: str = ""
    session_id: str | None = None
    cost_usd: float | None = None
    duration_ms: float | None = None
    num_turns: int | None = None
    is_error: bool = False

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, stdout: str, returncode: int = 0, stderr: str = "") -> "ClaudeResponse":
        return cls(text=stdout.strip(), returncode=returncode, stderr=stderr)

    @classmethod
    def from_json(cls, stdout: str, returncode: int = 0, stderr: str = "") -> "ClaudeResponse":
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return cls(text=stdout.strip(), returncode=returncode, stderr=stderr)

        resp = cls(
            json=data,
            returncode=returncode,
            stderr=stderr,
        )

        # Extract known top-level fields from the JSON envelope
        if isinstance(data, dict):
            resp.session_id = data.get("session_id")
            resp.cost_usd = data.get("cost_usd")
            resp.duration_ms = data.get("duration_ms")
            resp.num_turns = data.get("num_turns")
            resp.is_error = data.get("is_error", False)
            # Derive a text representation from the result field
            result = data.get("result", "")
            if isinstance(result, str):
                resp.text = result
            else:
                resp.text = json.dumps(result)
        return resp

    @classmethod
    def from_stream(
        cls, events: list[StreamEvent], returncode: int = 0, stderr: str = ""
    ) -> "ClaudeResponse":
        resp = cls(events=events, returncode=returncode, stderr=stderr)
        # The final event in a stream is typically the result summary
        for evt in reversed(events):
            if evt.type == "result":
                resp.session_id = evt.session_id
                resp.cost_usd = evt.cost_usd
                resp.duration_ms = evt.duration_ms
                resp.num_turns = evt.num_turns
                resp.is_error = evt.is_error
                resp.text = evt.raw.get("result", "")
                resp.json = evt.raw
                break

        # If no result event, concatenate assistant messages
        if not resp.text:
            parts = [e.message for e in events if e.message]
            resp.text = "\n".join(parts)
        return resp

    def __repr__(self) -> str:
        preview = self.text[:120] + "…" if len(self.text) > 120 else self.text
        return f"ClaudeResponse(text={preview!r}, returncode={self.returncode})"


# ---------------------------------------------------------------------------
# Typed model response
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse(Generic[T]):
    """Response wrapper that carries a deserialized Python object.

    Returned by :meth:`ClaudeCode.query_model`.  Provides both the typed
    ``data`` attribute and the underlying :class:`ClaudeResponse` for access
    to metadata like ``session_id`` and ``cost_usd``.

    Attributes:
        data:     The deserialized instance of the target class ``T``.
        raw:      The underlying :class:`ClaudeResponse`.
        schema:   The JSON Schema dict that was sent to the CLI.
    """

    data: T
    raw: ClaudeResponse
    schema: dict[str, Any] = field(default_factory=dict)

    # Convenience proxies so callers don't always need .raw
    @property
    def session_id(self) -> str | None:
        return self.raw.session_id

    @property
    def cost_usd(self) -> float | None:
        return self.raw.cost_usd

    @property
    def duration_ms(self) -> float | None:
        return self.raw.duration_ms

    @property
    def num_turns(self) -> int | None:
        return self.raw.num_turns

    def __repr__(self) -> str:
        return f"ModelResponse(data={self.data!r}, cost_usd={self.cost_usd})"
