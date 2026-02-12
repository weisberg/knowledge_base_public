"""Data models and enums for claude_code_cli."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Iterable, Iterator, Mapping, Sequence, TypeVar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Output format for print mode (`claude -p`)."""
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class InputFormat(str, Enum):
    """Input format for print mode (`claude -p`)."""
    TEXT = "text"
    STREAM_JSON = "stream-json"


class PermissionMode(str, Enum):
    """Permission modes for Claude Code.

    See: https://code.claude.com/docs/en/permissions
    """
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    DELEGATE = "delegate"
    DONT_ASK = "dontAsk"
    BYPASS_PERMISSIONS = "bypassPermissions"


class TeammateMode(str, Enum):
    """Teammate display mode for agent teams (`--teammate-mode`)."""
    AUTO = "auto"
    IN_PROCESS = "in-process"
    TMUX = "tmux"


# ---------------------------------------------------------------------------
# Agent definition (for --agents)
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Defines a custom subagent for the --agents flag.

    CLI reference fields:
      - description (required)
      - prompt (required)
      - tools (optional)
      - disallowedTools (optional)
      - model (optional)
      - skills (optional)
      - mcpServers (optional)
      - maxTurns (optional)

    See: https://code.claude.com/docs/en/cli-reference
    """

    name: str
    description: str
    prompt: str

    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    model: str | None = None
    skills: list[str] | None = None
    mcp_servers: list[Any] | None = None
    max_turns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the agent definition dict (without the name key)."""
        d: dict[str, Any] = {
            "description": self.description,
            "prompt": self.prompt,
        }
        if self.tools is not None:
            d["tools"] = self.tools
        if self.disallowed_tools is not None:
            d["disallowedTools"] = self.disallowed_tools
        if self.model is not None:
            d["model"] = self.model
        if self.skills is not None:
            d["skills"] = self.skills
        if self.mcp_servers is not None:
            d["mcpServers"] = self.mcp_servers
        if self.max_turns is not None:
            d["maxTurns"] = self.max_turns
        return d


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolSet:
    """Convenience wrapper for specifying tools.

    Examples:
        ToolSet.default()           -> all default tools ("default")
        ToolSet.none()              -> no tools (empty string)
        ToolSet(["Bash", "Read"])   -> specific tools
        ToolSet(["Bash(git log *)", "Read"])  -> tools with patterns

    CLI flag: --tools "Bash,Edit,Read" (or "" / "default")
    """

    tools: list[str]

    @classmethod
    def default(cls) -> "ToolSet":
        return cls(["default"])

    @classmethod
    def none(cls) -> "ToolSet":
        return cls([""])

    def to_cli_args(self) -> list[str]:
        return ["--tools", ",".join(self.tools)]


# ---------------------------------------------------------------------------
# Streaming event model
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from `--output-format stream-json`.

    Claude Code stream-json lines are JSON objects. The exact structure can vary
    across versions (and between the CLI and SDK), so this class is intentionally
    permissive and provides best-effort accessors.
    """

    raw: dict[str, Any]

    @property
    def type(self) -> str:
        t = self.raw.get("type")
        if isinstance(t, str):
            return t
        # Some formats nest under "event"
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            t2 = evt.get("type")
            if isinstance(t2, str):
                return t2
        return ""

    @property
    def subtype(self) -> str:
        st = self.raw.get("subtype")
        if isinstance(st, str):
            return st
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            st2 = evt.get("subtype")
            if isinstance(st2, str):
                return st2
        return ""

    @property
    def message(self) -> str:
        """Best-effort extraction of text from this event.

        Handles:
          - Agent SDK-style delta events: {"type":"stream_event","event":{"delta":{"type":"text_delta","text":"..."}}}
          - CLI-style content blocks: {"content_block":{"text":"..."}}
          - Plain message/result fields.
        """
        # Agent SDK-ish delta
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            delta = evt.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    return text

        # Content block
        block = self.raw.get("content_block")
        if isinstance(block, dict):
            txt = block.get("text")
            if isinstance(txt, str):
                return txt

        # Some variants: {"content":[{"type":"text","text":"..."}]}
        content = self.raw.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "".join(parts)

        # Plain fields
        for key in ("message", "text", "result"):
            v = self.raw.get(key)
            if isinstance(v, str):
                return v

        return ""

    @property
    def structured_output(self) -> Any | None:
        """Validated structured output (if present)."""
        if "structured_output" in self.raw:
            return self.raw.get("structured_output")
        evt = self.raw.get("event")
        if isinstance(evt, dict) and "structured_output" in evt:
            return evt.get("structured_output")
        return None

    @property
    def session_id(self) -> str | None:
        for key in ("session_id", "sessionId"):
            v = self.raw.get(key)
            if isinstance(v, str):
                return v
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            for key in ("session_id", "sessionId"):
                v = evt.get(key)
                if isinstance(v, str):
                    return v
        return None

    @property
    def cost_usd(self) -> float | None:
        v = self.raw.get("cost_usd")
        if isinstance(v, (int, float)):
            return float(v)
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            v2 = evt.get("cost_usd")
            if isinstance(v2, (int, float)):
                return float(v2)
        return None

    @property
    def duration_ms(self) -> float | None:
        v = self.raw.get("duration_ms")
        if isinstance(v, (int, float)):
            return float(v)
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            v2 = evt.get("duration_ms")
            if isinstance(v2, (int, float)):
                return float(v2)
        return None

    @property
    def num_turns(self) -> int | None:
        v = self.raw.get("num_turns")
        if isinstance(v, int):
            return v
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            v2 = evt.get("num_turns")
            if isinstance(v2, int):
                return v2
        return None

    @property
    def is_error(self) -> bool:
        v = self.raw.get("is_error")
        if isinstance(v, bool):
            return v
        evt = self.raw.get("event")
        if isinstance(evt, dict):
            v2 = evt.get("is_error")
            if isinstance(v2, bool):
                return v2
        return False

    def __repr__(self) -> str:
        preview = self.message[:80] + ("…" if len(self.message) > 80 else "")
        return f"StreamEvent(type={self.type!r}, message={preview!r})"


# ---------------------------------------------------------------------------
# Claude response model
# ---------------------------------------------------------------------------

@dataclass
class ClaudeResponse:
    """Parsed response from a Claude Code CLI invocation."""

    text: str = ""
    json: dict[str, Any] | list[Any] | None = None
    events: list[StreamEvent] = field(default_factory=list)

    returncode: int = 0
    stderr: str = ""
    stdout: str = ""
    cmd: list[str] | None = None
    output_format: OutputFormat | None = None

    # common metadata
    session_id: str | None = None
    cost_usd: float | None = None
    duration_ms: float | None = None
    num_turns: int | None = None
    subtype: str | None = None
    structured_output: Any | None = None
    is_error: bool = False

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        stdout: str,
        *,
        returncode: int = 0,
        stderr: str = "",
        cmd: Sequence[str] | None = None,
    ) -> "ClaudeResponse":
        return cls(
            text=stdout.strip(),
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            cmd=list(cmd) if cmd else None,
            output_format=OutputFormat.TEXT,
        )

    @classmethod
    def from_json(
        cls,
        stdout: str,
        *,
        returncode: int = 0,
        stderr: str = "",
        cmd: Sequence[str] | None = None,
    ) -> "ClaudeResponse":
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Not JSON for some reason; keep as text
            return cls.from_text(stdout, returncode=returncode, stderr=stderr, cmd=cmd)

        resp = cls(
            json=data,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            cmd=list(cmd) if cmd else None,
            output_format=OutputFormat.JSON,
        )

        if isinstance(data, dict):
            resp.session_id = _maybe_str(data.get("session_id"))
            resp.cost_usd = _maybe_float(data.get("cost_usd"))
            resp.duration_ms = _maybe_float(data.get("duration_ms"))
            resp.num_turns = _maybe_int(data.get("num_turns"))
            resp.subtype = _maybe_str(data.get("subtype"))
            resp.is_error = bool(data.get("is_error", False))

            if "structured_output" in data:
                resp.structured_output = data.get("structured_output")

            result = data.get("result", "")
            if isinstance(result, str):
                resp.text = result
            elif result is None:
                resp.text = ""
            else:
                # object/array -> JSON string
                resp.text = json.dumps(result, ensure_ascii=False)

        elif isinstance(data, list):
            resp.text = json.dumps(data, ensure_ascii=False)

        return resp

    @classmethod
    def from_stream(
        cls,
        events: list[StreamEvent],
        *,
        returncode: int = 0,
        stderr: str = "",
        stdout: str = "",
        cmd: Sequence[str] | None = None,
    ) -> "ClaudeResponse":
        resp = cls(
            events=events,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            cmd=list(cmd) if cmd else None,
            output_format=OutputFormat.STREAM_JSON,
        )

        # Look for a final result event first
        for evt in reversed(events):
            if evt.type == "result":
                resp.session_id = evt.session_id
                resp.cost_usd = evt.cost_usd
                resp.duration_ms = evt.duration_ms
                resp.num_turns = evt.num_turns
                resp.subtype = evt.subtype or None
                resp.is_error = evt.is_error
                resp.structured_output = evt.structured_output
                # Many formats include "result" text on the result event
                val = evt.raw.get("result")
                if isinstance(val, str):
                    resp.text = val
                elif val is not None:
                    resp.text = json.dumps(val, ensure_ascii=False)
                break

        # If no result text, concatenate deltas/messages
        if not resp.text:
            parts = [e.message for e in events if e.message]
            resp.text = "".join(parts).strip()

        return resp

    def __repr__(self) -> str:
        preview = self.text[:120] + ("…" if len(self.text) > 120 else "")
        return f"ClaudeResponse(text={preview!r}, returncode={self.returncode})"


def _maybe_str(v: Any) -> str | None:
    return v if isinstance(v, str) else None


def _maybe_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _maybe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Typed model response
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse(Generic[T]):
    """Response wrapper that carries a deserialized Python object.

    Returned by :meth:`claude_code_cli.client.ClaudeCode.query_model`.

    Attributes:
        data:           The deserialized instance of target class.
        raw:            The underlying ClaudeResponse.
        schema_sent:    The JSON Schema dict that was sent to the CLI.
        schema_original:The original (pre-transformation) schema (optional).
    """

    data: T
    raw: ClaudeResponse
    schema_sent: dict[str, Any] = field(default_factory=dict)
    schema_original: dict[str, Any] | None = None

    # Convenience proxies
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

    @property
    def subtype(self) -> str | None:
        return self.raw.subtype

    @property
    def structured_output(self) -> Any | None:
        return self.raw.structured_output

    def __repr__(self) -> str:
        return f"ModelResponse(data={self.data!r}, cost_usd={self.cost_usd})"
