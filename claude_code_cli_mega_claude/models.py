"""Response and event models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from ``--output-format stream-json``.

    The :attr:`raw` dict is the parsed JSON line.  Typed properties provide
    convenient access to common fields.
    """
    raw: dict[str, Any]
    index: int = 0

    # -- typed accessors ----------------------------------------------------

    @property
    def type(self) -> str:
        return self.raw.get("type", "")

    @property
    def subtype(self) -> str:
        return self.raw.get("subtype", "")

    @property
    def message(self) -> str:
        """Best-effort extraction of the text content."""
        # stream_event → event.delta.text (real streaming format from headless docs)
        if self.raw.get("type") == "stream_event":
            delta = (self.raw.get("event") or {}).get("delta") or {}
            if delta.get("type") == "text_delta":
                return delta.get("text", "")

        # content_block wrapper
        if "content_block" in self.raw:
            block = self.raw["content_block"]
            if isinstance(block, dict):
                return block.get("text", "")

        # Top-level result / message
        if "result" in self.raw:
            return str(self.raw["result"])
        if "message" in self.raw:
            return str(self.raw["message"])
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
        preview = self.message[:80] if self.message else ""
        return f"StreamEvent(type={self.type!r}, message={preview!r})"


# ---------------------------------------------------------------------------
# ClaudeResponse
# ---------------------------------------------------------------------------

@dataclass
class ClaudeResponse:
    """Parsed response from a ``claude -p`` invocation.

    Attributes
    ----------
    text : str
        The plain-text result (always populated on success).
    json : dict | list | None
        The full parsed JSON payload (when ``output_format`` is JSON).
    data : dict | None
        Alias for *json* (for compatibility / readability).
    structured_output : Any
        The ``structured_output`` field from the JSON envelope when
        ``--json-schema`` was used, if present.
    metadata : dict | None
        All non-result fields from the JSON envelope (session_id, cost, etc.).
    events : list[StreamEvent]
        List of stream events (when ``output_format`` is STREAM_JSON).
    returncode : int
        Process exit code.
    stdout : str
        Raw stdout from the process.
    stderr : str
        Raw stderr from the process.
    command : list[str]
        The full argv that was executed (useful for debugging).
    session_id, cost_usd, duration_ms, num_turns, is_error
        Convenience fields extracted from the JSON envelope or stream events.
    """

    text: str = ""
    json: dict[str, Any] | list[Any] | None = None
    data: dict[str, Any] | None = None
    structured_output: Any = None
    metadata: dict[str, Any] | None = None
    events: list[StreamEvent] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    command: list[str] = field(default_factory=list)
    session_id: str | None = None
    cost_usd: float | None = None
    duration_ms: float | None = None
    num_turns: int | None = None
    is_error: bool = False

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        stdout: str,
        returncode: int = 0,
        stderr: str = "",
        command: list[str] | None = None,
    ) -> ClaudeResponse:
        return cls(
            text=stdout.strip(),
            stdout=stdout,
            returncode=returncode,
            stderr=stderr,
            command=command or [],
        )

    @classmethod
    def from_json(
        cls,
        stdout: str,
        returncode: int = 0,
        stderr: str = "",
        command: list[str] | None = None,
    ) -> ClaudeResponse:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return cls(
                text=stdout.strip(),
                stdout=stdout,
                returncode=returncode,
                stderr=stderr,
                command=command or [],
            )

        resp = cls(
            json=payload,
            stdout=stdout,
            returncode=returncode,
            stderr=stderr,
            command=command or [],
        )

        if isinstance(payload, dict):
            resp.data = payload
            resp.session_id = payload.get("session_id")
            resp.cost_usd = payload.get("cost_usd")
            resp.duration_ms = payload.get("duration_ms")
            resp.num_turns = payload.get("num_turns")
            resp.is_error = payload.get("is_error", False)
            resp.structured_output = payload.get("structured_output")
            resp.metadata = {
                k: v for k, v in payload.items()
                if k not in {"result", "structured_output"}
            }

            # Derive text from the result field
            result = payload.get("result", "")
            resp.text = result if isinstance(result, str) else json.dumps(result)

        return resp

    @classmethod
    def from_stream(
        cls,
        events: list[StreamEvent],
        returncode: int = 0,
        stderr: str = "",
        command: list[str] | None = None,
    ) -> ClaudeResponse:
        resp = cls(
            events=events,
            returncode=returncode,
            stderr=stderr,
            command=command or [],
        )
        # The final "result" event carries the summary
        for evt in reversed(events):
            if evt.type == "result":
                resp.session_id = evt.session_id
                resp.cost_usd = evt.cost_usd
                resp.duration_ms = evt.duration_ms
                resp.num_turns = evt.num_turns
                resp.is_error = evt.is_error
                resp.text = evt.raw.get("result", "")
                resp.json = evt.raw
                resp.data = evt.raw
                break

        if not resp.text:
            parts = [e.message for e in events if e.message]
            resp.text = "".join(parts)

        return resp

    def __repr__(self) -> str:
        preview = self.text[:120] + "…" if len(self.text) > 120 else self.text
        return f"ClaudeResponse(text={preview!r}, returncode={self.returncode})"


# ---------------------------------------------------------------------------
# ModelResponse — typed structured output wrapper
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse(Generic[T]):
    """Response from :meth:`ClaudeCode.query_model`.

    Carries a deserialized Python object (``data``) alongside the full
    :class:`ClaudeResponse` (``raw``) for metadata access.
    """
    data: T
    raw: ClaudeResponse
    schema: dict[str, Any] = field(default_factory=dict)

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
