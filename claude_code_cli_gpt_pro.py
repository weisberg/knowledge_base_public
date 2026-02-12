"""
claude_code_cli.py

A Python wrapper around the Claude Code CLI (`claude`) focused on print/headless mode
(`claude -p`).

Why this exists:
- Claude Code already supports non-interactive usage via `-p` / `--print`.
- This module gives you a typed, object-oriented way to build commands, pipe stdin,
  and parse the output (text / json / stream-json).

Docs (authoritative):
- CLI reference: https://code.claude.com/docs/en/cli-reference
- Run Claude Code programmatically ("headless"): https://code.claude.com/docs/en/headless

Notes about `-p` / print mode:
- Some flags are *print-mode-only* (e.g., --output-format, --json-schema, --max-turns).
- Some features are *interactive-only* (e.g., slash commands like /commit), so you must
  describe tasks in natural language when using `-p`.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union


# -----------------------------
# Errors
# -----------------------------

class ClaudeCLIError(RuntimeError):
    """Base exception for this module."""


class ClaudeNotFoundError(ClaudeCLIError):
    """Raised when the `claude` executable cannot be found."""


class ClaudeProcessError(ClaudeCLIError):
    """Raised when the `claude` process exits non-zero."""

    def __init__(self, message: str, *, exit_code: int, stdout: str, stderr: str, command: Sequence[str]):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.command = list(command)


class ClaudeResponseParseError(ClaudeCLIError):
    """Raised when output parsing fails (e.g., invalid JSON)."""


class ClaudeOptionError(ValueError):
    """Raised for invalid flag combinations (especially in -p mode)."""


# -----------------------------
# Enums (key CLI concepts)
# -----------------------------

class OutputFormat(str, Enum):
    """`--output-format` values supported by Claude Code CLI."""
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class InputFormat(str, Enum):
    """`--input-format` values supported by Claude Code CLI (print mode)."""
    TEXT = "text"
    STREAM_JSON = "stream-json"


class PermissionMode(str, Enum):
    """
    Permission modes from Claude Code docs.
    See https://code.claude.com/docs/en/permissions (Permission modes).
    """
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    DELEGATE = "delegate"
    DONT_ASK = "dontAsk"
    BYPASS_PERMISSIONS = "bypassPermissions"


# -----------------------------
# Data classes
# -----------------------------

@dataclass(frozen=True)
class ClaudeResponse:
    """
    Parsed response from `claude -p`.
    - For text output: `text` is stdout, `data` is None.
    - For json output: `data` is parsed JSON, `text` is usually `.result` if present.
    """
    output_format: OutputFormat
    text: str
    exit_code: int
    stdout: str
    stderr: str
    command: List[str]

    # When output_format=JSON, this holds the parsed object.
    data: Optional[Dict[str, Any]] = None

    # Convenience fields (best-effort, may be None depending on format/version)
    session_id: Optional[str] = None
    structured_output: Any = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ClaudeStreamEvent:
    """One newline-delimited JSON event from `--output-format stream-json`."""
    index: int
    raw_line: str
    event: Dict[str, Any]


@dataclass
class ClaudeOptions:
    """
    A typed representation of common CLI flags.

    This module is intentionally permissive: you can also pass raw flags via
    `extra_args` if Claude Code adds new options.
    """
    # Working directories / context
    add_dir: List[Union[str, Path]] = field(default_factory=list)

    # Agents
    agent: Optional[str] = None
    agents: Optional[Union[str, Dict[str, Any]]] = None  # JSON string or dict

    # Permissions / tools
    permission_mode: Optional[Union[str, PermissionMode]] = None
    allow_dangerously_skip_permissions: bool = False
    dangerously_skip_permissions: bool = False

    allowed_tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)
    tools: Optional[Union[str, List[str]]] = None  # "", "default", or list like ["Bash","Read"]

    # System prompt customization
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[Union[str, Path]] = None
    append_system_prompt: Optional[str] = None
    append_system_prompt_file: Optional[Union[str, Path]] = None  # print-only

    # Output and parsing
    output_format: OutputFormat = OutputFormat.TEXT
    input_format: Optional[InputFormat] = None  # print-only
    include_partial_messages: bool = False  # requires stream-json
    json_schema: Optional[Union[str, Dict[str, Any]]] = None  # print-only

    # Budget/limits (print-only)
    max_budget_usd: Optional[float] = None
    max_turns: Optional[int] = None

    # Session management
    continue_: bool = False  # --continue / -c
    resume: Optional[str] = None  # --resume / -r
    fork_session: bool = False
    session_id: Optional[str] = None  # UUID

    # Model
    model: Optional[str] = None
    fallback_model: Optional[str] = None  # print-only

    # MCP / plugins / settings
    mcp_config: List[str] = field(default_factory=list)
    strict_mcp_config: bool = False
    plugin_dir: List[Union[str, Path]] = field(default_factory=list)
    settings: Optional[str] = None  # path or JSON string
    setting_sources: Optional[str] = None  # "user,project,local"
    permission_prompt_tool: Optional[str] = None  # print-only (MCP tool)

    # Logging / debugging
    verbose: bool = False
    debug: Optional[str] = None

    # Chrome integration
    chrome: Optional[bool] = None  # True -> --chrome, False -> --no-chrome

    # Less common / mostly interactive UX flags (kept for completeness)
    disable_slash_commands: bool = False
    ide: bool = False
    teammate_mode: Optional[str] = None
    teleport: bool = False
    remote: Optional[str] = None  # create web session
    init: bool = False

    # Escape hatch: raw args appended last
    extra_args: List[str] = field(default_factory=list)

    def _as_json_arg(self, value: Union[str, Dict[str, Any]]) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def to_args(self, *, print_mode: bool = True, strict: bool = False) -> List[str]:
        """
        Convert this options object to a CLI argv list (excluding the `claude` binary,
        and excluding `-p` + prompt).

        Validation:
        - Enforces known mutual exclusions / required combinations.
        - In strict mode, errors on flags that are likely no-ops in print mode.
        """
        # --- known mutual exclusions (from docs) ---
        if self.system_prompt and self.system_prompt_file:
            raise ClaudeOptionError("--system-prompt and --system-prompt-file are mutually exclusive")

        # --- print-mode-only flags ---
        if not print_mode:
            if self.append_system_prompt_file is not None:
                raise ClaudeOptionError("--append-system-prompt-file is print mode only")
            if self.input_format is not None:
                raise ClaudeOptionError("--input-format is print mode only")
            if self.json_schema is not None:
                raise ClaudeOptionError("--json-schema is print mode only")
            if self.max_budget_usd is not None:
                raise ClaudeOptionError("--max-budget-usd is print mode only")
            if self.max_turns is not None:
                raise ClaudeOptionError("--max-turns is print mode only")
            if self.fallback_model is not None:
                raise ClaudeOptionError("--fallback-model is print mode only")
            if self.permission_prompt_tool is not None:
                raise ClaudeOptionError("--permission-prompt-tool is print mode only")
            if self.include_partial_messages:
                raise ClaudeOptionError("--include-partial-messages requires print mode")

        # include_partial_messages requires stream-json output (docs say it requires --print and --output-format=stream-json)
        if self.include_partial_messages and self.output_format != OutputFormat.STREAM_JSON:
            raise ClaudeOptionError("--include-partial-messages requires output_format=OutputFormat.STREAM_JSON")

        # These flags tend to be UX/interactive oriented; in -p they may do nothing or be rejected.
        interactive_leaning = []
        if print_mode:
            if self.ide:
                interactive_leaning.append("--ide")
            if self.init:
                interactive_leaning.append("--init")
            if self.teammate_mode:
                interactive_leaning.append("--teammate-mode")
            if self.teleport:
                interactive_leaning.append("--teleport")
            if self.remote:
                interactive_leaning.append("--remote")
        if strict and interactive_leaning:
            raise ClaudeOptionError(
                "These flags are typically interactive/UX-focused and may be ignored in -p mode: "
                + ", ".join(interactive_leaning)
            )

        args: List[str] = []

        # Working dirs
        if self.add_dir:
            args.append("--add-dir")
            args.extend(str(Path(p)) for p in self.add_dir)

        # Agents
        if self.agent:
            args.extend(["--agent", self.agent])
        if self.agents is not None:
            args.extend(["--agents", self._as_json_arg(self.agents)])

        # Permissions and tools
        if self.permission_mode is not None:
            mode = self.permission_mode.value if isinstance(self.permission_mode, PermissionMode) else str(self.permission_mode)
            args.extend(["--permission-mode", mode])
        if self.allow_dangerously_skip_permissions:
            args.append("--allow-dangerously-skip-permissions")
        if self.dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")

        if self.allowed_tools:
            args.append("--allowedTools")
            # CLI supports both a single comma-separated string and multiple entries.
            if len(self.allowed_tools) == 1:
                args.append(self.allowed_tools[0])
            else:
                args.extend(self.allowed_tools)

        if self.disallowed_tools:
            args.append("--disallowedTools")
            if len(self.disallowed_tools) == 1:
                args.append(self.disallowed_tools[0])
            else:
                args.extend(self.disallowed_tools)

        if self.tools is not None:
            if isinstance(self.tools, list):
                args.extend(["--tools", ",".join(self.tools)])
            else:
                args.extend(["--tools", str(self.tools)])

        # System prompt flags
        if self.system_prompt:
            args.extend(["--system-prompt", self.system_prompt])
        if self.system_prompt_file is not None:
            args.extend(["--system-prompt-file", str(Path(self.system_prompt_file))])
        if self.append_system_prompt:
            args.extend(["--append-system-prompt", self.append_system_prompt])
        if self.append_system_prompt_file is not None:
            args.extend(["--append-system-prompt-file", str(Path(self.append_system_prompt_file))])

        # Output / schema / streaming
        if print_mode:
            # Only specify output-format when non-default, to keep argv clean.
            if self.output_format != OutputFormat.TEXT:
                args.extend(["--output-format", self.output_format.value])
            if self.input_format is not None:
                args.extend(["--input-format", self.input_format.value])
            if self.include_partial_messages:
                args.append("--include-partial-messages")
            if self.json_schema is not None:
                args.extend(["--json-schema", self._as_json_arg(self.json_schema)])
            if self.max_budget_usd is not None:
                args.extend(["--max-budget-usd", f"{self.max_budget_usd:.2f}"])
            if self.max_turns is not None:
                args.extend(["--max-turns", str(self.max_turns)])
            if self.fallback_model is not None:
                args.extend(["--fallback-model", self.fallback_model])
            if self.permission_prompt_tool is not None:
                args.extend(["--permission-prompt-tool", self.permission_prompt_tool])

        # Sessions
        if self.continue_:
            args.append("--continue")
        if self.resume:
            args.extend(["--resume", self.resume])
        if self.fork_session:
            args.append("--fork-session")
        if self.session_id:
            args.extend(["--session-id", self.session_id])

        # Model
        if self.model:
            args.extend(["--model", self.model])

        # MCP / plugins / settings
        if self.mcp_config:
            args.append("--mcp-config")
            args.extend(self.mcp_config)
        if self.strict_mcp_config:
            args.append("--strict-mcp-config")
        if self.plugin_dir:
            for p in self.plugin_dir:
                args.extend(["--plugin-dir", str(Path(p))])
        if self.settings:
            args.extend(["--settings", self.settings])
        if self.setting_sources:
            args.extend(["--setting-sources", self.setting_sources])

        # Logs / debug
        if self.verbose:
            args.append("--verbose")
        if self.debug:
            args.extend(["--debug", self.debug])

        # Chrome integration
        if self.chrome is True:
            args.append("--chrome")
        elif self.chrome is False:
            args.append("--no-chrome")

        # Misc / interactive-ish
        if self.disable_slash_commands:
            args.append("--disable-slash-commands")
        if self.ide:
            args.append("--ide")
        if self.teammate_mode:
            args.extend(["--teammate-mode", self.teammate_mode])
        if self.teleport:
            args.append("--teleport")
        if self.remote:
            args.extend(["--remote", self.remote])
        if self.init:
            args.append("--init")

        # Escape hatch
        args.extend(self.extra_args)

        return args


# -----------------------------
# Client
# -----------------------------

class Client:
    """
    Primary entry point.

    Example:
        import claude_code_cli as claude

        client = claude.Client()
        resp = client.query("Explain this function", stdin=open("foo.py").read())
        print(resp.text)
    """

    def __init__(
        self,
        *,
        cli_path: str = "claude",
        default_options: Optional[ClaudeOptions] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Mapping[str, str]] = None,
        strict_print_mode: bool = False,
    ) -> None:
        self._cli_path = cli_path
        self._default_options = default_options or ClaudeOptions()
        self._cwd = Path(cwd) if cwd is not None else None
        self._env = dict(env) if env is not None else None
        self._strict_print_mode = strict_print_mode

    # ---- public helpers ----

    @property
    def cli_path(self) -> str:
        return self._cli_path

    def build_command(self, prompt: str, *, options: Optional[ClaudeOptions] = None) -> List[str]:
        """Build argv for `claude -p <prompt>` with the provided options."""
        opts = self._merge_options(options)
        argv = [self._cli_path]
        argv.extend(opts.to_args(print_mode=True, strict=self._strict_print_mode))
        argv.extend(["-p", prompt])
        return argv

    def query(
        self,
        prompt: str,
        *,
        stdin: Optional[Union[str, bytes]] = None,
        options: Optional[ClaudeOptions] = None,
        timeout_s: Optional[float] = None,
        check: bool = True,
        encoding: str = "utf-8",
    ) -> ClaudeResponse:
        """
        Run a one-off `claude -p` query and return a parsed ClaudeResponse.

        Args:
            prompt: The prompt passed to `claude -p "<prompt>"`.
            stdin: Optional content to pipe to Claude (equivalent to `cat file | claude -p ...`).
            options: Optional ClaudeOptions overriding defaults.
            timeout_s: Kill the process if it doesn't exit within this many seconds.
            check: If True, raise ClaudeProcessError on non-zero exit codes.
            encoding: Used to decode stdout/stderr if needed.

        Returns:
            ClaudeResponse
        """
        opts = self._merge_options(options)

        argv = [self._cli_path]
        argv.extend(opts.to_args(print_mode=True, strict=self._strict_print_mode))
        argv.extend(["-p", prompt])

        in_text: Optional[str] = None
        if stdin is not None:
            in_text = stdin.decode(encoding, errors="replace") if isinstance(stdin, (bytes, bytearray)) else str(stdin)

        try:
            proc = subprocess.run(
                argv,
                input=in_text,
                text=True,
                capture_output=True,
                cwd=str(self._cwd) if self._cwd else None,
                env=self._merged_env(),
                timeout=timeout_s,
            )
        except FileNotFoundError as e:
            raise ClaudeNotFoundError(
                f"Could not find Claude Code CLI executable '{self._cli_path}'. Is Claude Code installed and on PATH?"
            ) from e

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if check and proc.returncode != 0:
            raise ClaudeProcessError(
                f"`claude` exited with code {proc.returncode}",
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                command=argv,
            )

        # Parse according to output format.
        if opts.output_format == OutputFormat.JSON:
            data = self._parse_json(stdout, argv)
            # headless docs mention `.result`, and `.structured_output` when using --json-schema
            text = str(data.get("result", "")) if isinstance(data, dict) else ""
            return ClaudeResponse(
                output_format=opts.output_format,
                text=text,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                command=list(argv),
                data=data,
                session_id=data.get("session_id"),
                structured_output=data.get("structured_output"),
                metadata={k: v for k, v in data.items() if k not in {"result", "structured_output"}},
            )

        if opts.output_format == OutputFormat.STREAM_JSON:
            # For stream-json, prefer using `stream()` which yields events.
            # Here we just return raw text and no parsing.
            return ClaudeResponse(
                output_format=opts.output_format,
                text=stdout,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                command=list(argv),
            )

        # Default: text output.
        return ClaudeResponse(
            output_format=opts.output_format,
            text=stdout,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            command=list(argv),
        )

    def stream(
        self,
        prompt: str,
        *,
        stdin: Optional[Union[str, bytes]] = None,
        options: Optional[ClaudeOptions] = None,
        timeout_s: Optional[float] = None,
        encoding: str = "utf-8",
        check: bool = True,
    ) -> Iterator[ClaudeStreamEvent]:
        """
        Stream events from `claude -p --output-format stream-json ...`.

        This yields ClaudeStreamEvent objects. Typical usage is to set:
            options.output_format = OutputFormat.STREAM_JSON
            options.include_partial_messages = True
            options.verbose = True

        Docs say each line is a JSON object representing an event.
        """
        opts = self._merge_options(options)
        if opts.output_format != OutputFormat.STREAM_JSON:
            raise ClaudeOptionError("stream() requires options.output_format = OutputFormat.STREAM_JSON")

        argv = [self._cli_path]
        argv.extend(opts.to_args(print_mode=True, strict=self._strict_print_mode))
        argv.extend(["-p", prompt])

        in_bytes: Optional[bytes] = None
        if stdin is not None:
            in_bytes = stdin if isinstance(stdin, (bytes, bytearray)) else str(stdin).encode(encoding, errors="replace")

        # Drain stderr in a background thread to avoid deadlocks on long runs.
        stderr_chunks: List[bytes] = []

        def _drain_stderr(pipe) -> None:
            try:
                for chunk in iter(lambda: pipe.read(4096), b""):
                    if chunk:
                        stderr_chunks.append(chunk)
            except Exception:
                # Best-effort drain
                pass

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._cwd) if self._cwd else None,
                env=self._merged_env(),
            )
        except FileNotFoundError as e:
            raise ClaudeNotFoundError(
                f"Could not find Claude Code CLI executable '{self._cli_path}'. Is Claude Code installed and on PATH?"
            ) from e

        assert proc.stdout is not None
        assert proc.stderr is not None

        t = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
        t.start()

        try:
            if in_bytes is not None and proc.stdin is not None:
                proc.stdin.write(in_bytes)
            if proc.stdin is not None:
                proc.stdin.close()

            i = 0
            for raw in proc.stdout:
                # Popen yields bytes by default.
                line = raw.decode(encoding, errors="replace").strip()
                if not line:
                    continue
                event = self._parse_json(line, argv)
                yield ClaudeStreamEvent(index=i, raw_line=line, event=event)
                i += 1

            proc.wait(timeout=timeout_s)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

        t.join(timeout=0.2)
        stderr_text = b"".join(stderr_chunks).decode(encoding, errors="replace")

        if check and proc.returncode not in (0, None):
            # We don't have stdout captured as a single string in streaming mode; caller already got events.
            raise ClaudeProcessError(
                f"`claude` exited with code {proc.returncode} (streaming mode)",
                exit_code=int(proc.returncode),
                stdout="",
                stderr=stderr_text,
                command=argv,
            )

    @staticmethod
    def iter_text_deltas(events: Iterable[ClaudeStreamEvent]) -> Iterator[str]:
        """
        Convenience helper: yield text tokens/deltas from stream-json events.

        The headless docs show a jq filter looking for:
            .type == "stream_event" and .event.delta.type == "text_delta"
        and reading:
            .event.delta.text

        This helper mirrors that pattern and ignores events that don't match.
        """
        for ev in events:
            try:
                if ev.event.get("type") != "stream_event":
                    continue
                delta = (ev.event.get("event") or {}).get("delta") or {}
                if delta.get("type") != "text_delta":
                    continue
                text = delta.get("text")
                if isinstance(text, str):
                    yield text
            except Exception:
                continue

    def version(self) -> str:
        """Return `claude --version` output."""
        argv = [self._cli_path, "--version"]
        try:
            proc = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                cwd=str(self._cwd) if self._cwd else None,
                env=self._merged_env(),
                timeout=10,
            )
        except FileNotFoundError as e:
            raise ClaudeNotFoundError(
                f"Could not find Claude Code CLI executable '{self._cli_path}'. Is Claude Code installed and on PATH?"
            ) from e

        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            raise ClaudeProcessError(
                f"`claude --version` exited with code {proc.returncode}",
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                command=argv,
            )
        return out

    # ---- internals ----

    def _merge_options(self, overrides: Optional[ClaudeOptions]) -> ClaudeOptions:
        if overrides is None:
            return self._default_options
        # Merge by shallow-copying dataclass fields.
        base = self._default_options
        merged = ClaudeOptions(**{**base.__dict__, **overrides.__dict__})
        return merged

    def _merged_env(self) -> Optional[Dict[str, str]]:
        if self._env is None:
            return None
        env = os.environ.copy()
        env.update(self._env)
        return env

    @staticmethod
    def _parse_json(text: str, argv: Sequence[str]) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ClaudeResponseParseError(
                f"Failed to parse JSON output from `claude`. First 200 chars:\n{text[:200]!r}\nCommand: {list(argv)}"
            ) from e
        if not isinstance(parsed, dict):
            # stream-json might technically emit other JSON types, but docs suggest objects.
            return {"_value": parsed}
        return parsed


# -----------------------------
# Module-level convenience API
# -----------------------------

_default_client = Client()

def ask(
    prompt: str,
    *,
    stdin: Optional[Union[str, bytes]] = None,
    options: Optional[ClaudeOptions] = None,
    timeout_s: Optional[float] = None,
) -> ClaudeResponse:
    """
    Convenience wrapper around the default client:
        claude_code_cli.ask("Explain recursion")
    """
    return _default_client.query(prompt, stdin=stdin, options=options, timeout_s=timeout_s)


__all__ = [
    "Client",
    "ClaudeOptions",
    "ClaudeResponse",
    "ClaudeStreamEvent",
    "ClaudeCLIError",
    "ClaudeNotFoundError",
    "ClaudeProcessError",
    "ClaudeResponseParseError",
    "ClaudeOptionError",
    "OutputFormat",
    "InputFormat",
    "PermissionMode",
    "ask",
]
