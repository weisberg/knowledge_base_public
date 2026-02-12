"""Core client for interacting with the Claude Code CLI via `claude -p`."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Generator, Iterator

from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)
from claude_code_cli.models import (
    Agent,
    ClaudeResponse,
    InputFormat,
    ModelResponse,
    OutputFormat,
    PermissionMode,
    StreamEvent,
    ToolSet,
)
from claude_code_cli.schema import extract_schema


def _find_claude_binary() -> str:
    """Locate the ``claude`` binary on PATH.

    Returns the absolute path string, or raises :class:`ClaudeNotFoundError`.
    """
    path = shutil.which("claude")
    if path is None:
        raise ClaudeNotFoundError()
    return path


class ClaudeCode:
    """High-level Python wrapper around the Claude Code CLI (``claude -p``).

    Instantiate once with shared defaults, then call :meth:`query`,
    :meth:`stream`, or :meth:`run` for each interaction.

    Parameters
    ----------
    model : str, optional
        Model alias (``"sonnet"``, ``"opus"``) or full model string.
    max_turns : int, optional
        Limit the number of agentic turns (``--max-turns``).
    system_prompt : str, optional
        Replace the default system prompt entirely (``--system-prompt``).
    append_system_prompt : str, optional
        Append text to the default system prompt (``--append-system-prompt``).
    system_prompt_file : str | Path, optional
        Load system prompt from a file (``--system-prompt-file``).
    output_format : OutputFormat, optional
        Default output format.  Defaults to ``OutputFormat.JSON``.
    tools : ToolSet | list[str], optional
        Built-in tools to enable.
    allowed_tools : list[str], optional
        Tools allowed without prompting (``--allowedTools``).
    disallowed_tools : list[str], optional
        Tools disallowed (``--disallowedTools``).
    mcp_config : str | Path | list[str | Path], optional
        Path(s) to MCP config JSON file(s) (``--mcp-config``).
    strict_mcp_config : bool
        Only use MCP servers from ``--mcp-config`` (``--strict-mcp-config``).
    permission_mode : PermissionMode, optional
        Permission mode (``--permission-mode``).
    permission_prompt_tool : str, optional
        MCP tool for permission prompts (``--permission-prompt-tool``).
    agents : list[Agent], optional
        Custom subagent definitions (``--agents``).
    agent : str, optional
        Use a specific named agent (``--agent``).
    add_dirs : list[str | Path], optional
        Additional working directories (``--add-dir``).
    verbose : bool
        Enable verbose logging (``--verbose``).
    debug : str | bool, optional
        Enable debug mode; pass ``True`` or a category filter string (``--debug``).
    dangerously_skip_permissions : bool
        Skip all permission prompts (``--dangerously-skip-permissions``).
    fallback_model : str, optional
        Fallback model when default is overloaded (``--fallback-model``).
    betas : list[str], optional
        Beta feature headers (``--betas``).
    settings : str | Path, optional
        Path to a settings JSON file (``--settings``).
    setting_sources : list[str], optional
        Setting sources to load (``--setting-sources``).
    plugin_dirs : list[str | Path], optional
        Plugin directory paths (``--plugin-dir``).
    cwd : str | Path, optional
        Default working directory for CLI invocations.
    timeout : float, optional
        Default timeout in seconds for non-streaming calls.
    claude_binary : str, optional
        Explicit path to the ``claude`` binary.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_turns: int | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        system_prompt_file: str | Path | None = None,
        output_format: OutputFormat = OutputFormat.JSON,
        tools: ToolSet | list[str] | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: str | Path | list[str | Path] | None = None,
        strict_mcp_config: bool = False,
        permission_mode: PermissionMode | None = None,
        permission_prompt_tool: str | None = None,
        agents: list[Agent] | None = None,
        agent: str | None = None,
        add_dirs: list[str | Path] | None = None,
        verbose: bool = False,
        debug: str | bool | None = None,
        dangerously_skip_permissions: bool = False,
        fallback_model: str | None = None,
        betas: list[str] | None = None,
        settings: str | Path | None = None,
        setting_sources: list[str] | None = None,
        plugin_dirs: list[str | Path] | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        claude_binary: str | None = None,
    ) -> None:
        self.claude_binary = claude_binary or _find_claude_binary()
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.append_system_prompt = append_system_prompt
        self.system_prompt_file = system_prompt_file
        self.output_format = output_format
        self.tools = tools
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.mcp_config = mcp_config
        self.strict_mcp_config = strict_mcp_config
        self.permission_mode = permission_mode
        self.permission_prompt_tool = permission_prompt_tool
        self.agents = agents
        self.agent = agent
        self.add_dirs = add_dirs
        self.verbose = verbose
        self.debug = debug
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.fallback_model = fallback_model
        self.betas = betas
        self.settings = settings
        self.setting_sources = setting_sources
        self.plugin_dirs = plugin_dirs
        self.cwd = str(cwd) if cwd else None
        self.timeout = timeout

    # ------------------------------------------------------------------
    # CLI argument builder
    # ------------------------------------------------------------------

    def _build_args(  # noqa: C901 – complexity is inherent in flag mapping
        self,
        prompt: str,
        *,
        # Per-call overrides (mirror constructor params)
        model: str | None = None,
        max_turns: int | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        system_prompt_file: str | Path | None = None,
        output_format: OutputFormat | None = None,
        input_format: InputFormat | None = None,
        json_schema: dict[str, Any] | str | None = None,
        tools: ToolSet | list[str] | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: str | Path | list[str | Path] | None = None,
        strict_mcp_config: bool | None = None,
        permission_mode: PermissionMode | None = None,
        permission_prompt_tool: str | None = None,
        agents: list[Agent] | None = None,
        agent: str | None = None,
        add_dirs: list[str | Path] | None = None,
        verbose: bool | None = None,
        debug: str | bool | None = None,
        dangerously_skip_permissions: bool | None = None,
        fallback_model: str | None = None,
        include_partial_messages: bool = False,
        betas: list[str] | None = None,
        settings: str | Path | None = None,
        setting_sources: list[str] | None = None,
        plugin_dirs: list[str | Path] | None = None,
        continue_conversation: bool = False,
        resume: str | None = None,
        session_id: str | None = None,
        fork_session: bool = False,
    ) -> list[str]:
        """Build the full ``claude -p ...`` argument list."""
        args: list[str] = [self.claude_binary, "-p"]

        # --- resolve effective values (call-level overrides > instance) ---
        _model = model or self.model
        _max_turns = max_turns if max_turns is not None else self.max_turns
        _sys = system_prompt or self.system_prompt
        _sys_append = append_system_prompt or self.append_system_prompt
        _sys_file = system_prompt_file or self.system_prompt_file
        _fmt = output_format or self.output_format
        _tools = tools or self.tools
        _allowed = allowed_tools or self.allowed_tools
        _disallowed = disallowed_tools or self.disallowed_tools
        _mcp = mcp_config or self.mcp_config
        _strict_mcp = strict_mcp_config if strict_mcp_config is not None else self.strict_mcp_config
        _perm_mode = permission_mode or self.permission_mode
        _perm_tool = permission_prompt_tool or self.permission_prompt_tool
        _agents = agents or self.agents
        _agent = agent or self.agent
        _add_dirs = add_dirs or self.add_dirs
        _verbose = verbose if verbose is not None else self.verbose
        _debug = debug if debug is not None else self.debug
        _skip_perms = (
            dangerously_skip_permissions
            if dangerously_skip_permissions is not None
            else self.dangerously_skip_permissions
        )
        _fallback = fallback_model or self.fallback_model
        _betas = betas or self.betas
        _settings = settings or self.settings
        _setting_src = setting_sources or self.setting_sources
        _plugin_dirs = plugin_dirs or self.plugin_dirs

        # --- conversation continuity ---
        if continue_conversation:
            args.append("--continue")
        if resume:
            args.extend(["--resume", resume])
        if session_id:
            args.extend(["--session-id", session_id])
        if fork_session:
            args.append("--fork-session")

        # --- model ---
        if _model:
            args.extend(["--model", _model])
        if _fallback:
            args.extend(["--fallback-model", _fallback])

        # --- output / input format ---
        if _fmt and _fmt != OutputFormat.TEXT:
            args.extend(["--output-format", _fmt.value])
        if input_format:
            args.extend(["--input-format", input_format.value])

        # --- structured output ---
        if json_schema:
            schema_str = json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
            args.extend(["--json-schema", schema_str])

        # --- system prompt ---
        if _sys:
            args.extend(["--system-prompt", _sys])
        if _sys_file:
            args.extend(["--system-prompt-file", str(_sys_file)])
        if _sys_append:
            args.extend(["--append-system-prompt", _sys_append])

        # --- turns ---
        if _max_turns is not None:
            args.extend(["--max-turns", str(_max_turns)])

        # --- tools ---
        if _tools:
            if isinstance(_tools, ToolSet):
                args.extend(_tools.to_cli_args())
            else:
                args.extend(["--tools", ",".join(_tools)])
        if _allowed:
            args.extend(["--allowedTools"] + _allowed)
        if _disallowed:
            args.extend(["--disallowedTools"] + _disallowed)

        # --- MCP ---
        if _mcp:
            configs = _mcp if isinstance(_mcp, list) else [_mcp]
            args.extend(["--mcp-config"] + [str(c) for c in configs])
        if _strict_mcp:
            args.append("--strict-mcp-config")

        # --- permissions ---
        if _perm_mode:
            args.extend(["--permission-mode", _perm_mode.value])
        if _perm_tool:
            args.extend(["--permission-prompt-tool", _perm_tool])
        if _skip_perms:
            args.append("--dangerously-skip-permissions")

        # --- agents ---
        if _agents:
            agents_dict = {a.name: a.to_dict() for a in _agents}
            args.extend(["--agents", json.dumps(agents_dict)])
        if _agent:
            args.extend(["--agent", _agent])

        # --- directories ---
        if _add_dirs:
            args.extend(["--add-dir"] + [str(d) for d in _add_dirs])

        # --- logging / debug ---
        if _verbose:
            args.append("--verbose")
        if _debug:
            if isinstance(_debug, bool):
                args.append("--debug")
            else:
                args.extend(["--debug", _debug])

        # --- partial messages ---
        if include_partial_messages:
            args.append("--include-partial-messages")

        # --- betas ---
        if _betas:
            args.extend(["--betas"] + _betas)

        # --- settings ---
        if _settings:
            args.extend(["--settings", str(_settings)])
        if _setting_src:
            args.extend(["--setting-sources", ",".join(_setting_src)])

        # --- plugins ---
        if _plugin_dirs:
            for pd in _plugin_dirs:
                args.extend(["--plugin-dir", str(pd)])

        # --- prompt (always last) ---
        args.append(prompt)

        return args

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _resolve_cwd(self, cwd: str | Path | None) -> str | None:
        return str(cwd) if cwd else self.cwd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """Send a prompt to Claude Code and return the full response.

        Parameters
        ----------
        prompt : str
            The user prompt / question.
        stdin : str, optional
            Data to pipe into Claude's stdin (equivalent to ``cat data | claude -p``).
        cwd : str | Path, optional
            Working directory for this call (overrides instance default).
        timeout : float, optional
            Timeout in seconds (overrides instance default).
        **kwargs
            Any per-call flag overrides accepted by :meth:`_build_args`.

        Returns
        -------
        ClaudeResponse
        """
        # Determine the effective output format
        fmt = kwargs.get("output_format") or self.output_format

        args = self._build_args(prompt, **kwargs)
        effective_timeout = timeout or self.timeout

        try:
            result = subprocess.run(
                args,
                input=stdin,
                capture_output=True,
                text=True,
                cwd=self._resolve_cwd(cwd),
                timeout=effective_timeout,
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError()
        except subprocess.TimeoutExpired:
            raise ClaudeTimeoutError(effective_timeout or 0)

        if result.returncode != 0:
            raise ClaudeProcessError(
                returncode=result.returncode,
                stderr=result.stderr,
                cmd=args,
            )

        if fmt == OutputFormat.JSON:
            return ClaudeResponse.from_json(result.stdout, result.returncode, result.stderr)
        elif fmt == OutputFormat.STREAM_JSON:
            events = _parse_stream_events(result.stdout)
            return ClaudeResponse.from_stream(events, result.returncode, result.stderr)
        else:
            return ClaudeResponse.from_text(result.stdout, result.returncode, result.stderr)

    def stream(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        include_partial_messages: bool = False,
        **kwargs: Any,
    ) -> Generator[StreamEvent, None, ClaudeResponse]:
        """Stream events from Claude Code in real-time.

        Yields :class:`StreamEvent` objects as they arrive. The final
        :class:`ClaudeResponse` is available as the generator's return value
        (accessible via ``StopIteration.value``).

        Usage::

            gen = client.stream("Refactor this file")
            try:
                while True:
                    event = next(gen)
                    print(event.message, end="", flush=True)
            except StopIteration as e:
                response = e.value
                print(f"\\nDone — cost: ${response.cost_usd}")

        Or use the convenience :meth:`stream_text` for simple cases.

        Parameters
        ----------
        prompt : str
            The user prompt.
        stdin : str, optional
            Data to pipe to stdin.
        cwd : str | Path, optional
            Working directory.
        include_partial_messages : bool
            Include partial streaming events (``--include-partial-messages``).
        **kwargs
            Any per-call flag overrides accepted by :meth:`_build_args`.

        Yields
        ------
        StreamEvent
        """
        kwargs["output_format"] = OutputFormat.STREAM_JSON
        kwargs["include_partial_messages"] = include_partial_messages

        args = self._build_args(prompt, **kwargs)

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE if stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._resolve_cwd(cwd),
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError()

        # Write stdin if provided, then close
        if stdin and proc.stdin:
            proc.stdin.write(stdin)
            proc.stdin.close()

        events: list[StreamEvent] = []
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event = StreamEvent(raw=data)
                events.append(event)
                yield event
            except json.JSONDecodeError:
                continue

        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""

        if proc.returncode != 0:
            raise ClaudeProcessError(
                returncode=proc.returncode,
                stderr=stderr,
                cmd=args,
            )

        return ClaudeResponse.from_stream(events, proc.returncode, stderr)

    def stream_text(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Convenience iterator that yields only the text fragments.

        Usage::

            for chunk in client.stream_text("Explain this code"):
                print(chunk, end="", flush=True)
        """
        gen = self.stream(prompt, stdin=stdin, cwd=cwd, include_partial_messages=True, **kwargs)
        try:
            while True:
                event = next(gen)
                text = event.message
                if text:
                    yield text
        except StopIteration:
            return

    def run(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Minimal convenience method — returns just the text output.

        Equivalent to ``client.query(...).text``.
        """
        return self.query(prompt, stdin=stdin, cwd=cwd, timeout=timeout, **kwargs).text

    def query_model(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        json_schema_override: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Query Claude Code and return a populated instance of *model_class*.

        This is the structured-output convenience method.  It:

        1. Extracts a JSON Schema from *model_class* (dataclass, Pydantic
           model, TypedDict, or annotated class).
        2. Passes it to the CLI via ``--json-schema``.
        3. Parses the JSON response and deserialises it into an instance of
           *model_class*.

        Parameters
        ----------
        prompt : str
            The user prompt.
        model_class : type
            The target Python class.  Must be a dataclass, Pydantic
            ``BaseModel``, ``TypedDict``, or a class with ``__annotations__``.
        stdin : str, optional
            Data to pipe to stdin.
        cwd : str | Path, optional
            Working directory override.
        timeout : float, optional
            Timeout override.
        json_schema_override : dict, optional
            Provide an explicit JSON Schema instead of auto-extracting from
            *model_class*.  Useful when you need to tweak the generated schema.
        **kwargs
            Any additional per-call flag overrides.

        Returns
        -------
        ModelResponse[T]
            A wrapper carrying ``.data`` (the deserialized instance),
            ``.raw`` (the full :class:`ClaudeResponse`), and ``.schema``.

        Example
        -------
        ::

            @dataclass
            class CodeReview:
                summary: str
                issues: list[str]
                severity: str  # "low" | "medium" | "high"

            result = client.query_model("Review auth.py", CodeReview)
            print(result.data.summary)
            print(result.data.issues)
            print(result.cost_usd)
        """
        schema = json_schema_override or extract_schema(model_class)

        # Force JSON output so we can parse the structured result
        kwargs["output_format"] = OutputFormat.JSON
        kwargs["json_schema"] = schema

        response = self.query(
            prompt, stdin=stdin, cwd=cwd, timeout=timeout, **kwargs
        )

        data = _deserialize(response, model_class)

        return ModelResponse(data=data, raw=response, schema=schema)

    def query_model_text(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Shorthand that returns just the deserialized object.

        Equivalent to ``client.query_model(...).data``.
        """
        return self.query_model(
            prompt, model_class, stdin=stdin, cwd=cwd, timeout=timeout, **kwargs
        ).data

    # ------------------------------------------------------------------
    # Version / health
    # ------------------------------------------------------------------

    def version(self) -> str:
        """Return the installed Claude Code CLI version string."""
        try:
            result = subprocess.run(
                [self.claude_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except FileNotFoundError:
            raise ClaudeNotFoundError()

    def is_available(self) -> bool:
        """Check whether the ``claude`` binary is reachable."""
        try:
            self.version()
            return True
        except (ClaudeCodeError, subprocess.SubprocessError):
            return False

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = ["ClaudeCode("]
        if self.model:
            parts.append(f"model={self.model!r}, ")
        if self.cwd:
            parts.append(f"cwd={self.cwd!r}, ")
        parts.append(")")
        return "".join(parts)


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------

def _parse_stream_events(text: str) -> list[StreamEvent]:
    """Parse newline-delimited JSON stream output into StreamEvent objects."""
    events: list[StreamEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(StreamEvent(raw=json.loads(line)))
        except json.JSONDecodeError:
            continue
    return events


# --------------------------------------------------------------------------
# Deserialization
# --------------------------------------------------------------------------

def _deserialize(response: ClaudeResponse, model_class: type) -> Any:
    """Deserialize a :class:`ClaudeResponse` into an instance of *model_class*.

    Handles dataclasses, Pydantic models, TypedDicts, and plain annotated
    classes with a best-effort recursive approach.
    """
    from claude_code_cli.schema import is_pydantic_model, is_dataclass, is_typed_dict

    # Get the raw data dict from the response
    data = _extract_data_dict(response)

    if not isinstance(data, dict):
        raise ClaudeCodeError(
            f"Expected a JSON object from Claude, got {type(data).__name__}: {str(data)[:200]}"
        )

    return _instantiate(data, model_class)


def _extract_data_dict(response: ClaudeResponse) -> Any:
    """Pull the structured output dict from a ClaudeResponse.

    The JSON-mode response from ``claude -p --output-format json --json-schema``
    wraps the structured output in a result envelope. We need to find the
    actual object data.
    """
    # If the response has parsed JSON, use it
    if response.json is not None:
        payload = response.json
        # The CLI JSON envelope has a "result" field containing the text output.
        # When --json-schema is used, the "result" field is a JSON *string*
        # with the structured data.
        if isinstance(payload, dict) and "result" in payload:
            result = payload["result"]
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass
            if isinstance(result, dict):
                return result
        # Maybe the whole payload is the structured output directly
        return payload

    # Fall back to parsing the text
    if response.text:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            pass

    raise ClaudeCodeError(
        "Could not extract structured data from Claude response. "
        f"text={response.text[:200]!r}"
    )


def _instantiate(data: dict[str, Any], cls: type) -> Any:
    """Recursively instantiate a class from a data dict."""
    import dataclasses as dc
    from typing import get_type_hints, get_origin, get_args

    from claude_code_cli.schema import is_pydantic_model, is_dataclass, is_typed_dict

    # --- Pydantic v2 ---
    if is_pydantic_model(cls):
        try:
            import pydantic
            if hasattr(cls, "model_validate"):
                return cls.model_validate(data)
            else:
                return cls(**data)  # Pydantic v1 fallback
        except Exception:
            return cls(**data)

    # --- dataclass ---
    if is_dataclass(cls):
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            hints = {}
        fields = {f.name for f in dc.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in fields:
                continue
            if key in hints:
                kwargs[key] = _coerce_value(value, hints[key])
            else:
                kwargs[key] = value
        return cls(**kwargs)

    # --- TypedDict ---
    if is_typed_dict(cls):
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            hints = {}
        coerced: dict[str, Any] = {}
        for key, value in data.items():
            if key in hints:
                coerced[key] = _coerce_value(value, hints[key])
            else:
                coerced[key] = value
        return coerced  # TypedDicts are just dicts at runtime

    # --- Plain annotated class ---
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}

    # Try to construct with kwargs
    try:
        instance = cls.__new__(cls)
        for key, value in data.items():
            if key in hints:
                setattr(instance, key, _coerce_value(value, hints[key]))
            else:
                setattr(instance, key, value)
        return instance
    except Exception:
        # Last resort: just try calling the constructor
        return cls(**data)


def _coerce_value(value: Any, target_type: Any) -> Any:
    """Recursively coerce a JSON value into the target Python type.

    Handles nested dataclasses/models, lists of models, etc.
    """
    import inspect
    from typing import get_origin, get_args, Union

    from claude_code_cli.schema import (
        is_pydantic_model,
        is_dataclass,
        is_typed_dict,
        _is_annotated,
    )

    if value is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Annotated[X, ...] → unwrap to X
    if origin is not None and _is_annotated(target_type):
        return _coerce_value(value, args[0])

    # Optional[X] / Union[X, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(value, non_none[0])
        # For complex unions, return as-is
        return value

    # list[X]
    if origin is list and args and isinstance(value, list):
        return [_coerce_value(item, args[0]) for item in value]

    # dict[K, V]
    if origin is dict and args and len(args) == 2 and isinstance(value, dict):
        return {k: _coerce_value(v, args[1]) for k, v in value.items()}

    # Nested model class
    if (
        inspect.isclass(target_type)
        and isinstance(value, dict)
        and (is_dataclass(target_type) or is_typed_dict(target_type) or is_pydantic_model(target_type)
             or hasattr(target_type, "__annotations__"))
    ):
        return _instantiate(value, target_type)

    # Enum
    if inspect.isclass(target_type) and issubclass(target_type, __import__("enum").Enum):
        try:
            return target_type(value)
        except (ValueError, KeyError):
            return value

    return value
