"""Core client for interacting with the Claude Code CLI via `claude -p`."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generator, Iterator, Mapping, Sequence

from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeOptionError,
    ClaudeProcessError,
    ClaudeTimeoutError,
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
from claude_code_cli.schema import extract_schema, simplify_schema_for_claude


def _find_claude_binary(cli_path: str | None = None) -> str:
    """Locate the `claude` binary on PATH or validate an explicit path."""
    if cli_path and cli_path != "claude":
        # If it's an explicit path, accept it (and let subprocess raise if invalid).
        return str(cli_path)

    path = shutil.which(cli_path or "claude")
    if path is None:
        raise ClaudeNotFoundError(cli_path or "claude")
    return path


def _merge_env(extra_env: Mapping[str, str] | None) -> dict[str, str] | None:
    if extra_env is None:
        return None
    merged = dict(os.environ)
    merged.update({k: str(v) for k, v in extra_env.items()})
    return merged


@dataclass(frozen=True)
class ClaudeOptions:
    """Typed option bundle (optional).

    You can either:
      1) Configure ClaudeCode(...) with keyword args (high-level API), OR
      2) Build a ClaudeOptions object and pass it to methods via `options=...`.

    The wrapper is primarily designed for `claude -p` usage; some flags are
    interactive-only but are included for completeness.
    """

    # model selection
    model: str | None = None
    fallback_model: str | None = None

    # formats
    output_format: OutputFormat = OutputFormat.JSON
    input_format: InputFormat | None = None

    # structured output
    json_schema: dict[str, Any] | str | None = None

    # cost/turn limits
    max_turns: int | None = None
    max_budget_usd: float | None = None

    # system prompt
    system_prompt: str | None = None
    system_prompt_file: str | Path | None = None
    append_system_prompt: str | None = None
    append_system_prompt_file: str | Path | None = None

    # tools
    tools: ToolSet | list[str] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # MCP
    mcp_config: str | Path | list[str | Path] | None = None
    strict_mcp_config: bool = False

    # permissions
    permission_mode: PermissionMode | None = None
    permission_prompt_tool: str | None = None
    allow_dangerously_skip_permissions: bool = False
    dangerously_skip_permissions: bool = False

    # agents
    agents: list[Agent] | dict[str, Any] | None = None
    agent: str | None = None

    # dirs
    add_dirs: list[str | Path] | None = None

    # debug/logging
    verbose: bool = False
    debug: str | bool | None = None

    # streaming
    include_partial_messages: bool = False

    # misc
    betas: list[str] | None = None
    settings: str | Path | None = None
    setting_sources: list[str] | None = None
    plugin_dirs: list[str | Path] | None = None

    chrome: bool | None = None
    disable_slash_commands: bool = False
    no_session_persistence: bool = False
    teammate_mode: TeammateMode | str | None = None

    # session continuity
    continue_conversation: bool = False
    resume: str | None = None
    session_id: str | None = None
    fork_session: bool = False
    from_pr: str | int | None = None

    # interactive-ish flows (included for completeness)
    init: bool = False
    init_only: bool = False
    maintenance: bool = False
    remote: str | None = None
    teleport: bool = False

    # escape hatch for flags not yet modeled
    extra_args: list[str] | None = None

    def with_overrides(self, *, strict: bool = True, **kwargs: Any) -> "ClaudeOptions":
        """Return a copy with provided fields overridden.

        By default this is strict: unknown option keys raise `ClaudeOptionError`.
        Use `strict=False` if you want to ignore unknown keys.
        """
        fields = {f.name for f in self.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = [k for k in kwargs.keys() if k not in fields]
        if unknown and strict:
            raise ClaudeOptionError(f"Unknown ClaudeOptions fields: {unknown}")
        clean: dict[str, Any] = {k: v for k, v in kwargs.items() if k in fields}
        return replace(self, **clean)


class ClaudeCode:
    """High-level Python wrapper around the Claude Code CLI (`claude -p`)."""

    def __init__(
        self,
        *,
        # core defaults
        claude_binary: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        # options (either pass a ClaudeOptions, or use keyword args below)
        options: ClaudeOptions | None = None,
        # keyword-arg convenience (mirrors ClaudeOptions)
        model: str | None = None,
        fallback_model: str | None = None,
        output_format: OutputFormat = OutputFormat.JSON,
        input_format: InputFormat | None = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        system_prompt_file: str | Path | None = None,
        append_system_prompt_file: str | Path | None = None,
        tools: ToolSet | list[str] | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: str | Path | list[str | Path] | None = None,
        strict_mcp_config: bool = False,
        permission_mode: PermissionMode | None = None,
        permission_prompt_tool: str | None = None,
        allow_dangerously_skip_permissions: bool = False,
        dangerously_skip_permissions: bool = False,
        agents: list[Agent] | dict[str, Any] | None = None,
        agent: str | None = None,
        add_dirs: list[str | Path] | None = None,
        verbose: bool = False,
        debug: str | bool | None = None,
        include_partial_messages: bool = False,
        betas: list[str] | None = None,
        settings: str | Path | None = None,
        setting_sources: list[str] | None = None,
        plugin_dirs: list[str | Path] | None = None,
        chrome: bool | None = None,
        disable_slash_commands: bool = False,
        no_session_persistence: bool = False,
        teammate_mode: TeammateMode | str | None = None,
        # session continuity defaults
        continue_conversation: bool = False,
        resume: str | None = None,
        session_id: str | None = None,
        fork_session: bool = False,
        from_pr: str | int | None = None,
        # lifecycle flags (rarely used with -p)
        init: bool = False,
        init_only: bool = False,
        maintenance: bool = False,
        remote: str | None = None,
        teleport: bool = False,
        # escape hatch
        extra_args: list[str] | None = None,
    ) -> None:
        self.claude_binary = claude_binary or _find_claude_binary("claude")
        self.cwd = str(cwd) if cwd else None
        self.timeout = timeout
        self.env = _merge_env(env)

        if options is not None:
            self.defaults = options
        else:
            self.defaults = ClaudeOptions(
                model=model,
                fallback_model=fallback_model,
                output_format=output_format,
                input_format=input_format,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                system_prompt=system_prompt,
                system_prompt_file=system_prompt_file,
                append_system_prompt=append_system_prompt,
                append_system_prompt_file=append_system_prompt_file,
                tools=tools,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                mcp_config=mcp_config,
                strict_mcp_config=strict_mcp_config,
                permission_mode=permission_mode,
                permission_prompt_tool=permission_prompt_tool,
                allow_dangerously_skip_permissions=allow_dangerously_skip_permissions,
                dangerously_skip_permissions=dangerously_skip_permissions,
                agents=agents,
                agent=agent,
                add_dirs=add_dirs,
                verbose=verbose,
                debug=debug,
                include_partial_messages=include_partial_messages,
                betas=betas,
                settings=settings,
                setting_sources=setting_sources,
                plugin_dirs=plugin_dirs,
                chrome=chrome,
                disable_slash_commands=disable_slash_commands,
                no_session_persistence=no_session_persistence,
                teammate_mode=teammate_mode,
                continue_conversation=continue_conversation,
                resume=resume,
                session_id=session_id,
                fork_session=fork_session,
                from_pr=from_pr,
                init=init,
                init_only=init_only,
                maintenance=maintenance,
                remote=remote,
                teleport=teleport,
                extra_args=extra_args,
            )

    # ------------------------------------------------------------------
    # CLI argument builder
    # ------------------------------------------------------------------

    def build_command(
        self,
        prompt: str,
        *,
        options: ClaudeOptions | None = None,
        **overrides: Any,
    ) -> list[str]:
        """Build the full `claude -p ...` argv list (for debugging)."""
        opts = (options or self.defaults).with_overrides(**overrides)
        return self._build_args(prompt, opts)

    def _build_args(self, prompt: str, opts: ClaudeOptions) -> list[str]:
        args: list[str] = [self.claude_binary, "-p"]

        # --- session continuity ---
        if opts.continue_conversation:
            args.append("--continue")
        if opts.resume:
            args.extend(["--resume", opts.resume])
        if opts.session_id:
            args.extend(["--session-id", opts.session_id])
        if opts.fork_session:
            args.append("--fork-session")
        if opts.from_pr is not None:
            args.extend(["--from-pr", str(opts.from_pr)])

        # --- model ---
        if opts.model:
            args.extend(["--model", opts.model])
        if opts.fallback_model:
            args.extend(["--fallback-model", opts.fallback_model])

        # --- formats ---
        if opts.output_format and opts.output_format != OutputFormat.TEXT:
            args.extend(["--output-format", opts.output_format.value])
        if opts.input_format:
            args.extend(["--input-format", opts.input_format.value])

        # --- structured outputs ---
        if opts.json_schema is not None:
            schema_str = opts.json_schema if isinstance(opts.json_schema, str) else json.dumps(opts.json_schema, ensure_ascii=False)
            args.extend(["--json-schema", schema_str])

        # --- system prompt flags ---
        if opts.system_prompt:
            args.extend(["--system-prompt", opts.system_prompt])
        if opts.system_prompt_file:
            args.extend(["--system-prompt-file", str(opts.system_prompt_file)])
        if opts.append_system_prompt:
            args.extend(["--append-system-prompt", opts.append_system_prompt])
        if opts.append_system_prompt_file:
            args.extend(["--append-system-prompt-file", str(opts.append_system_prompt_file)])

        # --- limits ---
        if opts.max_turns is not None:
            args.extend(["--max-turns", str(opts.max_turns)])
        if opts.max_budget_usd is not None:
            args.extend(["--max-budget-usd", f"{float(opts.max_budget_usd):.2f}"])

        # --- tools ---
        if opts.tools is not None:
            if isinstance(opts.tools, ToolSet):
                args.extend(opts.tools.to_cli_args())
            else:
                args.extend(["--tools", ",".join(opts.tools)])
        if opts.allowed_tools:
            args.extend(["--allowedTools"] + list(opts.allowed_tools))
        if opts.disallowed_tools:
            args.extend(["--disallowedTools"] + list(opts.disallowed_tools))

        # --- MCP ---
        if opts.mcp_config:
            configs = opts.mcp_config if isinstance(opts.mcp_config, list) else [opts.mcp_config]
            args.extend(["--mcp-config"] + [str(c) for c in configs])
        if opts.strict_mcp_config:
            args.append("--strict-mcp-config")

        # --- permissions ---
        if opts.permission_mode:
            args.extend(["--permission-mode", opts.permission_mode.value])
        if opts.permission_prompt_tool:
            args.extend(["--permission-prompt-tool", opts.permission_prompt_tool])
        if opts.allow_dangerously_skip_permissions:
            args.append("--allow-dangerously-skip-permissions")
        if opts.dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")

        # --- agents ---
        if opts.agents:
            if isinstance(opts.agents, dict):
                agents_dict = opts.agents
            else:
                agents_dict = {a.name: a.to_dict() for a in opts.agents}
            args.extend(["--agents", json.dumps(agents_dict, ensure_ascii=False)])
        if opts.agent:
            args.extend(["--agent", opts.agent])

        # --- directories ---
        if opts.add_dirs:
            args.extend(["--add-dir"] + [str(d) for d in opts.add_dirs])

        # --- logging / debug ---
        if opts.verbose:
            args.append("--verbose")
        if opts.debug:
            if isinstance(opts.debug, bool):
                args.append("--debug")
            else:
                args.extend(["--debug", str(opts.debug)])

        # --- partial streaming ---
        if opts.include_partial_messages:
            # Docs: requires --print and --output-format=stream-json
            if opts.output_format != OutputFormat.STREAM_JSON:
                raise ClaudeOptionError(
                    "--include-partial-messages requires output_format=OutputFormat.STREAM_JSON"
                )
            args.append("--include-partial-messages")

        # --- betas / settings / plugins ---
        if opts.betas:
            args.extend(["--betas"] + list(opts.betas))
        if opts.settings:
            args.extend(["--settings", str(opts.settings)])
        if opts.setting_sources:
            args.extend(["--setting-sources", ",".join([str(s) for s in opts.setting_sources])])
        if opts.plugin_dirs:
            for pd in opts.plugin_dirs:
                args.extend(["--plugin-dir", str(pd)])

        # --- chrome ---
        if opts.chrome is True:
            args.append("--chrome")
        elif opts.chrome is False:
            args.append("--no-chrome")

        # --- misc ---
        if opts.disable_slash_commands:
            args.append("--disable-slash-commands")
        if opts.no_session_persistence:
            args.append("--no-session-persistence")
        if opts.teammate_mode:
            tm = opts.teammate_mode.value if isinstance(opts.teammate_mode, TeammateMode) else str(opts.teammate_mode)
            args.extend(["--teammate-mode", tm])

        # --- interactive-ish flows (rarely used with -p, but included) ---
        if opts.init:
            args.append("--init")
        if opts.init_only:
            args.append("--init-only")
        if opts.maintenance:
            args.append("--maintenance")
        if opts.remote:
            args.extend(["--remote", opts.remote])
        if opts.teleport:
            args.append("--teleport")

        
        # --- escape hatch ---
        if opts.extra_args:
            args.extend([str(a) for a in opts.extra_args])
# --- prompt last ---
        args.append(prompt)
        return args

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
        env: Mapping[str, str] | None = None,
        options: ClaudeOptions | None = None,
        **overrides: Any,
    ) -> ClaudeResponse:
        """Send a prompt to Claude Code and return the full response."""
        opts = (options or self.defaults).with_overrides(**overrides)

        args = self._build_args(prompt, opts)
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_env = _merge_env(env) if env is not None else self.env

        try:
            result = subprocess.run(
                args,
                input=stdin,
                capture_output=True,
                text=True,
                cwd=self._resolve_cwd(cwd),
                timeout=effective_timeout,
                env=effective_env,
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError(self.claude_binary)
        except subprocess.TimeoutExpired:
            raise ClaudeTimeoutError(effective_timeout or 0)

        if result.returncode != 0:
            raise ClaudeProcessError(
                returncode=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
                cmd=args,
            )

        if opts.output_format == OutputFormat.JSON:
            return ClaudeResponse.from_json(
                result.stdout,
                returncode=result.returncode,
                stderr=result.stderr,
                cmd=args,
            )
        if opts.output_format == OutputFormat.STREAM_JSON:
            events = _parse_stream_events(result.stdout)
            return ClaudeResponse.from_stream(
                events,
                returncode=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
                cmd=args,
            )
        return ClaudeResponse.from_text(
            result.stdout,
            returncode=result.returncode,
            stderr=result.stderr,
            cmd=args,
        )

    def run(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        options: ClaudeOptions | None = None,
        **overrides: Any,
    ) -> str:
        """Convenience: return only the text output."""
        return self.query(
            prompt,
            stdin=stdin,
            cwd=cwd,
            timeout=timeout,
            env=env,
            options=options,
            **overrides,
        ).text

    def stream(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        options: ClaudeOptions | None = None,
        **overrides: Any,
    ) -> Generator[StreamEvent, None, ClaudeResponse]:
        """Stream events from Claude Code in real-time.

        Yields StreamEvent objects. The final ClaudeResponse is available as the
        generator's return value (StopIteration.value).
        """
        # Force stream-json output
        overrides["output_format"] = OutputFormat.STREAM_JSON
        # Streaming is much more useful with partial messages
        if "include_partial_messages" not in overrides:
            overrides["include_partial_messages"] = True

        opts = (options or self.defaults).with_overrides(**overrides)
        args = self._build_args(prompt, opts)

        effective_env = _merge_env(env) if env is not None else self.env

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE if stdin is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._resolve_cwd(cwd),
                env=effective_env,
                bufsize=1,
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError(self.claude_binary)

        # Feed stdin then close
        if stdin is not None and proc.stdin:
            try:
                proc.stdin.write(stdin)
            finally:
                proc.stdin.close()

        # Drain stderr in background to avoid deadlocks on large stderr output
        stderr_chunks: list[str] = []
        stderr_done = threading.Event()

        def _drain_stderr() -> None:
            try:
                if proc.stderr:
                    for line in proc.stderr:
                        stderr_chunks.append(line)
            finally:
                stderr_done.set()

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        events: list[StreamEvent] = []
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Some versions may interleave non-JSON lines; ignore.
                continue

            evt = StreamEvent(raw=data)
            events.append(evt)
            yield evt

        proc.wait()
        stderr_done.wait(timeout=1.0)
        stderr = "".join(stderr_chunks)

        if proc.returncode != 0:
            raise ClaudeProcessError(
                returncode=proc.returncode or 1,
                stderr=stderr,
                stdout="",
                cmd=args,
            )

        return ClaudeResponse.from_stream(
            events,
            returncode=proc.returncode or 0,
            stderr=stderr,
            stdout="",
            cmd=args,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        options: ClaudeOptions | None = None,
        **overrides: Any,
    ) -> Iterator[str]:
        """Convenience iterator yielding only text deltas/fragments."""
        gen = self.stream(
            prompt,
            stdin=stdin,
            cwd=cwd,
            env=env,
            options=options,
            **overrides,
        )
        try:
            while True:
                evt = next(gen)
                if evt.message:
                    yield evt.message
        except StopIteration:
            return

    # ------------------------------------------------------------------
    # Structured outputs
    # ------------------------------------------------------------------

    def query_model(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        json_schema_override: dict[str, Any] | None = None,
        transform_schema: bool = True,
        **overrides: Any,
    ) -> ModelResponse:
        """Query Claude Code and return a populated instance of *model_class*.

        This is the structured-output convenience method. It:
          1) Extracts JSON Schema from model_class (dataclass / Pydantic / TypedDict / annotated class).
          2) Optionally simplifies the schema to avoid unsupported features.
          3) Sends it via `--json-schema` with `--output-format json`.
          4) Extracts the structured output and instantiates model_class.
        """
        original_schema = json_schema_override or extract_schema(model_class)
        schema_sent = simplify_schema_for_claude(original_schema) if transform_schema else original_schema

        # Force JSON output so we can parse structured output reliably
        overrides["output_format"] = OutputFormat.JSON
        overrides["json_schema"] = schema_sent

        response = self.query(
            prompt,
            stdin=stdin,
            cwd=cwd,
            timeout=timeout,
            env=env,
            **overrides,
        )

        data_obj = _extract_structured_data(response)
        data = _instantiate(data_obj, model_class)

        return ModelResponse(
            data=data,
            raw=response,
            schema_sent=schema_sent,
            schema_original=original_schema if transform_schema else None,
        )

    def query_model_text(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> Any:
        """Shorthand returning just the deserialized object."""
        return self.query_model(
            prompt,
            model_class,
            stdin=stdin,
            cwd=cwd,
            timeout=timeout,
            env=env,
            **overrides,
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
                env=self.env,
            )
            return result.stdout.strip()
        except FileNotFoundError:
            raise ClaudeNotFoundError(self.claude_binary)

    def is_available(self) -> bool:
        """Check whether the `claude` binary is reachable."""
        try:
            _ = self.version()
            return True
        except (ClaudeCodeError, subprocess.SubprocessError):
            return False

    def __repr__(self) -> str:
        parts = ["ClaudeCode("]
        parts.append(f"claude_binary={self.claude_binary!r}, ")
        if self.defaults.model:
            parts.append(f"model={self.defaults.model!r}, ")
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


def _extract_structured_data(response: ClaudeResponse) -> Any:
    """Extract validated structured output from a ClaudeResponse.

    Handles multiple CLI output shapes across versions:
      - Newer: response.json["structured_output"]
      - Older: response.json["result"] is a JSON string
      - Fallback: response.text is JSON
    """
    # 1) Preferred: parsed structured_output (already extracted by ClaudeResponse.from_json)
    if response.structured_output is not None:
        return response.structured_output

    payload = response.json
    if isinstance(payload, dict):
        if payload.get("structured_output") is not None:
            return payload["structured_output"]

        # When --json-schema is used, some versions put JSON in "result" as a string
        if "result" in payload:
            result = payload["result"]
            if isinstance(result, dict) or isinstance(result, list):
                return result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass

        # Some versions may embed output under "output" or similar
        for key in ("output", "data"):
            if key in payload and isinstance(payload[key], (dict, list)):
                return payload[key]

    # Fallback to parsing text
    if response.text:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            pass

    raise ClaudeStructuredOutputError(
        "Could not extract structured output from Claude response.",
        stdout=response.stdout,
        stderr=response.stderr,
        returncode=response.returncode,
        cmd=response.cmd or [],
    )


# --------------------------------------------------------------------------
# Deserialization / instantiation
# --------------------------------------------------------------------------

def _instantiate(data: Any, cls: type) -> Any:
    """Recursively instantiate a class from JSON data (dict)."""
    if not isinstance(data, dict):
        raise ClaudeStructuredOutputError(
            f"Expected a JSON object for structured output, got {type(data).__name__}.",
            stdout=json.dumps(data, ensure_ascii=False)[:500],
        )
    return _instantiate_dict(data, cls)


def _instantiate_dict(data: dict[str, Any], cls: type) -> Any:
    """Instantiate dataclasses, Pydantic models, TypedDicts, and annotated classes."""
    import dataclasses as dc
    import inspect
    from typing import get_type_hints

    from claude_code_cli.schema import is_pydantic_model, is_dataclass, is_typed_dict

    # --- Pydantic ---
    if is_pydantic_model(cls):
        # v2: model_validate, v1: parse_obj / constructor
        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)  # type: ignore[attr-defined]
        if hasattr(cls, "parse_obj"):
            return cls.parse_obj(data)  # type: ignore[attr-defined]
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
        return coerced

    # --- Plain annotated class ---
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}

    # Best-effort: set attributes without calling __init__
    try:
        instance = cls.__new__(cls)
        for key, value in data.items():
            if key in hints:
                setattr(instance, key, _coerce_value(value, hints[key]))
            else:
                setattr(instance, key, value)
        return instance
    except Exception:
        # Last resort: try calling constructor
        return cls(**data)


def _coerce_value(value: Any, target_type: Any) -> Any:
    """Recursively coerce JSON values into target Python types (best-effort)."""
    import inspect
    from typing import Union, get_args, get_origin

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

    # Annotated[X, ...] -> X
    if origin is not None and _is_annotated(target_type):
        return _coerce_value(value, args[0])

    # Optional / Union
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(value, non_none[0])
        return value

    # list[X]
    if origin in (list, tuple) and args and isinstance(value, list):
        return [_coerce_value(item, args[0]) for item in value]

    # dict[K, V]
    if origin is dict and args and len(args) == 2 and isinstance(value, dict):
        return {k: _coerce_value(v, args[1]) for k, v in value.items()}

    # Nested models
    if (
        inspect.isclass(target_type)
        and isinstance(value, dict)
        and (
            is_dataclass(target_type)
            or is_typed_dict(target_type)
            or is_pydantic_model(target_type)
            or hasattr(target_type, "__annotations__")
        )
    ):
        return _instantiate_dict(value, target_type)

    # Enum
    if inspect.isclass(target_type) and issubclass(target_type, __import__("enum").Enum):
        try:
            return target_type(value)
        except Exception:
            return value

    return value
