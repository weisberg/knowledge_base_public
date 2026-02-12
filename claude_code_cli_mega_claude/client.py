"""Core client for interacting with the Claude Code CLI via ``claude -p``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Generator, Iterator, Mapping

from claude_code_cli.errors import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeOptionError,
    ClaudeProcessError,
    ClaudeResponseParseError,
    ClaudeTimeoutError,
)
from claude_code_cli.models import (
    ClaudeResponse,
    ModelResponse,
    StreamEvent,
)
from claude_code_cli.options import (
    Agent,
    ClaudeOptions,
    InputFormat,
    OutputFormat,
    PermissionMode,
    ToolSet,
)
from claude_code_cli.schema import extract_schema


# ======================================================================
# ClaudeCode — primary entry point
# ======================================================================

class ClaudeCode:
    """High-level Python wrapper around the Claude Code CLI (``claude -p``).

    Supports **two configuration styles** that can be mixed freely:

    **1) Flat keyword arguments** (convenient for simple use)::

        client = ClaudeCode(model="sonnet", max_turns=5)
        response = client.query("Explain this code")

    **2) Explicit ClaudeOptions** (full control, reusable, inspectable)::

        opts = ClaudeOptions(model="sonnet", max_turns=5, tools=["Read", "Bash"])
        client = ClaudeCode()
        response = client.query("Explain this code", options=opts)

    Per-call ``**kwargs`` override instance defaults, and ``options=``
    overrides everything.

    Parameters
    ----------
    cli_path : str
        Path or name of the ``claude`` binary (default: auto-detect via PATH).
    cwd : str | Path, optional
        Default working directory for all CLI invocations.
    env : dict, optional
        Extra environment variables merged with ``os.environ`` for each call.
    timeout : float, optional
        Default timeout in seconds for non-streaming calls.
    default_options : ClaudeOptions, optional
        Base options applied to every call (lowest priority).
    **kwargs
        Any field name from :class:`ClaudeOptions` — sets instance-level
        defaults (e.g. ``model="sonnet"``, ``verbose=True``).
    """

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        default_options: ClaudeOptions | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve the binary path
        if cli_path:
            self._cli_path = cli_path
        else:
            found = shutil.which("claude")
            if found is None:
                raise ClaudeNotFoundError()
            self._cli_path = found

        self._cwd = str(cwd) if cwd else None
        self._env = dict(env) if env else None
        self._timeout = timeout

        # Build the base options: explicit default_options ← kwargs overlay
        if default_options is not None:
            self._base_options = default_options
        else:
            self._base_options = ClaudeOptions()

        # Apply any flat kwargs onto the base options
        if kwargs:
            self._base_options = _apply_kwargs(self._base_options, kwargs)

    # ==================================================================
    # Option resolution
    # ==================================================================

    def _resolve_options(
        self,
        options: ClaudeOptions | None,
        kwargs: dict[str, Any],
    ) -> ClaudeOptions:
        """Merge base → kwargs → explicit options."""
        effective = ClaudeOptions(**self._base_options.__dict__)

        # Layer on per-call kwargs
        if kwargs:
            effective = _apply_kwargs(effective, kwargs)

        # Layer on explicit options (highest priority)
        if options is not None:
            effective = effective.merge(options)

        return effective

    def _merged_env(self) -> dict[str, str] | None:
        if self._env is None:
            return None
        env = os.environ.copy()
        env.update(self._env)
        return env

    # ==================================================================
    # Command builder (inspectable)
    # ==================================================================

    def build_command(
        self,
        prompt: str,
        *,
        options: ClaudeOptions | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Build the full ``claude -p ...`` argv without executing it.

        Useful for logging, debugging, or running the command yourself.
        """
        opts = self._resolve_options(options, kwargs)
        argv = [self._cli_path]
        argv.extend(opts.to_args())
        argv.extend(["-p", prompt])
        return argv

    # ==================================================================
    # query() — synchronous, full response
    # ==================================================================

    def query(
        self,
        prompt: str,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        check: bool = True,
        encoding: str = "utf-8",
        options: ClaudeOptions | None = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """Send a prompt and return the full response.

        Parameters
        ----------
        prompt : str
            The user prompt.
        stdin : str | bytes, optional
            Data piped to Claude's stdin (like ``cat file | claude -p``).
        cwd : str | Path, optional
            Working directory override for this call.
        timeout : float, optional
            Timeout override for this call.
        check : bool
            If True (default), raise :class:`ClaudeProcessError` on non-zero exit.
        encoding : str
            Encoding for bytes stdin (default ``utf-8``).
        options : ClaudeOptions, optional
            Explicit options (highest priority).
        **kwargs
            Per-call option overrides (e.g. ``model="opus"``).

        Returns
        -------
        ClaudeResponse
        """
        opts = self._resolve_options(options, kwargs)
        argv = [self._cli_path]
        argv.extend(opts.to_args())
        argv.extend(["-p", prompt])

        # Prepare stdin
        in_text: str | None = None
        if stdin is not None:
            in_text = (
                stdin.decode(encoding, errors="replace")
                if isinstance(stdin, (bytes, bytearray))
                else str(stdin)
            )

        effective_cwd = str(cwd) if cwd else self._cwd
        effective_timeout = timeout if timeout is not None else self._timeout

        try:
            result = subprocess.run(
                argv,
                input=in_text,
                capture_output=True,
                text=True,
                cwd=effective_cwd,
                env=self._merged_env(),
                timeout=effective_timeout,
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError(self._cli_path)
        except subprocess.TimeoutExpired:
            raise ClaudeTimeoutError(effective_timeout or 0)

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if check and result.returncode != 0:
            raise ClaudeProcessError(
                returncode=result.returncode,
                stderr=stderr,
                cmd=argv,
                stdout=stdout,
            )

        # Parse based on output format
        fmt = opts.output_format
        if fmt == OutputFormat.JSON:
            return ClaudeResponse.from_json(stdout, result.returncode, stderr, argv)
        elif fmt == OutputFormat.STREAM_JSON:
            events = _parse_stream_lines(stdout)
            return ClaudeResponse.from_stream(events, result.returncode, stderr, argv)
        else:
            return ClaudeResponse.from_text(stdout, result.returncode, stderr, argv)

    # ==================================================================
    # run() — shorthand, returns text only
    # ==================================================================

    def run(
        self,
        prompt: str,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience — returns just the text output.

        Equivalent to ``client.query(...).text``.
        """
        return self.query(
            prompt, stdin=stdin, cwd=cwd, timeout=timeout, **kwargs
        ).text

    # ==================================================================
    # stream() — real-time event streaming
    # ==================================================================

    def stream(
        self,
        prompt: str,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        include_partial_messages: bool = True,
        check: bool = True,
        encoding: str = "utf-8",
        options: ClaudeOptions | None = None,
        **kwargs: Any,
    ) -> Generator[StreamEvent, None, ClaudeResponse]:
        """Stream events from Claude Code in real-time.

        Yields :class:`StreamEvent` objects. The final :class:`ClaudeResponse`
        is the generator's return value (access via ``StopIteration.value``).

        Usage::

            gen = client.stream("Refactor this file")
            try:
                while True:
                    event = next(gen)
                    print(event.message, end="", flush=True)
            except StopIteration as e:
                response = e.value

        Parameters
        ----------
        prompt, stdin, cwd, encoding
            Same as :meth:`query`.
        include_partial_messages : bool
            Include partial streaming events (default True for streaming).
        check : bool
            Raise on non-zero exit.
        options : ClaudeOptions, optional
            Explicit options.
        **kwargs
            Per-call option overrides.
        """
        # Force stream-json output
        kwargs["output_format"] = OutputFormat.STREAM_JSON
        kwargs["include_partial_messages"] = include_partial_messages

        opts = self._resolve_options(options, kwargs)
        argv = [self._cli_path]
        argv.extend(opts.to_args())
        argv.extend(["-p", prompt])

        effective_cwd = str(cwd) if cwd else self._cwd

        # Prepare stdin bytes
        in_bytes: bytes | None = None
        if stdin is not None:
            in_bytes = (
                stdin
                if isinstance(stdin, (bytes, bytearray))
                else str(stdin).encode(encoding, errors="replace")
            )

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE if in_bytes is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=effective_cwd,
                env=self._merged_env(),
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError(self._cli_path)

        assert proc.stdout is not None
        assert proc.stderr is not None

        # Drain stderr in a background thread to prevent pipe deadlocks
        stderr_chunks: list[bytes] = []

        def _drain_stderr(pipe: Any) -> None:
            try:
                for chunk in iter(lambda: pipe.read(4096), b""):
                    if chunk:
                        stderr_chunks.append(chunk)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
        stderr_thread.start()

        # Write stdin and close
        try:
            if in_bytes is not None and proc.stdin is not None:
                proc.stdin.write(in_bytes)
            if proc.stdin is not None:
                proc.stdin.close()
        except BrokenPipeError:
            pass

        events: list[StreamEvent] = []
        idx = 0

        try:
            for raw_line in proc.stdout:
                line = raw_line.decode(encoding, errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = StreamEvent(raw=data, index=idx)
                    events.append(event)
                    idx += 1
                    yield event
                except json.JSONDecodeError:
                    continue

            proc.wait()
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

        stderr_thread.join(timeout=0.5)
        stderr_text = b"".join(stderr_chunks).decode(encoding, errors="replace")

        if check and proc.returncode not in (0, None):
            raise ClaudeProcessError(
                returncode=proc.returncode or 1,
                stderr=stderr_text,
                cmd=argv,
            )

        return ClaudeResponse.from_stream(events, proc.returncode or 0, stderr_text, argv)

    def stream_text(
        self,
        prompt: str,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Convenience iterator that yields only text fragments.

        Usage::

            for chunk in client.stream_text("Explain this code"):
                print(chunk, end="", flush=True)
        """
        gen = self.stream(prompt, stdin=stdin, cwd=cwd, **kwargs)
        try:
            while True:
                event = next(gen)
                if event.message:
                    yield event.message
        except StopIteration:
            return

    # ==================================================================
    # query_model() — structured output with Python types
    # ==================================================================

    def query_model(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        json_schema_override: dict[str, Any] | None = None,
        options: ClaudeOptions | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Query Claude and return a deserialized instance of *model_class*.

        1. Extracts a JSON Schema from *model_class* (dataclass, Pydantic,
           TypedDict, or annotated class).
        2. Passes it via ``--json-schema``.
        3. Deserializes the JSON response into a populated instance.

        Parameters
        ----------
        prompt : str
            The user prompt.
        model_class : type
            Target Python class (dataclass, Pydantic BaseModel, TypedDict, etc.).
        json_schema_override : dict, optional
            Provide an explicit JSON Schema instead of auto-extracting.
        options : ClaudeOptions, optional
            Explicit options.
        **kwargs
            Per-call option overrides.

        Returns
        -------
        ModelResponse[T]
            ``.data`` is the deserialized instance, ``.raw`` is the full response.
        """
        schema = json_schema_override or extract_schema(model_class)

        kwargs["output_format"] = OutputFormat.JSON
        kwargs["json_schema"] = schema

        response = self.query(
            prompt, stdin=stdin, cwd=cwd, timeout=timeout, options=options, **kwargs
        )

        data = _deserialize(response, model_class)
        return ModelResponse(data=data, raw=response, schema=schema)

    def query_model_text(
        self,
        prompt: str,
        model_class: type,
        *,
        stdin: str | bytes | None = None,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Shorthand — returns just the deserialized object.

        Equivalent to ``client.query_model(...).data``.
        """
        return self.query_model(
            prompt, model_class, stdin=stdin, cwd=cwd, timeout=timeout, **kwargs
        ).data

    # ==================================================================
    # Utilities
    # ==================================================================

    def version(self) -> str:
        """Return the installed Claude Code CLI version string."""
        try:
            result = subprocess.run(
                [self._cli_path, "--version"],
                capture_output=True,
                text=True,
                cwd=self._cwd,
                env=self._merged_env(),
                timeout=10,
            )
        except FileNotFoundError:
            raise ClaudeNotFoundError(self._cli_path)
        if result.returncode != 0:
            raise ClaudeProcessError(
                returncode=result.returncode,
                stderr=result.stderr or "",
                cmd=[self._cli_path, "--version"],
                stdout=result.stdout or "",
            )
        return (result.stdout or "").strip()

    def is_available(self) -> bool:
        """Check whether the ``claude`` binary is reachable."""
        try:
            self.version()
            return True
        except (ClaudeCodeError, subprocess.SubprocessError):
            return False

    @property
    def cli_path(self) -> str:
        """The resolved path to the ``claude`` binary."""
        return self._cli_path

    def __repr__(self) -> str:
        parts = [f"ClaudeCode(cli_path={self._cli_path!r}"]
        if self._base_options.model:
            parts.append(f", model={self._base_options.model!r}")
        if self._cwd:
            parts.append(f", cwd={self._cwd!r}")
        parts.append(")")
        return "".join(parts)


# ======================================================================
# Module-level convenience API
# ======================================================================

def ask(
    prompt: str,
    *,
    stdin: str | bytes | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> ClaudeResponse:
    """Zero-config one-liner — uses a default client.

    ::

        import claude_code_cli as claude
        print(claude.ask("What is 2+2?").text)
    """
    client = ClaudeCode()
    return client.query(prompt, stdin=stdin, timeout=timeout, **kwargs)


# ======================================================================
# Internal helpers
# ======================================================================

def _apply_kwargs(opts: ClaudeOptions, kwargs: dict[str, Any]) -> ClaudeOptions:
    """Apply flat keyword arguments to a :class:`ClaudeOptions` instance.

    Returns a new instance.  Unknown keys raise :class:`ClaudeOptionError`.
    """
    new = ClaudeOptions(**opts.__dict__)
    valid_fields = set(ClaudeOptions.__dataclass_fields__)

    for key, value in kwargs.items():
        if key in valid_fields:
            setattr(new, key, value)
        else:
            raise ClaudeOptionError(
                f"Unknown option {key!r}. Valid options: {sorted(valid_fields)}"
            )
    return new


def _parse_stream_lines(text: str) -> list[StreamEvent]:
    """Parse newline-delimited JSON into StreamEvent objects."""
    events: list[StreamEvent] = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(StreamEvent(raw=json.loads(line), index=i))
        except json.JSONDecodeError:
            continue
    return events


# ======================================================================
# Deserialization (for query_model)
# ======================================================================

def _deserialize(response: ClaudeResponse, model_class: type) -> Any:
    """Deserialize a :class:`ClaudeResponse` into an instance of *model_class*."""
    from claude_code_cli.schema import is_pydantic_model, is_dataclass, is_typed_dict

    data = _extract_data_dict(response)

    if not isinstance(data, dict):
        raise ClaudeResponseParseError(
            f"Expected a JSON object from Claude, got {type(data).__name__}: {str(data)[:200]}",
            raw_text=response.stdout,
        )

    return _instantiate(data, model_class)


def _extract_data_dict(response: ClaudeResponse) -> Any:
    """Pull the structured data from a ClaudeResponse.

    The JSON envelope from ``claude -p --output-format json --json-schema``
    has the structured output in either ``structured_output`` or ``result``.
    """
    # Prefer the explicit structured_output field
    if response.structured_output is not None:
        if isinstance(response.structured_output, dict):
            return response.structured_output
        if isinstance(response.structured_output, str):
            try:
                return json.loads(response.structured_output)
            except json.JSONDecodeError:
                pass

    # Fall back to parsing the "result" field in the JSON envelope
    if response.json is not None:
        payload = response.json
        if isinstance(payload, dict) and "result" in payload:
            result = payload["result"]
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass
            if isinstance(result, dict):
                return result
        return payload

    # Last resort: parse the text
    if response.text:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            pass

    raise ClaudeResponseParseError(
        "Could not extract structured data from Claude response.",
        raw_text=response.stdout,
    )


def _instantiate(data: dict[str, Any], cls: type) -> Any:
    """Recursively instantiate a class from a data dict."""
    import dataclasses as dc
    from typing import get_type_hints

    from claude_code_cli.schema import is_pydantic_model, is_dataclass, is_typed_dict

    # Pydantic
    if is_pydantic_model(cls):
        try:
            if hasattr(cls, "model_validate"):
                return cls.model_validate(data)
            return cls(**data)
        except Exception:
            return cls(**data)

    # dataclass
    if is_dataclass(cls):
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            hints = {}
        fields = {f.name for f in dc.fields(cls)}
        built: dict[str, Any] = {}
        for key, value in data.items():
            if key in fields:
                built[key] = _coerce_value(value, hints.get(key)) if key in hints else value
        return cls(**built)

    # TypedDict
    if is_typed_dict(cls):
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            hints = {}
        return {
            k: (_coerce_value(v, hints[k]) if k in hints else v)
            for k, v in data.items()
        }

    # Plain annotated class
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}

    try:
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, _coerce_value(value, hints.get(key)) if key in hints else value)
        return instance
    except Exception:
        return cls(**data)


def _coerce_value(value: Any, target_type: Any) -> Any:
    """Recursively coerce a JSON value into the target Python type."""
    import inspect
    from typing import get_origin, get_args, Union

    from claude_code_cli.schema import (
        is_pydantic_model,
        is_dataclass,
        is_typed_dict,
        _is_annotated,
    )

    if value is None or target_type is None:
        return value

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Annotated[X, ...] → unwrap
    if origin is not None and _is_annotated(target_type):
        return _coerce_value(value, args[0])

    # Optional[X] / Union[X, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(value, non_none[0])
        return value

    # list[X]
    if origin is list and args and isinstance(value, list):
        return [_coerce_value(item, args[0]) for item in value]

    # dict[K, V]
    if origin is dict and args and len(args) == 2 and isinstance(value, dict):
        return {k: _coerce_value(v, args[1]) for k, v in value.items()}

    # Nested model
    if (
        inspect.isclass(target_type)
        and isinstance(value, dict)
        and (is_dataclass(target_type) or is_typed_dict(target_type)
             or is_pydantic_model(target_type) or hasattr(target_type, "__annotations__"))
    ):
        return _instantiate(value, target_type)

    # Enum
    if inspect.isclass(target_type) and issubclass(target_type, __import__("enum").Enum):
        try:
            return target_type(value)
        except (ValueError, KeyError):
            return value

    return value
