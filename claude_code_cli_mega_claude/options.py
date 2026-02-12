"""
CLI options, agent definitions, and tool helpers.

:class:`ClaudeOptions` is a typed, validating representation of every CLI flag
relevant to ``claude -p`` (print mode).  It can be built independently,
inspected, merged, and serialized — then handed to :class:`Client` methods.

For convenience, :class:`ClaudeCode` also accepts flat keyword arguments
that are converted to :class:`ClaudeOptions` internally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from claude_code_cli.errors import ClaudeOptionError


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """``--output-format`` values."""
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class InputFormat(str, Enum):
    """``--input-format`` values."""
    TEXT = "text"
    STREAM_JSON = "stream-json"


class PermissionMode(str, Enum):
    """``--permission-mode`` values.

    See https://code.claude.com/docs/en/permissions.
    """
    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "acceptEdits"
    DELEGATE = "delegate"
    DONT_ASK = "dontAsk"
    BYPASS_PERMISSIONS = "bypassPermissions"


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Defines a custom subagent for the ``--agents`` flag.

    Example::

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
        d: dict[str, Any] = {"description": self.description, "prompt": self.prompt}
        if self.tools is not None:
            d["tools"] = self.tools
        if self.model is not None:
            d["model"] = self.model
        return d


# ---------------------------------------------------------------------------
# ToolSet helper
# ---------------------------------------------------------------------------

@dataclass
class ToolSet:
    """Convenience wrapper for specifying tools.

    Examples::

        ToolSet.default()                          # all default tools
        ToolSet.none()                             # no tools
        ToolSet(["Bash", "Read"])                  # specific tools
        ToolSet(["Bash(git log:*)", "Read"])       # tools with patterns
    """
    tools: list[str]

    @classmethod
    def default(cls) -> ToolSet:
        return cls(["default"])

    @classmethod
    def none(cls) -> ToolSet:
        return cls([""])

    def to_cli_value(self) -> str:
        """Return the comma-joined string for ``--tools``."""
        return ",".join(self.tools)


# ---------------------------------------------------------------------------
# ClaudeOptions
# ---------------------------------------------------------------------------

@dataclass
class ClaudeOptions:
    """Typed, validating representation of ``claude -p`` CLI flags.

    Every field maps 1:1 to a CLI flag.  Build an instance, optionally
    call :meth:`validate`, then pass it to :meth:`Client.query` via
    ``options=``.

    Unknown / future flags can be passed via :attr:`extra_args`.
    """

    # -- Model ----------------------------------------------------------------
    model: str | None = None
    fallback_model: str | None = None

    # -- Output / input format ------------------------------------------------
    output_format: OutputFormat = OutputFormat.JSON
    input_format: InputFormat | None = None
    include_partial_messages: bool = False
    json_schema: dict[str, Any] | str | None = None

    # -- System prompt --------------------------------------------------------
    system_prompt: str | None = None
    system_prompt_file: str | Path | None = None
    append_system_prompt: str | None = None

    # -- Budget / turns -------------------------------------------------------
    max_turns: int | None = None
    max_budget_usd: float | None = None

    # -- Tools ----------------------------------------------------------------
    tools: ToolSet | list[str] | str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)

    # -- Permissions ----------------------------------------------------------
    permission_mode: PermissionMode | str | None = None
    permission_prompt_tool: str | None = None
    dangerously_skip_permissions: bool = False

    # -- Agents ---------------------------------------------------------------
    agents: list[Agent] | dict[str, Any] | str | None = None
    agent: str | None = None

    # -- Session / conversation -----------------------------------------------
    continue_conversation: bool = False
    resume: str | None = None
    session_id: str | None = None
    fork_session: bool = False

    # -- Directories ----------------------------------------------------------
    add_dirs: list[str | Path] = field(default_factory=list)

    # -- MCP / plugins / settings ---------------------------------------------
    mcp_config: list[str | Path] | str | Path | None = None
    strict_mcp_config: bool = False
    plugin_dirs: list[str | Path] = field(default_factory=list)
    settings: str | Path | None = None
    setting_sources: list[str] | str | None = None

    # -- Logging / debug ------------------------------------------------------
    verbose: bool = False
    debug: str | bool | None = None
    betas: list[str] = field(default_factory=list)

    # -- Escape hatch ---------------------------------------------------------
    extra_args: list[str] = field(default_factory=list)

    # =================================================================
    # Validation
    # =================================================================

    def validate(self, *, strict: bool = False) -> None:
        """Check for known invalid flag combinations.

        Parameters
        ----------
        strict : bool
            If True, also raise on flags that are likely no-ops in print mode
            (e.g. interactive-only flags).
        """
        if self.system_prompt and self.system_prompt_file:
            raise ClaudeOptionError(
                "--system-prompt and --system-prompt-file are mutually exclusive"
            )
        if self.include_partial_messages and self.output_format != OutputFormat.STREAM_JSON:
            raise ClaudeOptionError(
                "--include-partial-messages requires output_format=STREAM_JSON"
            )
        if self.continue_conversation and self.resume:
            raise ClaudeOptionError(
                "--continue and --resume are mutually exclusive"
            )
        if self.fork_session and not (self.continue_conversation or self.resume):
            raise ClaudeOptionError(
                "--fork-session requires --continue or --resume"
            )

    # =================================================================
    # Argument builder
    # =================================================================

    def to_args(self) -> list[str]:
        """Convert to a CLI argv fragment (excluding the binary, ``-p``, and prompt)."""
        self.validate()
        args: list[str] = []

        # -- Session / conversation (before other flags) ----------------
        if self.continue_conversation:
            args.append("--continue")
        if self.resume:
            args.extend(["--resume", self.resume])
        if self.session_id:
            args.extend(["--session-id", self.session_id])
        if self.fork_session:
            args.append("--fork-session")

        # -- Model ------------------------------------------------------
        if self.model:
            args.extend(["--model", self.model])
        if self.fallback_model:
            args.extend(["--fallback-model", self.fallback_model])

        # -- Output / input format --------------------------------------
        if self.output_format != OutputFormat.TEXT:
            args.extend(["--output-format", self.output_format.value])
        if self.input_format is not None:
            args.extend(["--input-format", self.input_format.value])
        if self.json_schema is not None:
            schema_str = (
                self.json_schema
                if isinstance(self.json_schema, str)
                else json.dumps(self.json_schema)
            )
            args.extend(["--json-schema", schema_str])
        if self.include_partial_messages:
            args.append("--include-partial-messages")

        # -- System prompt ----------------------------------------------
        if self.system_prompt:
            args.extend(["--system-prompt", self.system_prompt])
        if self.system_prompt_file is not None:
            args.extend(["--system-prompt-file", str(self.system_prompt_file)])
        if self.append_system_prompt:
            args.extend(["--append-system-prompt", self.append_system_prompt])

        # -- Budget / turns ---------------------------------------------
        if self.max_turns is not None:
            args.extend(["--max-turns", str(self.max_turns)])
        if self.max_budget_usd is not None:
            args.extend(["--max-budget-usd", f"{self.max_budget_usd:.2f}"])

        # -- Tools ------------------------------------------------------
        if self.tools is not None:
            if isinstance(self.tools, ToolSet):
                args.extend(["--tools", self.tools.to_cli_value()])
            elif isinstance(self.tools, list):
                args.extend(["--tools", ",".join(self.tools)])
            else:
                args.extend(["--tools", str(self.tools)])
        if self.allowed_tools:
            args.extend(["--allowedTools"] + list(self.allowed_tools))
        if self.disallowed_tools:
            args.extend(["--disallowedTools"] + list(self.disallowed_tools))

        # -- Permissions ------------------------------------------------
        if self.permission_mode is not None:
            mode = (
                self.permission_mode.value
                if isinstance(self.permission_mode, PermissionMode)
                else str(self.permission_mode)
            )
            args.extend(["--permission-mode", mode])
        if self.permission_prompt_tool:
            args.extend(["--permission-prompt-tool", self.permission_prompt_tool])
        if self.dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")

        # -- Agents -----------------------------------------------------
        if self.agents is not None:
            if isinstance(self.agents, list):
                agents_dict = {a.name: a.to_dict() for a in self.agents}
                args.extend(["--agents", json.dumps(agents_dict)])
            elif isinstance(self.agents, dict):
                args.extend(["--agents", json.dumps(self.agents)])
            else:
                args.extend(["--agents", str(self.agents)])
        if self.agent:
            args.extend(["--agent", self.agent])

        # -- Directories ------------------------------------------------
        if self.add_dirs:
            args.extend(["--add-dir"] + [str(d) for d in self.add_dirs])

        # -- MCP / plugins / settings -----------------------------------
        if self.mcp_config is not None:
            configs = self.mcp_config if isinstance(self.mcp_config, list) else [self.mcp_config]
            args.extend(["--mcp-config"] + [str(c) for c in configs])
        if self.strict_mcp_config:
            args.append("--strict-mcp-config")
        for pd in self.plugin_dirs:
            args.extend(["--plugin-dir", str(pd)])
        if self.settings is not None:
            args.extend(["--settings", str(self.settings)])
        if self.setting_sources is not None:
            sources = (
                ",".join(self.setting_sources)
                if isinstance(self.setting_sources, list)
                else str(self.setting_sources)
            )
            args.extend(["--setting-sources", sources])

        # -- Logging / debug --------------------------------------------
        if self.verbose:
            args.append("--verbose")
        if self.debug:
            if isinstance(self.debug, bool):
                args.append("--debug")
            else:
                args.extend(["--debug", self.debug])
        if self.betas:
            args.extend(["--betas"] + list(self.betas))

        # -- Escape hatch ----------------------------------------------
        args.extend(self.extra_args)

        return args

    # =================================================================
    # Merge / copy
    # =================================================================

    def merge(self, overrides: ClaudeOptions) -> ClaudeOptions:
        """Return a new :class:`ClaudeOptions` with *overrides* applied on top.

        Only non-default values from *overrides* take effect.
        """
        defaults = ClaudeOptions()
        merged = ClaudeOptions(**self.__dict__)
        for fname, fval in overrides.__dict__.items():
            default_val = getattr(defaults, fname)
            override_val = fval
            # Only apply if the override differs from the dataclass default
            if override_val != default_val:
                setattr(merged, fname, override_val)
        return merged
