# claude_code_cli

A Python wrapper for the [Claude Code CLI](https://code.claude.com/docs/en/cli-reference), providing a clean object-oriented interface to `claude -p` (print/pipe mode).

## Requirements

- Python 3.10+
- [Claude Code CLI](https://code.claude.com/docs/en/quickstart) installed and on your PATH

## Installation

```bash
pip install .          # from source
# or just copy the claude_code_cli/ package into your project
```

## Quick Start

```python
import claude_code_cli as claude

client = claude.ClaudeCode()
response = client.query("Explain what this project does")
print(response.text)
```

## Usage

### Basic Query

```python
client = claude.ClaudeCode(model="sonnet")

# Full response object (JSON output by default)
response = client.query("What does this function do?")
print(response.text)         # extracted text result
print(response.session_id)   # session ID for resuming
print(response.cost_usd)     # cost of the call
print(response.num_turns)    # agentic turns used

# Shorthand — just get the text
text = client.run("Summarize README.md")
```

### Piping Data via stdin

```python
# Equivalent to: cat myfile.py | claude -p "Review this code"
with open("myfile.py") as f:
    code = f.read()

response = client.query("Review this code for bugs", stdin=code)
```

### Working Directory

```python
# Run Claude in the context of a specific project
client = claude.ClaudeCode(cwd="/path/to/my-project")
response = client.query("Find and fix type errors")

# Or per-call override
response = client.query("Explain the architecture", cwd="/other/project")
```

### Streaming

```python
# Real-time streaming with full event objects
gen = client.stream("Write unit tests for auth.py")
try:
    while True:
        event = next(gen)
        if event.message:
            print(event.message, end="", flush=True)
except StopIteration as e:
    response = e.value  # ClaudeResponse with final stats
    print(f"\nCost: ${response.cost_usd}")

# Simple text-only streaming
for chunk in client.stream_text("Refactor the database layer"):
    print(chunk, end="", flush=True)
```

### Conversation Continuity

```python
# Continue the most recent conversation
response = client.query("Now add error handling", continue_conversation=True)

# Resume a specific session by ID
response = client.query(
    "What about edge cases?",
    resume="550e8400-e29b-41d4-a716-446655440000",
)

# Resume but fork into a new session
response = client.query(
    "Try a different approach",
    resume="550e8400-e29b-41d4-a716-446655440000",
    fork_session=True,
)

# Use a specific session ID
response = client.query(
    "Continue here",
    session_id="550e8400-e29b-41d4-a716-446655440000",
)
```

### Structured Output (JSON Schema)

```python
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "string"}
        },
        "severity": {"type": "string", "enum": ["low", "medium", "high"]}
    },
    "required": ["summary", "issues", "severity"]
}

response = client.query("Analyze this codebase for security issues", json_schema=schema)
print(response.json)  # Parsed dict matching the schema
```

### Structured Output with Python Classes (`query_model`)

Instead of hand-writing JSON Schemas, define a Python class and let
`query_model()` handle schema extraction, CLI invocation, and deserialization
automatically.  Supports **dataclasses**, **Pydantic models**, **TypedDicts**,
and plain annotated classes.

#### Dataclass

```python
from dataclasses import dataclass

@dataclass
class CodeReview:
    summary: str
    issues: list[str]
    severity: str      # Claude fills this as a string

result = client.query_model("Review auth.py for security issues", CodeReview)
print(result.data.summary)    # str — typed CodeReview instance
print(result.data.issues)     # list[str]
print(result.cost_usd)        # API cost from the response metadata
print(result.session_id)      # session ID for continuing later
```

#### Pydantic Model

```python
from pydantic import BaseModel, Field

class ProjectAnalysis(BaseModel):
    name: str
    language: str
    complexity: int = Field(ge=1, le=10, description="1-10 complexity score")
    suggestions: list[str]
    has_tests: bool

result = client.query_model("Analyze this project", ProjectAnalysis)
analysis = result.data           # fully validated Pydantic instance
print(analysis.complexity)       # int, validated by Pydantic
print(analysis.model_dump())     # dict via Pydantic
```

#### Nested Types

```python
from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Issue:
    line: int
    message: str
    severity: Severity

@dataclass
class AuditResult:
    summary: str
    issues: list[Issue]    # nested dataclass — deserialized recursively
    score: float

result = client.query_model("Audit this codebase for security", AuditResult)
for issue in result.data.issues:
    print(f"  Line {issue.line}: {issue.message} [{issue.severity.value}]")
```

#### TypedDict

```python
from typing import TypedDict

class TodoItem(TypedDict, total=False):
    text: str
    file: str
    line: int
    author: str

result = client.query_model("Find all TODO comments", TodoItem)
print(result.data)  # plain dict (TypedDicts are dicts at runtime)
```

#### Field Metadata via `Annotated` + `FieldMeta`

```python
from dataclasses import dataclass
from typing import Annotated, Optional
from claude_code_cli import FieldMeta, schema

@schema(description="Performance benchmark results")
@dataclass
class BenchmarkResult:
    name: str
    duration_ms: Annotated[float, FieldMeta(description="Execution time in ms", minimum=0)]
    passed: bool
    notes: Annotated[Optional[str], FieldMeta(max_length=500)] = None

result = client.query_model("Benchmark the sort functions", BenchmarkResult)
```

The `FieldMeta` annotations are converted to JSON Schema constraints
(`minimum`, `maxLength`, `pattern`, `enum`, etc.) and sent to the CLI.

#### Manual Schema Override

```python
# Auto-extract but tweak before sending
my_schema = claude.extract_schema(CodeReview)
my_schema["properties"]["severity"]["enum"] = ["low", "medium", "high"]

result = client.query_model(
    "Review this code",
    CodeReview,
    json_schema_override=my_schema,
)
```

#### Shorthand

```python
# Returns just the deserialized object (no metadata)
review = client.query_model_text("Review auth.py", CodeReview)
print(review.summary)
```

### System Prompt Customization

```python
# Replace the entire system prompt
client = claude.ClaudeCode(system_prompt="You are a Python testing expert.")

# Append to the default prompt (safer — keeps built-in capabilities)
client = claude.ClaudeCode(append_system_prompt="Always use type hints and docstrings.")

# Load from file (print mode only)
client = claude.ClaudeCode(system_prompt_file="./prompts/reviewer.txt")
```

### Tool Configuration

```python
# Specific tools only
client = claude.ClaudeCode(tools=["Bash", "Read", "Edit"])

# Using the ToolSet helper
client = claude.ClaudeCode(tools=claude.ToolSet.default())
client = claude.ClaudeCode(tools=claude.ToolSet.none())
client = claude.ClaudeCode(tools=claude.ToolSet(["Bash(git log:*)", "Read"]))

# Allow specific tools without prompting
client = claude.ClaudeCode(allowed_tools=["Bash(git log:*)", "Bash(git diff:*)", "Read"])

# Disallow specific tools
client = claude.ClaudeCode(disallowed_tools=["Edit"])
```

### Custom Subagents

```python
reviewer = claude.Agent(
    name="code-reviewer",
    description="Expert code reviewer. Use proactively after code changes.",
    prompt="You are a senior code reviewer. Focus on quality, security, best practices.",
    tools=["Read", "Grep", "Glob", "Bash"],
    model="sonnet",
)

debugger = claude.Agent(
    name="debugger",
    description="Debugging specialist for errors and test failures.",
    prompt="You are an expert debugger. Analyze errors and provide fixes.",
)

client = claude.ClaudeCode(agents=[reviewer, debugger])
response = client.query("Review the recent changes and debug any test failures")

# Or select a specific named agent
client = claude.ClaudeCode(agent="my-custom-agent")
```

### MCP Server Configuration

```python
client = claude.ClaudeCode(
    mcp_config="./mcp-servers.json",
    strict_mcp_config=True,  # only use servers from the config file
)
```

### Permission Handling

```python
# Skip all permission prompts (use with caution!)
client = claude.ClaudeCode(dangerously_skip_permissions=True)

# Use a specific permission mode
client = claude.ClaudeCode(permission_mode=claude.PermissionMode.PLAN)

# Use an MCP tool for permission prompts in automation
client = claude.ClaudeCode(permission_prompt_tool="my_mcp_auth_tool")
```

### Model Selection & Fallback

```python
# Use model aliases
client = claude.ClaudeCode(model="opus")
client = claude.ClaudeCode(model="sonnet")

# Full model string
client = claude.ClaudeCode(model="claude-sonnet-4-5-20250929")

# Automatic fallback when primary model is overloaded
client = claude.ClaudeCode(model="opus", fallback_model="sonnet")
```

### Output Formats

```python
# JSON output (default) — parsed automatically
response = client.query("Explain this", output_format=claude.OutputFormat.JSON)
print(response.json)

# Plain text
response = client.query("Explain this", output_format=claude.OutputFormat.TEXT)
print(response.text)

# Stream JSON (collected, not real-time — use .stream() for real-time)
response = client.query("Explain this", output_format=claude.OutputFormat.STREAM_JSON)
for event in response.events:
    print(event.type, event.message)
```

### Error Handling

```python
from claude_code_cli import (
    ClaudeCodeError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeProcessError,
)

try:
    response = client.query("Do something complex", timeout=120)
except ClaudeNotFoundError:
    print("Claude CLI not installed!")
except ClaudeTimeoutError:
    print("Operation timed out")
except ClaudeProcessError as e:
    print(f"CLI exited with code {e.returncode}: {e.stderr}")
except ClaudeCodeError as e:
    print(f"General error: {e}")
```

### Health Check

```python
client = claude.ClaudeCode()

if client.is_available():
    print(f"Claude CLI version: {client.version()}")
else:
    print("Claude CLI is not available")
```

## API Reference

### `ClaudeCode` — Main Client

| Constructor Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Model alias or full name |
| `max_turns` | `int` | Limit agentic turns |
| `system_prompt` | `str` | Replace default system prompt |
| `append_system_prompt` | `str` | Append to default system prompt |
| `system_prompt_file` | `str \| Path` | Load system prompt from file |
| `output_format` | `OutputFormat` | Default output format (default: JSON) |
| `tools` | `ToolSet \| list[str]` | Built-in tools to enable |
| `allowed_tools` | `list[str]` | Tools allowed without prompting |
| `disallowed_tools` | `list[str]` | Tools to disallow |
| `mcp_config` | `str \| Path \| list` | MCP config file path(s) |
| `strict_mcp_config` | `bool` | Only use specified MCP servers |
| `permission_mode` | `PermissionMode` | Permission mode |
| `agents` | `list[Agent]` | Custom subagent definitions |
| `agent` | `str` | Named agent to use |
| `add_dirs` | `list[str \| Path]` | Additional working directories |
| `verbose` | `bool` | Enable verbose logging |
| `debug` | `str \| bool` | Enable debug mode |
| `dangerously_skip_permissions` | `bool` | Skip permission prompts |
| `fallback_model` | `str` | Fallback model alias |
| `cwd` | `str \| Path` | Default working directory |
| `timeout` | `float` | Default timeout (seconds) |
| `claude_binary` | `str` | Explicit path to claude binary |

### `ModelResponse[T]` — Typed Structured Output

Returned by `query_model()`. Wraps a deserialized Python object with response metadata.

| Attribute | Type | Description |
|---|---|---|
| `data` | `T` | The deserialized instance of the target class |
| `raw` | `ClaudeResponse` | The full underlying response |
| `schema` | `dict` | The JSON Schema that was sent to the CLI |
| `session_id` | `str \| None` | Proxy for `raw.session_id` |
| `cost_usd` | `float \| None` | Proxy for `raw.cost_usd` |
| `duration_ms` | `float \| None` | Proxy for `raw.duration_ms` |
| `num_turns` | `int \| None` | Proxy for `raw.num_turns` |

### Schema Utilities

| Export | Description |
|---|---|
| `extract_schema(cls)` | Derive a JSON Schema dict from a dataclass, Pydantic model, TypedDict, or annotated class |
| `@schema(description=..., title=..., examples=...)` | Decorator to attach JSON Schema metadata to a class |
| `FieldMeta(description=..., minimum=..., ...)` | Use with `Annotated[type, FieldMeta(...)]` to add per-field constraints |

### `ClaudeResponse`

| Attribute | Type | Description |
|---|---|---|
| `text` | `str` | Plain-text result |
| `json` | `dict \| list \| None` | Parsed JSON response |
| `events` | `list[StreamEvent]` | Stream events (stream-json mode) |
| `session_id` | `str \| None` | Session ID |
| `cost_usd` | `float \| None` | Total cost in USD |
| `duration_ms` | `float \| None` | Duration in milliseconds |
| `num_turns` | `int \| None` | Number of agentic turns |
| `is_error` | `bool` | Whether the response is an error |
| `returncode` | `int` | Process return code |
| `stderr` | `str` | Stderr output |

### `StreamEvent`

| Property | Type | Description |
|---|---|---|
| `type` | `str` | Event type |
| `subtype` | `str` | Event subtype |
| `message` | `str` | Extracted text content |
| `session_id` | `str \| None` | Session ID (result events) |
| `cost_usd` | `float \| None` | Cost (result events) |
| `is_error` | `bool` | Error flag |
| `raw` | `dict` | The full raw JSON dict |

## License

MIT
