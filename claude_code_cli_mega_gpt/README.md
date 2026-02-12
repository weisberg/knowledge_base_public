# claude-code-cli (mega edition)

A Python wrapper around the **Claude Code CLI** using **print/pipe mode** (`claude -p`) so you can:
- pipe data in via stdin,
- pass a prompt as an argument,
- and parse replies (text, JSON, or streaming JSON).

This package is designed to be imported exactly like:

```python
import claude_code_cli as claude
```

It targets the official CLI flags documented in the Claude Code docs. citeturn1view0

## Install (local dev)

```bash
pip install -e .
```

## Quick start

```python
import claude_code_cli as claude

client = claude.ClaudeCode()

resp = client.query("Summarize this piped text.", stdin="hello\nworld\n")
print(resp.text)

# Want the raw JSON envelope (cost, session id, etc.)?
resp = client.query(
    "Summarize this piped text.",
    stdin="hello\nworld\n",
    output_format=claude.OutputFormat.JSON,
)
print(resp.session_id, resp.cost_usd, resp.text)
```

## Streaming output (stream-json)

Claude Code supports `--output-format stream-json` in print mode. citeturn1view0

```python
import claude_code_cli as claude

client = claude.ClaudeCode()

for chunk in client.stream_text("Explain what this code does.", stdin="def f():\n  return 1\n"):
    print(chunk, end="", flush=True)
```

Under the hood, `stream_text()` uses `--include-partial-messages`, which the CLI docs note requires print mode and `--output-format=stream-json`. citeturn1view0

## Structured outputs (validated JSON) via Python classes

Claude Code supports **structured outputs** in print mode with `--json-schema`. citeturn1view0turn6view0  
In structured output mode, the agent’s final message includes a `structured_output` field containing validated data that matches your schema. citeturn6view0

This wrapper lets you define a Python model and get back a typed object.

### Pydantic (best runtime validation)

```python
from pydantic import BaseModel
import claude_code_cli as claude

class Contact(BaseModel):
    name: str
    email: str

client = claude.ClaudeCode()
result = client.query_model(
    "Extract name/email from the text.",
    Contact,
    stdin="Jane Doe <jane@example.com>",
)

print(result.data.name, result.data.email)
print(result.raw.structured_output)  # the underlying validated dict
```

### Dataclasses / TypedDict / annotated classes

Dataclasses and TypedDicts also work:

```python
from dataclasses import dataclass
import claude_code_cli as claude

@dataclass
class Summary:
    title: str
    bullets: list[str]

client = claude.ClaudeCode()
res = client.query_model("Summarize this:", Summary, stdin="...")
print(res.data)
```

### Schema simplification (optional but recommended)

Anthropic’s structured output docs describe how SDKs may **simplify schemas** before sending them:
- remove unsupported constraints (like `minimum`, `maximum`, `minLength`, `maxLength`)
- add `additionalProperties: false`
- filter unsupported string formats
…and then validate the response against the original schema. citeturn5view0

This wrapper includes a lightweight version of that transform via:

```python
from claude_code_cli.schema import simplify_schema_for_claude
```

`query_model()` applies this transform by default (`transform_schema=True`) to reduce schema rejection errors.

## CLI flags coverage

This wrapper models the most useful flags for `-p` mode, including:
- `--model`, `--fallback-model` (print-mode only) citeturn1view0
- `--output-format` (`text`, `json`, `stream-json`) citeturn1view0
- `--input-format` (`text`, `stream-json`) citeturn1view0
- `--json-schema` (structured outputs) citeturn1view0
- prompt customization flags (system prompt / append prompt; file variants) citeturn1view0
- permissions flags + modes citeturn1view0turn2view0
- `--tools`, `--allowedTools`, `--disallowedTools` citeturn1view0
- `--mcp-config`, `--strict-mcp-config` citeturn1view0
- `--settings`, `--setting-sources`, `--plugin-dir` citeturn1view0

Not every interactive-only flag is useful in `-p` mode, but the `ClaudeOptions.extra_args` escape hatch lets you pass a flag the wrapper doesn't yet model.

## License

MIT
