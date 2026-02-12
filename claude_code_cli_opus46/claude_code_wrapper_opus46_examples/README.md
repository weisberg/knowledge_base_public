# claude_code_cli Examples

A progressive series of examples from "can the agent answer a question" to
full typed-output pipelines. Each script is self-contained and runnable.

## Prerequisites

```bash
# Claude Code CLI must be installed and on PATH
claude --version

# Install the module (from the parent directory)
pip install .
```

## Examples

| # | File | What it demonstrates |
|---|---|---|
| 01 | `01_hello_world.py` | **Simplest test** — preflight check, ask a question, print the answer |
| 02 | `02_text_mode_and_shorthand.py` | `run()` one-liner, `OutputFormat.TEXT` vs `JSON`, response metadata |
| 03 | `03_piping_stdin.py` | Pipe code/data via `stdin=` (equivalent to `cat file \| claude -p`) |
| 04 | `04_streaming.py` | Real-time streaming with `stream_text()` and `stream()` |
| 05 | `05_system_prompt_and_model.py` | Custom system prompts, model aliases, fallback models |
| 06 | `06_json_schema_raw.py` | Hand-written JSON Schema via `json_schema=` kwarg |
| 07 | `07_query_model_dataclass.py` | **`query_model()`** — auto-schema from dataclass, typed deserialization |
| 08 | `08_enums_and_constraints.py` | Enums, `Literal`, `@schema`, `FieldMeta`, `Annotated` constraints |
| 09 | `09_error_handling.py` | Exception hierarchy, timeouts, graceful fallbacks, max turns |
| 10 | `10_conversation_continuity.py` | `continue_conversation`, `resume`, `fork_session`, `session_id` |
| 11 | `11_tools_and_permissions.py` | `ToolSet`, allowed/disallowed tools, permission modes |
| 12 | `12_subagents.py` | Custom `Agent` definitions for multi-step delegation |
| 13 | `13_codebase_analysis_pipeline.py` | **End-to-end**: analyze a directory → typed `CodebaseReport` |

## Suggested reading order

**If you're new**, start with 01 → 02 → 03 and you'll have the basics.

**For structured output** (the main feature), read 06 → 07 → 08.

**For automation/CI**, read 09 → 11 → 13.

## Running

```bash
# Simple test
python examples/01_hello_world.py

# Analyze the current project
python examples/13_codebase_analysis_pipeline.py .

# Analyze a specific directory, dump the schema
python examples/13_codebase_analysis_pipeline.py /path/to/project --json
```
