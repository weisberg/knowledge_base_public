#!/usr/bin/env python3
"""
Example 13 — End-to-End Pipeline
==================================
A realistic example: point Claude at a directory, have it analyze the
codebase, and return a fully typed report using query_model().

This combines: working directory, system prompt, structured output,
nested dataclasses, and metadata inspection.
"""

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional

import claude_code_cli as claude
from claude_code_cli import FieldMeta, schema


# ------------------------------------------------------------------
# Define the report structure
# ------------------------------------------------------------------

@dataclass
class FileInfo:
    path: str
    language: str
    line_count: int
    purpose: Annotated[str, FieldMeta(description="One-sentence description of what this file does")]


@dataclass
class Dependency:
    name: str
    version: Optional[str] = None
    purpose: Optional[str] = None


@dataclass
class SecurityNote:
    severity: Literal["info", "low", "medium", "high", "critical"]
    description: str
    file: Optional[str] = None
    recommendation: Optional[str] = None


@schema(description="Comprehensive codebase analysis report")
@dataclass
class CodebaseReport:
    project_name: str
    primary_language: str
    summary: Annotated[str, FieldMeta(
        description="2-3 sentence executive summary of the project",
        max_length=500,
    )]
    files: list[FileInfo]
    dependencies: list[Dependency]
    security_notes: list[SecurityNote]
    test_coverage: Annotated[str, FieldMeta(
        description="Assessment of test coverage: none, minimal, partial, good, comprehensive"
    )]
    complexity_score: Annotated[int, FieldMeta(
        description="Overall complexity from 1 (trivial) to 10 (very complex)",
        minimum=1,
        maximum=10,
    )]
    strengths: list[str]
    improvements: list[str]


# ------------------------------------------------------------------
# Run the analysis
# ------------------------------------------------------------------

def analyze_codebase(directory: str) -> claude.ModelResponse:
    """Analyze a codebase and return a typed CodebaseReport."""

    client = claude.ClaudeCode(
        model="sonnet",
        append_system_prompt=(
            "You are a senior software architect performing a codebase review. "
            "Be thorough but concise. Focus on actionable insights."
        ),
        max_turns=15,           # allow the agent to explore files
        tools=["Read", "Grep", "Glob", "Bash"],  # read-only + safe commands
    )

    return client.query_model(
        "Analyze this codebase thoroughly. Examine the files, dependencies, "
        "architecture, and security posture. Fill out every field in the report.",
        CodebaseReport,
        cwd=directory,
    )


def print_report(result: claude.ModelResponse) -> None:
    """Pretty-print the analysis report."""
    report = result.data

    print("=" * 60)
    print(f"  CODEBASE REPORT: {report.project_name}")
    print("=" * 60)
    print()
    print(f"Language:   {report.primary_language}")
    print(f"Complexity: {report.complexity_score}/10")
    print(f"Tests:      {report.test_coverage}")
    print()
    print(f"Summary:\n  {report.summary}")

    print(f"\n--- Files ({len(report.files)}) ---")
    for f in report.files:
        print(f"  {f.path:30s}  {f.language:10s}  {f.line_count:>5} lines  {f.purpose}")

    if report.dependencies:
        print(f"\n--- Dependencies ({len(report.dependencies)}) ---")
        for dep in report.dependencies:
            ver = f" ({dep.version})" if dep.version else ""
            purpose = f" — {dep.purpose}" if dep.purpose else ""
            print(f"  {dep.name}{ver}{purpose}")

    if report.security_notes:
        print(f"\n--- Security Notes ({len(report.security_notes)}) ---")
        for note in report.security_notes:
            loc = f" [{note.file}]" if note.file else ""
            print(f"  [{note.severity:8s}]{loc} {note.description}")
            if note.recommendation:
                print(f"             Fix: {note.recommendation}")

    print("\n--- Strengths ---")
    for s in report.strengths:
        print(f"  + {s}")

    print("\n--- Suggested Improvements ---")
    for imp in report.improvements:
        print(f"  - {imp}")

    print()
    print("-" * 60)
    print(f"Cost:     ${result.cost_usd}")
    print(f"Duration: {result.duration_ms}ms")
    print(f"Turns:    {result.num_turns}")
    print(f"Session:  {result.session_id}")


def main():
    import sys

    # Default to current directory, or accept a path argument
    target = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Analyzing: {target}")
    print("This may take a minute as Claude explores the files...\n")

    result = analyze_codebase(target)
    print_report(result)

    # Optionally dump the raw JSON
    if "--json" in sys.argv:
        print("\n=== Raw JSON Schema Used ===")
        print(json.dumps(result.schema, indent=2))


if __name__ == "__main__":
    main()
