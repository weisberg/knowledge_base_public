#!/usr/bin/env python3
"""
Example 08 — Enums, Literals, and Field Constraints
=====================================================
Demonstrates richer type annotations that produce more constrained
JSON Schemas — enums become "enum" arrays, Literal becomes string
enums, and FieldMeta adds validation keywords like minimum/maxLength.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Optional

import claude_code_cli as claude
from claude_code_cli import FieldMeta, schema


# ------------------------------------------------------------------
# Enum → JSON Schema "enum" array
# ------------------------------------------------------------------

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


# ------------------------------------------------------------------
# @schema decorator adds class-level metadata
# ------------------------------------------------------------------

@schema(description="A single issue found during code review")
@dataclass
class ReviewIssue:
    # Annotated + FieldMeta adds per-field JSON Schema constraints
    line: Annotated[int, FieldMeta(description="Line number in the file", minimum=1)]
    message: Annotated[str, FieldMeta(description="Description of the issue", max_length=300)]
    priority: Priority
    category: Literal["bug", "style", "security", "performance", "docs"]
    suggestion: Optional[str] = None


@schema(description="Complete code review output")
@dataclass
class CodeReviewReport:
    file_reviewed: str
    summary: Annotated[str, FieldMeta(description="Executive summary", max_length=500)]
    issues: list[ReviewIssue]
    overall_quality: Annotated[int, FieldMeta(
        description="Quality score from 1-10",
        minimum=1,
        maximum=10,
    )]
    status: Status


def main():
    client = claude.ClaudeCode()

    # ---- Show the generated schema ----
    generated = claude.extract_schema(CodeReviewReport)
    print("=== Generated JSON Schema ===")
    print(json.dumps(generated, indent=2))
    print()

    # ---- Sample code to review ----
    sample_code = '''\
import os, sys, json
def process(data):
    password = "admin123"
    result = eval(data["expression"])
    file = open("/tmp/output.txt", "w")
    file.write(str(result))
    return result
'''

    # ---- query_model returns typed objects with enums ----
    print("=== Review result ===\n")

    result = client.query_model(
        "Review this Python code. Find all issues. Be thorough.",
        CodeReviewReport,
        stdin=sample_code,
    )

    report = result.data
    print(f"File:    {report.file_reviewed}")
    print(f"Summary: {report.summary}")
    print(f"Quality: {report.overall_quality}/10")
    print(f"Status:  {report.status.value if isinstance(report.status, Enum) else report.status}")
    print()

    for i, issue in enumerate(report.issues, 1):
        priority = issue.priority.value if isinstance(issue.priority, Enum) else issue.priority
        print(f"  [{priority:8s}] Line {issue.line}: {issue.message}")
        print(f"            Category: {issue.category}")
        if issue.suggestion:
            print(f"            Fix: {issue.suggestion}")
        print()

    print(f"Cost: ${result.cost_usd}")


if __name__ == "__main__":
    main()
