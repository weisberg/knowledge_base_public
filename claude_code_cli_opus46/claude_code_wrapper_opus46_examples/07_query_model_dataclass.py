#!/usr/bin/env python3
"""
Example 07 — query_model() with Dataclasses
=============================================
Define a Python dataclass, and query_model() will:
  1. Extract a JSON Schema from it automatically
  2. Pass the schema to Claude via --json-schema
  3. Deserialize the response into a populated instance

This is the recommended way to get typed, structured output.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import claude_code_cli as claude


# ------------------------------------------------------------------
# Define your output shape as a plain dataclass
# ------------------------------------------------------------------

@dataclass
class BookRecommendation:
    title: str
    author: str
    year: int
    genre: str
    why: str                          # why this book is recommended
    pages: Optional[int] = None       # optional fields get schema defaults


@dataclass
class ReadingList:
    topic: str
    books: list[BookRecommendation]   # nested dataclass — handled recursively
    total_pages: Optional[int] = None


def main():
    client = claude.ClaudeCode()

    # ---- Peek at the auto-generated schema ----
    schema = claude.extract_schema(ReadingList)
    print("=== Auto-generated JSON Schema ===")
    print(json.dumps(schema, indent=2))
    print()

    # ---- Ask Claude and get back a typed object ----
    print("=== query_model() ===\n")

    result = client.query_model(
        "Recommend 3 classic books about computer science. "
        "Include the approximate page count for each.",
        ReadingList,
    )

    # result.data is a ReadingList instance
    reading_list = result.data
    print(f"Topic: {reading_list.topic}")
    print(f"Total pages: {reading_list.total_pages}")
    print()

    for i, book in enumerate(reading_list.books, 1):
        print(f"  {i}. {book.title} by {book.author} ({book.year})")
        print(f"     Genre: {book.genre}")
        print(f"     Pages: {book.pages}")
        print(f"     Why:   {book.why}")
        print()

    # ---- Response metadata is still available ----
    print(f"Session: {result.session_id}")
    print(f"Cost:    ${result.cost_usd}")
    print(f"Turns:   {result.num_turns}")

    # ---- Verify types ----
    print()
    print(f"Type of result.data:          {type(result.data).__name__}")
    print(f"Type of result.data.books[0]: {type(result.data.books[0]).__name__}")
    assert isinstance(result.data, ReadingList)
    assert all(isinstance(b, BookRecommendation) for b in result.data.books)
    print("Type assertions passed!")


if __name__ == "__main__":
    main()
