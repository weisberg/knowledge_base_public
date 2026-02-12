#!/usr/bin/env python3
"""
Example 06 — Structured Output (Raw JSON Schema)
==================================================
Force Claude to return validated JSON matching a hand-written schema.
This uses the --json-schema flag directly via the `json_schema` kwarg.
"""

import json
import claude_code_cli as claude


def main():
    client = claude.ClaudeCode()

    # ------------------------------------------------------------------
    # Simple schema — get structured data back
    # ------------------------------------------------------------------
    print("=== Simple JSON schema ===\n")

    schema = {
        "type": "object",
        "properties": {
            "planet": {"type": "string"},
            "distance_from_sun_km": {"type": "number"},
            "has_rings": {"type": "boolean"},
            "number_of_moons": {"type": "integer"},
        },
        "required": ["planet", "distance_from_sun_km", "has_rings", "number_of_moons"],
    }

    response = client.query(
        "Give me facts about Saturn.",
        json_schema=schema,
    )

    print(f"Raw text : {response.text}")
    print(f"Parsed   : {json.dumps(response.json, indent=2) if response.json else 'N/A'}")
    print()

    # ------------------------------------------------------------------
    # Array schema — get a list of items
    # ------------------------------------------------------------------
    print("=== Array-of-objects schema ===\n")

    list_schema = {
        "type": "object",
        "properties": {
            "colors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                        "is_warm": {"type": "boolean"},
                    },
                    "required": ["name", "hex", "is_warm"],
                },
            }
        },
        "required": ["colors"],
    }

    response = client.query(
        "List 5 common colors with their hex codes. Classify each as warm or cool.",
        json_schema=list_schema,
    )

    # Parse the result (it may be nested inside the JSON envelope)
    data = response.json
    if isinstance(data, dict) and "result" in data:
        result = data["result"]
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

    if isinstance(data, dict) and "colors" in data:
        for color in data["colors"]:
            warm = "warm" if color["is_warm"] else "cool"
            print(f"  {color['name']:12s}  {color['hex']}  ({warm})")
    else:
        print(f"  Raw: {response.text}")


if __name__ == "__main__":
    main()
