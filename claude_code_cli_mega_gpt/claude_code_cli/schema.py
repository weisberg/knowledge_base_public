"""Automatic JSON Schema extraction and light transformation helpers.

Supports schema extraction from:
  - dataclasses
  - Pydantic BaseModel (v2 and v1)
  - TypedDict
  - Plain classes with ``__annotations__``
  - ``@schema`` decorator for class-level metadata
  - Field metadata via ``typing.Annotated[..., FieldMeta(...)]``

Also includes `simplify_schema_for_claude()`, inspired by Anthropic's SDK behavior:
  - Remove unsupported constraints (example: minimum/maximum/minLength/maxLength)
  - Add `additionalProperties: false` to objects
  - Optionally filter unsupported string `format` values

CLI usage uses `claude -p --json-schema '<schema>' --output-format json`.
Docs:
  - CLI reference: https://code.claude.com/docs/en/cli-reference
  - Structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import sys
from typing import (
    Any,
    ForwardRef,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# ---------------------------------------------------------------------------
# Pydantic detection (optional dependency)
# ---------------------------------------------------------------------------

_PYDANTIC_V2 = False
_PYDANTIC_V1 = False

try:
    import pydantic  # type: ignore

    if hasattr(pydantic, "BaseModel"):
        _ver = getattr(pydantic, "VERSION", "0")
        if str(_ver).startswith("2"):
            _PYDANTIC_V2 = True
        else:
            _PYDANTIC_V1 = True
except ImportError:
    pydantic = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_pydantic_model(cls: type) -> bool:
    """Return True if *cls* is a Pydantic BaseModel subclass."""
    if pydantic is None:
        return False
    return inspect.isclass(cls) and issubclass(cls, pydantic.BaseModel)  # type: ignore[attr-defined]


def is_dataclass(cls: type) -> bool:
    return dataclasses.is_dataclass(cls) and inspect.isclass(cls)


def is_typed_dict(cls: type) -> bool:
    """Detect TypedDict classes (both typing and typing_extensions)."""
    if not inspect.isclass(cls):
        return False
    # Python 3.12+ or typing_extensions
    for base in getattr(cls, "__mro__", []):
        if base.__name__ == "TypedDict":
            return True
    # Heuristic: TypedDict classes have __annotations__ and __required_keys__
    return hasattr(cls, "__required_keys__") and hasattr(cls, "__optional_keys__")


# ---------------------------------------------------------------------------
# ``@schema`` decorator — optional metadata for plain classes
# ---------------------------------------------------------------------------


def schema(
    *,
    description: str | None = None,
    examples: list[Any] | None = None,
    title: str | None = None,
):
    """Class decorator to attach JSON Schema metadata to any class.

    Usage::

        @schema(description="A code review result")
        @dataclass
        class ReviewResult:
            summary: str
            issues: list[str]
    """

    def _decorator(cls: type) -> type:
        if description is not None:
            cls.__schema_description__ = description  # type: ignore[attr-defined]
        if examples is not None:
            cls.__schema_examples__ = examples  # type: ignore[attr-defined]
        if title is not None:
            cls.__schema_title__ = title  # type: ignore[attr-defined]
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Field-level metadata via ``Annotated``
# ---------------------------------------------------------------------------


class FieldMeta:
    """Attach JSON Schema metadata to a field via ``Annotated[str, FieldMeta(...)]``."""

    def __init__(
        self,
        *,
        description: str | None = None,
        examples: list[Any] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        pattern: str | None = None,
        default: Any = dataclasses.MISSING,
        enum: list[Any] | None = None,
        format: str | None = None,
    ):
        self.description = description
        self.examples = examples
        self.min_length = min_length
        self.max_length = max_length
        self.minimum = minimum
        self.maximum = maximum
        self.pattern = pattern
        self.default = default
        self.enum = enum
        self.format = format

    def apply(self, prop: dict[str, Any]) -> None:
        """Merge this metadata into a JSON Schema property dict."""
        if self.description:
            prop["description"] = self.description
        if self.examples:
            prop["examples"] = self.examples
        if self.min_length is not None:
            prop["minLength"] = self.min_length
        if self.max_length is not None:
            prop["maxLength"] = self.max_length
        if self.minimum is not None:
            prop["minimum"] = self.minimum
        if self.maximum is not None:
            prop["maximum"] = self.maximum
        if self.pattern:
            prop["pattern"] = self.pattern
        if self.enum:
            prop["enum"] = self.enum
        if self.format:
            prop["format"] = self.format


# ---------------------------------------------------------------------------
# Core schema extraction
# ---------------------------------------------------------------------------

def extract_schema(cls: type) -> dict[str, Any]:
    """Derive a JSON Schema ``dict`` from a Python class.

    Supports dataclasses, Pydantic BaseModel (v1/v2), TypedDict, and
    plain classes with ``__annotations__``.

    Raises TypeError if *cls* is not a supported type.
    """

    # --- Pydantic v2 (best path — native schema generation) ---
    if _PYDANTIC_V2 and is_pydantic_model(cls):
        return cls.model_json_schema()  # type: ignore[union-attr]

    # --- Pydantic v1 ---
    if _PYDANTIC_V1 and is_pydantic_model(cls):
        return cls.schema()  # type: ignore[union-attr]

    # --- dataclass / TypedDict / annotated class ---
    if is_dataclass(cls) or is_typed_dict(cls) or hasattr(cls, "__annotations__"):
        return _schema_from_annotations(cls)

    raise TypeError(
        f"Cannot extract JSON Schema from {cls!r}. "
        "Supported types: dataclass, Pydantic BaseModel, TypedDict, "
        "or any class with __annotations__."
    )


# ---------------------------------------------------------------------------
# Annotation-based schema builder
# ---------------------------------------------------------------------------

def _schema_from_annotations(cls: type) -> dict[str, Any]:
    """Build a JSON Schema from class annotations and metadata."""
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    # Determine which fields are required
    if is_typed_dict(cls):
        required_keys = getattr(cls, "__required_keys__", set())
    elif is_dataclass(cls):
        dc_fields = {f.name: f for f in dataclasses.fields(cls)}
    else:
        dc_fields = None

    for field_name, field_type in hints.items():
        if field_name.startswith("_"):
            continue

        prop, field_meta = _type_to_schema(field_type)

        # Apply FieldMeta from Annotated if present
        if field_meta:
            field_meta.apply(prop)

        properties[field_name] = prop

        # Determine if required
        if is_typed_dict(cls):
            if field_name in required_keys:
                required.append(field_name)
        elif is_dataclass(cls) and dc_fields is not None:
            f = dc_fields.get(field_name)
            if f and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                required.append(field_name)
        else:
            # For plain annotated classes, check for class-level defaults
            if not hasattr(cls, field_name):
                required.append(field_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    # Apply class-level metadata from @schema decorator
    if hasattr(cls, "__schema_description__"):
        schema["description"] = cls.__schema_description__  # type: ignore[attr-defined]
    if hasattr(cls, "__schema_title__"):
        schema["title"] = cls.__schema_title__  # type: ignore[attr-defined]
    elif hasattr(cls, "__name__"):
        schema["title"] = cls.__name__
    if hasattr(cls, "__schema_examples__"):
        schema["examples"] = cls.__schema_examples__  # type: ignore[attr-defined]

    return schema


def _type_to_schema(tp: Any) -> tuple[dict[str, Any], FieldMeta | None]:
    """Convert a single Python type annotation to a JSON Schema property.

    Returns ``(schema_dict, optional_field_meta)``.
    """
    origin = get_origin(tp)
    args = get_args(tp)

    field_meta: FieldMeta | None = None

    # --- typing.Annotated ---
    if origin is not None and _is_annotated(tp):
        # Annotated[base_type, metadata...]
        base = args[0]
        for meta in args[1:]:
            if isinstance(meta, FieldMeta):
                field_meta = meta
        prop, _ = _type_to_schema(base)
        return prop, field_meta

    # --- None / NoneType ---
    if tp is type(None):
        return {"type": "null"}, None

    # --- Literal ---
    if origin is Literal:
        values = list(args)
        # Infer type from values
        types = {type(v) for v in values}
        if types == {str}:
            return {"type": "string", "enum": values}, None
        if types == {int}:
            return {"type": "integer", "enum": values}, None
        if types == {float}:
            return {"type": "number", "enum": values}, None
        return {"enum": values}, None

    # --- Union (includes Optional) ---
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)

        if len(non_none) == 1 and has_none:
            # Optional[X] → schema for X (field is not in required list)
            prop, meta = _type_to_schema(non_none[0])
            return prop, meta

        schemas = [_type_to_schema(a)[0] for a in args]
        return {"anyOf": schemas}, None

    # --- list / List[X] ---
    if origin is list or (origin is not None and _safe_issubclass(origin, list)):
        items = _type_to_schema(args[0])[0] if args else {}
        return {"type": "array", "items": items}, None

    # --- tuple / Tuple ---
    if origin is tuple or (origin is not None and _safe_issubclass(origin, tuple)):
        if args:
            items = [_type_to_schema(a)[0] for a in args if a is not Ellipsis]
            return {"type": "array", "prefixItems": items, "items": False}, None
        return {"type": "array"}, None

    # --- dict / Dict[K, V] ---
    if origin is dict or (origin is not None and _safe_issubclass(origin, dict)):
        if args and len(args) == 2:
            val_schema = _type_to_schema(args[1])[0]
            return {"type": "object", "additionalProperties": val_schema}, None
        return {"type": "object"}, None

    # --- set / frozenset → array with uniqueItems ---
    if origin in (set, frozenset) or (origin is not None and _safe_issubclass(origin, (set, frozenset))):
        items = _type_to_schema(args[0])[0] if args else {}
        return {"type": "array", "items": items, "uniqueItems": True}, None

    # --- Sequence / Mapping (abstract) ---
    if origin is not None and _safe_issubclass(origin, Sequence):
        items = _type_to_schema(args[0])[0] if args else {}
        return {"type": "array", "items": items}, None
    if origin is not None and _safe_issubclass(origin, Mapping):
        if args and len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1])[0]}, None
        return {"type": "object"}, None

    # --- Enum subclasses ---
    if inspect.isclass(tp) and issubclass(tp, __import__("enum").Enum):  # type: ignore[arg-type]
        values = [e.value for e in tp]  # type: ignore[union-attr]
        return {"enum": values}, None

    # --- Nested model / dataclass / TypedDict ---
    if inspect.isclass(tp) and (is_dataclass(tp) or is_typed_dict(tp) or is_pydantic_model(tp)):
        return extract_schema(tp), None

    # --- Plain classes with annotations (nested) ---
    if inspect.isclass(tp) and hasattr(tp, "__annotations__") and tp.__annotations__:
        return _schema_from_annotations(tp), None

    # --- Primitive types ---
    return _primitive_schema(tp), None


def _primitive_schema(tp: Any) -> dict[str, Any]:
    """Map a primitive Python type to its JSON Schema equivalent."""
    mapping: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }
    if inspect.isclass(tp) and tp in mapping:
        return mapping[tp]

    # ForwardRef / string annotations we can't resolve
    if isinstance(tp, (str, ForwardRef)):
        return {}

    # Fallback
    return {}


# ---------------------------------------------------------------------------
# Schema transformation (lightweight, CLI-friendly)
# ---------------------------------------------------------------------------

DEFAULT_SUPPORTED_STRING_FORMATS: set[str] = {
    # Conservative set commonly accepted for JSON schema + LLM structured outputs.
    # Override via `supported_string_formats=...` if you need more.
    "date",
    "date-time",
    "email",
    "uri",
    "uuid",
}


def simplify_schema_for_claude(
    schema: dict[str, Any],
    *,
    supported_string_formats: set[str] | None = None,
    add_additional_properties_false: bool = True,
    remove_unsupported_constraints: bool = True,
    move_constraints_to_description: bool = True,
    filter_string_formats: bool = True,
) -> dict[str, Any]:
    """Return a CLI/LLM-friendly schema by removing/adjusting risky features.

    This mirrors the high-level behavior described in Anthropic's structured
    output docs: remove unsupported constraints, add `additionalProperties: false`,
    filter string formats. (You can keep your original schema for validation.)

    The goal is to reduce 400 errors like "Schema is too complex" / "unsupported".
    """
    out = copy.deepcopy(schema)
    fmt_allow = supported_string_formats if supported_string_formats is not None else DEFAULT_SUPPORTED_STRING_FORMATS
    _simplify_node(
        out,
        fmt_allow=fmt_allow,
        add_additional_properties_false=add_additional_properties_false,
        remove_unsupported_constraints=remove_unsupported_constraints,
        move_constraints_to_description=move_constraints_to_description,
        filter_string_formats=filter_string_formats,
    )
    return out


def _simplify_node(
    node: Any,
    *,
    fmt_allow: set[str],
    add_additional_properties_false: bool,
    remove_unsupported_constraints: bool,
    move_constraints_to_description: bool,
    filter_string_formats: bool,
) -> None:
    if isinstance(node, list):
        for it in node:
            _simplify_node(
                it,
                fmt_allow=fmt_allow,
                add_additional_properties_false=add_additional_properties_false,
                remove_unsupported_constraints=remove_unsupported_constraints,
                move_constraints_to_description=move_constraints_to_description,
                filter_string_formats=filter_string_formats,
            )
        return

    if not isinstance(node, dict):
        return

    # Recurse into known schema containers
    for key in ("properties", "$defs", "definitions"):
        v = node.get(key)
        if isinstance(v, dict):
            for sub in v.values():
                _simplify_node(
                    sub,
                    fmt_allow=fmt_allow,
                    add_additional_properties_false=add_additional_properties_false,
                    remove_unsupported_constraints=remove_unsupported_constraints,
                    move_constraints_to_description=move_constraints_to_description,
                    filter_string_formats=filter_string_formats,
                )

    for key in ("items", "additionalProperties", "contains"):
        v = node.get(key)
        if isinstance(v, dict):
            _simplify_node(
                v,
                fmt_allow=fmt_allow,
                add_additional_properties_false=add_additional_properties_false,
                remove_unsupported_constraints=remove_unsupported_constraints,
                move_constraints_to_description=move_constraints_to_description,
                filter_string_formats=filter_string_formats,
            )

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        v = node.get(key)
        if isinstance(v, list):
            _simplify_node(
                v,
                fmt_allow=fmt_allow,
                add_additional_properties_false=add_additional_properties_false,
                remove_unsupported_constraints=remove_unsupported_constraints,
                move_constraints_to_description=move_constraints_to_description,
                filter_string_formats=filter_string_formats,
            )

    # 1) Add additionalProperties:false to objects
    if add_additional_properties_false and node.get("type") == "object":
        if "additionalProperties" not in node:
            node["additionalProperties"] = False

    # 2) Remove unsupported constraints (common offenders)
    # Only remove those explicitly called out by Anthropic docs (and a few close cousins).
    removed: list[str] = []
    if remove_unsupported_constraints:
        for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength"):
            if k in node:
                removed.append(f"{k}={node.get(k)!r}")
                node.pop(k, None)

    # 3) Optionally filter string formats to supported list
    if filter_string_formats and node.get("type") == "string" and "format" in node:
        fmt = node.get("format")
        if isinstance(fmt, str) and fmt not in fmt_allow:
            removed.append(f"format={fmt!r}")
            node.pop("format", None)

    # 4) Move constraints into description (so the model still sees them)
    if removed and move_constraints_to_description:
        desc = node.get("description")
        suffix = "Constraints: " + ", ".join(removed)
        if isinstance(desc, str) and desc.strip():
            if suffix not in desc:
                node["description"] = desc.rstrip() + "\n" + suffix
        else:
            node["description"] = suffix


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_annotated(tp: Any) -> bool:
    """Check if a type is ``typing.Annotated``."""
    try:
        from typing import Annotated
        return get_origin(tp) is Annotated
    except ImportError:
        pass
    try:
        from typing_extensions import Annotated, get_origin as ext_get_origin  # type: ignore
        return ext_get_origin(tp) is Annotated
    except ImportError:
        return False


def _safe_issubclass(cls: Any, bases: Any) -> bool:
    """``issubclass`` that doesn't raise on non-classes."""
    try:
        return inspect.isclass(cls) and issubclass(cls, bases)
    except TypeError:
        return False
