"""The small JSON-schema subset an extraction template checks outputs against.

Supported keywords: ``type`` (object, array, string, number, integer, boolean, null, or a list of them),
``properties``, ``required``, ``additionalProperties`` (true / false), ``items`` (one schema), ``enum``, plus the
annotations ``title``, ``description``, ``$schema``, ``$id``, ``default``, ``examples``. Any other keyword is refused
when the schema is checked, so a rule the user wrote is never silently ignored.
"""
from __future__ import annotations

from typing import Any

SUPPORTED = frozenset({"type", "properties", "required", "additionalProperties", "items", "enum"})
ANNOTATIONS = frozenset({"title", "description", "$schema", "$id", "default", "examples"})
_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


class SchemaError(ValueError):
    """The schema itself is unusable (message is shown to the user)."""


def check_schema(schema: Any, path: str = "$") -> None:
    """Raise :class:`SchemaError` when ``schema`` uses anything outside the supported subset."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be a JSON object")
    unknown = sorted(set(schema) - SUPPORTED - ANNOTATIONS)
    if unknown:
        raise SchemaError(f"{path}: unsupported schema keyword(s) {', '.join(unknown)}; supported: "
                          f"{', '.join(sorted(SUPPORTED))}")
    types = schema.get("type")
    for t in (types if isinstance(types, list) else [types] if types is not None else []):
        if t not in _TYPES:
            raise SchemaError(f"{path}: unknown type {t!r}")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise SchemaError(f"{path}: properties must be an object")
    for key, sub in props.items():
        check_schema(sub, f"{path}.{key}")
    req = schema.get("required", [])
    if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
        raise SchemaError(f"{path}: required must be a list of field names")
    if not isinstance(schema.get("additionalProperties", True), bool):
        raise SchemaError(f"{path}: additionalProperties must be true or false (a schema there is not supported)")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{path}: enum must be a list")


def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, (int, float)) and not isinstance(value, bool)  # number


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Every violation of ``schema`` in ``value`` (empty = valid). ``schema`` must have passed :func:`check_schema`."""
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if not any(_type_ok(value, t) for t in allowed):
            return [f"{path}: expected {' or '.join(allowed)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties", True) is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected field {key!r}")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors
