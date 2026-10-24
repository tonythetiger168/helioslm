"""The JSON Schema subset hsh tools may declare, and its validator.

dsh enforces a fixed subset (docs/subsystems/tools.md, "The enforced raw
JSON Schema subset"): a schema using anything outside it is refused when
the tool registers, not when the model first calls it. hsh does the same
with this smaller subset:

  type (one name or a list), properties, required, additionalProperties
  (bool or schema), items, enum, const, minimum, maximum, minLength,
  maxLength, minItems, maxItems, oneOf (>= 2 branches, exactly one must
  match), plus the annotations description, title, default, examples.

An empty schema ``{}`` accepts any JSON value.
"""
from __future__ import annotations

_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
_KEYS = {"type", "properties", "required", "additionalProperties", "items",
         "enum", "const", "minimum", "maximum", "minLength", "maxLength",
         "minItems", "maxItems", "oneOf"}
_ANNOTATIONS = {"description", "title", "default", "examples"}


class SchemaError(ValueError):
    pass


def check_schema(schema, path="$"):
    """Raise SchemaError listing the first unsupported or malformed part."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be an object")
    extra = set(schema) - _KEYS - _ANNOTATIONS
    if extra:
        raise SchemaError(f"{path}: unsupported keyword(s) {sorted(extra)}")
    t = schema.get("type")
    for name in ([t] if isinstance(t, str) else t or []):
        if name not in _TYPES:
            raise SchemaError(f"{path}: unknown type {name!r}")
    for k, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}.properties.{k}")
    req = schema.get("required", [])
    if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
        raise SchemaError(f"{path}: required must be a list of names")
    ap = schema.get("additionalProperties", True)
    if isinstance(ap, dict):
        check_schema(ap, f"{path}.additionalProperties")
    elif not isinstance(ap, bool):
        raise SchemaError(f"{path}: additionalProperties must be bool or schema")
    if "items" in schema:
        check_schema(schema["items"], f"{path}.items")
    if "oneOf" in schema:
        branches = schema["oneOf"]
        if not isinstance(branches, list) or len(branches) < 2:
            raise SchemaError(f"{path}: oneOf needs at least two branches")
        for i, b in enumerate(branches):
            check_schema(b, f"{path}.oneOf[{i}]")
    return schema


def _type_ok(name, v):
    if name == "object":
        return isinstance(v, dict)
    if name == "array":
        return isinstance(v, list)
    if name == "string":
        return isinstance(v, str)
    if name == "boolean":
        return isinstance(v, bool)
    if name == "null":
        return v is None
    if isinstance(v, bool):  # bool is an int in Python, not in JSON
        return False
    if name == "integer":
        return isinstance(v, int) or (isinstance(v, float) and v.is_integer())
    return isinstance(v, (int, float))  # number


def validate(value, schema, path="$"):
    """Return a list of error strings (empty when ``value`` conforms)."""
    errs = []
    t = schema.get("type")
    if t is not None:
        names = [t] if isinstance(t, str) else t
        if not any(_type_ok(n, value) for n in names):
            return [f"{path}: expected {'/'.join(names)}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += validate(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}: missing required '{r}'")
        ap = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif ap is False:
                errs.append(f"{path}: unexpected property '{k}'")
            elif isinstance(ap, dict):
                errs += validate(v, ap, f"{path}.{k}")
    if "oneOf" in schema:
        hits = sum(not validate(value, b, path) for b in schema["oneOf"])
        if hits != 1:
            errs.append(f"{path}: matches {hits} oneOf branches, expected exactly 1")
    return errs
