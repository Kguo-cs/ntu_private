"""Resolve backward-compatible, per-field ego conditions."""

EGO_FIELDS = ("position", "heading", "shape", "velocity", "type")


def resolve_ego_conditioning(fix_ego, **overrides):
    if not isinstance(fix_ego, bool):
        raise ValueError("fix_ego must be boolean")
    result = {}
    for field in EGO_FIELDS:
        name = f"fix_ego_{field}"
        value = overrides.get(name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean or null")
        result[name] = fix_ego if value is None else value
    return result
