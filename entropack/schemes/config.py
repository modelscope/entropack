from dataclasses import dataclass, fields
from numbers import Real
from typing import Annotated, get_args, get_origin, get_type_hints

__all__ = ["CompressionConfig", "OneOf", "Range", "RawConfig", "config_fields", "validate_config"]


@dataclass
class CompressionConfig:
    """Common execution settings for compression schemes.

    Choose a concrete configuration class to select a scheme. Its fields control compression
    or decompression as documented for that scheme."""

    #: Execution backend: ``"auto"`` or ``None`` selects automatically, or use ``"cuda"`` or ``"eager"``.
    execution_backend: str | None = "auto"


@dataclass
class RawConfig(CompressionConfig):
    """Store tensor values without entropy coding.

    Used explicitly for uncoded FP8 or INT8 weights and as the tensor API's fallback when
    encoding fails. Storage still includes container metadata."""


class Range:
    """An inclusive numeric bound. ``message`` overrides the generated text."""

    def __init__(self, low=None, high=None, *, message: str | None = None):
        self.low = low
        self.high = high
        self.message = message

    def holds(self, value) -> bool:
        return (self.low is None or value >= self.low) and (self.high is None or value <= self.high)

    def __repr__(self) -> str:
        return f"Range({self.low!r}, {self.high!r})"

    def describe(self, name: str) -> str:
        if self.message is not None:
            return self.message
        if self.low is None:
            return f"{name} must be <= {self.high}"
        if self.high is None:
            return f"{name} must be >= {self.low}"
        return f"{name} must be in [{self.low}, {self.high}]"


class OneOf:
    """A closed set of values. ``silent`` holds the ones that ask for automatic selection."""

    def __init__(self, choices, *, silent=(), message: str | None = None):
        self.choices = tuple(choices)
        self.silent = frozenset(silent)
        self.message = message

    def holds(self, value) -> bool:
        return value in self.silent or value in self.choices

    def __repr__(self) -> str:
        return f"OneOf({list(self.choices)!r})"

    def describe(self, name: str) -> str:
        return self.message or f"{name} must be one of {self.choices}"


_hints: dict[type, dict] = {}


def config_fields(cls: type) -> dict:
    cached = _hints.get(cls)
    if cached is None:
        hints = get_type_hints(cls, include_extras=True)
        cached = {field.name: hints[field.name] for field in fields(cls) if field.name != "execution_backend"}
        _hints[cls] = cached
    return cached


def _check(name: str, value, hint) -> None:
    if get_origin(hint) is Annotated:
        hint, *markers = get_args(hint)
    else:
        markers = []
    union = get_args(hint) if get_origin(hint) is not None else (hint,)
    kinds = tuple(kind for kind in union if kind is not type(None))
    optional = " or None" if len(kinds) < len(union) else ""

    if value is None:
        if not optional:
            raise TypeError(f"{name} must be {'an integer' if kinds[0] is int else 'a real number'}")
        return

    if kinds[0] is int:
        holds, expected = isinstance(value, int) and not isinstance(value, bool), "an integer"
    elif kinds[0] is bool:
        holds, expected = isinstance(value, bool), "a bool"
    else:
        holds, expected = isinstance(value, Real) and not isinstance(value, bool), "a real number"
    if not holds:
        raise TypeError(f"{name} must be {expected}{optional}")

    for marker in markers:
        if not marker.holds(value):
            raise ValueError(marker.describe(name))


def validate_config(config) -> None:
    for name, hint in config_fields(type(config)).items():
        _check(name, getattr(config, name), hint)
