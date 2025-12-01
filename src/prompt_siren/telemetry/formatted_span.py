# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Provides a wrapper for logfire.span that formats the span name before creating the span."""

from __future__ import annotations

import string
import warnings
from collections.abc import Iterator
from typing import Any, TypeGuard, TypeVar

import logfire

# Type variables for better typing
T = TypeVar("T")
SpanContextManager = TypeVar("SpanContextManager")


Primitive = str | bool | int | float
AttrValue = Primitive | list[Primitive]


def is_primitive(x: object) -> TypeGuard[Primitive]:
    return isinstance(x, str | bool | int | float)


def _coerce_to_str(x: object) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def flatten_attribute(prefix: str, value: Any) -> Iterator[tuple[str, Primitive]]:
    """
    Flatten value under key prefix, ensuring that:
    - only dicts are expanded into dotted keys
    - iterables (list, tuple, set) are always flattened with index in key
    - no nested dicts or non-primitive objects are left as values, except via string coercion
    - None/null values omitted
    """

    # Drop None / null
    if value is None:
        return

    # Primitives → good
    if is_primitive(value):
        yield (prefix, value)
        return

    # Dict → expand
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_attribute(key, v)
        return

    # Iterable / sequence → always by index
    if isinstance(value, list | tuple | set):
        for i, item in enumerate(value):
            key = f"{prefix}[{i}]"
            yield from flatten_attribute(key, item)
        return

    # Everything else → coerce to string
    yield (prefix, _coerce_to_str(value))


def flatten_attributes(root_key: str, obj: Any) -> dict[str, AttrValue]:
    """
    Wrapper to return a dict of flattened attributes under root_key.
    """

    return dict(flatten_attribute(root_key, obj))


class PartialFormatter(string.Formatter):
    """Formatter that ignores missing keys."""

    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except (IndexError, KeyError):
            # Return the original placeholder if key is missing
            return f"{{{key}}}"


_FORMATTER = PartialFormatter()


def formatted_span(name_template: str, **kwargs: Any) -> logfire.LogfireSpan:
    """
    Wrapper for logfire.span that formats the name before creating the span.

    Works like logfire.span but formats the name template using the provided kwargs.
    This ensures the span name is properly formatted in Phoenix and other visualization tools.

    Automatically flattens nested structures in attributes for better telemetry compatibility.
    Nested dicts are converted to dot notation (e.g., {"a": {"b": 1}} becomes {"a.b": 1}).
    Lists and sets are indexed (e.g., {"items": [1, 2]} becomes {"items[0]": 1, "items[1]": 2}).

    Args:
        name_template: Template string for the span name
        **kwargs: Attributes for the span, also used for formatting the name.
                 Nested structures will be automatically flattened.

    Returns:
        A context manager that yields a span with a properly formatted name
    """
    try:
        formatted_name = _FORMATTER.format(name_template, **kwargs)
    except (ValueError, TypeError):
        # Fall back to the template if formatting fails
        formatted_name = name_template

    # Flatten all attributes before passing to logfire
    flattened_kwargs = {}
    for key, value in kwargs.items():
        # Special keys that logfire might need as-is (e.g., 'kind' for span type)
        # These are passed through without flattening
        if key in {"kind", "_span_name"}:
            flattened_kwargs[key] = value
        else:
            # Flatten each top-level attribute
            flattened = flatten_attributes(key, value)
            flattened_kwargs.update(flattened)

    # Call logfire.span with the pre-formatted name
    # Note: Logfire will generate FormattingFailedWarning but we suppress it
    # since we're intentionally pre-formatting to fix Phoenix display issues

    with warnings.catch_warnings():
        # FormattingFailedWarning is a specific warning type in logfire
        warnings.filterwarnings(
            "ignore",
            message=".*passing.*preformatted string.*",
            module="logfire",
        )
        return logfire.span(formatted_name, **flattened_kwargs)
