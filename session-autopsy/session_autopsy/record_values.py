"""Lenient JSON value coercers used when rebuilding stored records."""

from __future__ import annotations


def as_int(value: object) -> int:
    """Coerce a JSON value to int, treating bools and junk as 0.

    Args:
        value: Any value read from a transcript or store line.

    Returns:
        The int value, or 0 when absent/bool/non-numeric.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def as_float(value: object) -> float:
    """Coerce a JSON value to float, treating bools and junk as 0.0.

    Args:
        value: Any value read from a transcript or store line.

    Returns:
        The float value, or 0.0 when absent/bool/non-numeric.
    """
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def opt_str(value: object) -> str | None:
    """Return the value if it is a non-empty string, else None.

    Args:
        value: Any JSON value.

    Returns:
        The string, or None.
    """
    if isinstance(value, str) and value:
        return value
    return None


def dict_get(value: object, key: str) -> object:
    """Safely index into a value that may not be a dict.

    Args:
        value: Any JSON value.
        key: Key to read when the value is a dict.

    Returns:
        The mapped value, or None.
    """
    if isinstance(value, dict):
        return value.get(key)
    return None


def list_value(value: object) -> list[object]:
    """Return the value if it is a list, else an empty list.

    Args:
        value: Any JSON value.

    Returns:
        A list (possibly empty).
    """
    if isinstance(value, list):
        return value
    return []


def list_of_dicts(value: object) -> list[dict[str, object]]:
    """Filter a JSON value down to its dict elements.

    Args:
        value: Any JSON value.

    Returns:
        The dict elements of the list, or an empty list.
    """
    return [item for item in list_value(value) if isinstance(item, dict)]


def str_list(value: object) -> list[str]:
    """Filter a JSON value down to its string elements.

    Args:
        value: Any JSON value.

    Returns:
        The string elements of the list, or an empty list.
    """
    return [item for item in list_value(value) if isinstance(item, str)]


def str_int_map(value: object) -> dict[str, int]:
    """Coerce a JSON value to a ``str -> int`` counter mapping.

    Args:
        value: Any JSON value.

    Returns:
        A dict of the string keys to int counts, or an empty dict.
    """
    if not isinstance(value, dict):
        return {}
    return {str(key): as_int(val) for key, val in value.items()}
