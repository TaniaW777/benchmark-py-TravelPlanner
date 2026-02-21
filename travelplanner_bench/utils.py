"""Shared utility functions for the TravelPlanner benchmark."""

from __future__ import annotations

import re
from typing import Any

from travelplanner_bench.constants import COST_SUFFIX_RE, NO_DATA


def parse_cost(val: str | int | float | Any) -> float:
    """Parse a cost value from various formats ($120, '120.5', 120, etc.).

    Returns 0.0 for unparseable values.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    cleaned = val.strip().replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def normalize_name(name: str) -> str:
    """Normalize a name for case-insensitive comparison."""
    # Normalize common Unicode variants to ASCII equivalents
    name = name.replace("\u2011", "-").replace("\u2010", "-")
    name = name.replace("\u2013", "-").replace("\u2014", "-")
    name = name.replace("\u2018", "'").replace("\u2019", "'")
    name = name.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse consecutive whitespace (LLMs often collapse double spaces)
    name = re.sub(r"\s+", " ", name)
    return name.lower().strip()


def extract_name(value: str) -> str | None:
    """Extract an entity name from a plan field value.

    Strips cost suffixes like ", Cost: $120".
    Returns None for empty / sentinel values.
    """
    if not value or value.strip() == NO_DATA:
        return None
    name = COST_SUFFIX_RE.sub("", value).strip()
    return name if name and name != NO_DATA else None


def extract_names(value: str, separator: str = ";") -> list[str]:
    """Extract multiple entity names from a separator-delimited field.

    Used for attraction fields like "Attr1;Attr2;Attr3".
    """
    if not value or value.strip() == NO_DATA:
        return []
    names: list[str] = []
    for part in value.split(separator):
        name = extract_name(part.strip())
        if name:
            names.append(name)
    return names


def name_in_set(name: str, name_set: set[str]) -> bool:
    """Check if a name matches any entry in a set (case-insensitive)."""
    n = normalize_name(name)
    return any(normalize_name(entry) == n for entry in name_set)


def strip_city_suffix(name: str) -> str:
    """Strip trailing ', CityName' from an entity name.

    Preserves names that don't contain a comma.
    """
    parts = name.rsplit(",", 1)
    return parts[0].strip() if len(parts) > 1 else name
