"""Shared constants for the TravelPlanner benchmark."""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

NO_DATA: Final = "-"

# ---------------------------------------------------------------------------
# Day-plan field keys
# ---------------------------------------------------------------------------

CURRENT_CITY: Final = "current_city"
TRANSPORTATION: Final = "transportation"
BREAKFAST: Final = "breakfast"
LUNCH: Final = "lunch"
DINNER: Final = "dinner"
ATTRACTION: Final = "attraction"
ACCOMMODATION: Final = "accommodation"

MEAL_KEYS: Final = (BREAKFAST, LUNCH, DINNER)

# All fields that appear in a single day-plan dict
DAY_PLAN_FIELDS: Final = (
    CURRENT_CITY, TRANSPORTATION, BREAKFAST, ATTRACTION,
    LUNCH, DINNER, ACCOMMODATION,
)

# ---------------------------------------------------------------------------
# Transport modes
# ---------------------------------------------------------------------------

FLIGHT: Final = "flight"
SELF_DRIVING: Final = "self-driving"
TAXI: Final = "taxi"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches cost suffixes in plan field values: ", Cost: 120", ", cost: $12.50"
COST_SUFFIX_RE: Final = re.compile(r",\s*[Cc]ost:?\s*\$?[\d,.]+")

# ---------------------------------------------------------------------------
# Known column sets for fixed-width format parsing (tools.py)
# ---------------------------------------------------------------------------

FLIGHT_COLUMNS: Final = [
    "Flight Number", "Price", "DepTime", "ArrTime",
    "ActualElapsedTime", "FlightDate", "OriginCityName",
    "DestCityName", "Distance",
]

RESTAURANT_COLUMNS: Final = [
    "Name", "Average Cost", "Cuisines", "Aggregate Rating", "City",
]

ACCOMMODATION_COLUMNS: Final = [
    "NAME", "price", "room type", "house_rules",
    "minimum nights", "maximum occupancy", "review rate number", "city",
]

ATTRACTION_COLUMNS: Final = [
    "Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City",
]

DISTANCE_COLUMNS: Final = ["duration", "distance", "cost"]

KNOWN_COLUMN_SETS: Final[list[list[str]]] = [
    FLIGHT_COLUMNS,
    RESTAURANT_COLUMNS,
    ACCOMMODATION_COLUMNS,
    ATTRACTION_COLUMNS,
    DISTANCE_COLUMNS,
]

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS: Final = 10
DEFAULT_RETRIEVAL_MAX_ITERATIONS: Final = 5
DEFAULT_ORCHESTRATOR_MAX_ITERATIONS: Final = 5
DEFAULT_MAX_PLAN_RETRIES: Final = 3
DEFAULT_MAX_LOOP_ITERATIONS: Final = 50
DEFAULT_MAX_TOTAL_PRIMITIVE_CALLS: Final = 200
DEFAULT_PARALLEL_WORKERS: Final = 3
DEFAULT_SEED: Final = 42
