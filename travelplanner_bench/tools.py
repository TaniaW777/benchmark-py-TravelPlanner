"""Search tool implementations backed by parsed reference_information.

Each TravelPlanner task comes with pre-collected reference data. The
ReferenceDatabase indexes this data so the agent's search primitives
can query it as if querying a real database.
"""

from __future__ import annotations

import logging
import re

from travelplanner_bench.constants import FLIGHT, KNOWN_COLUMN_SETS, SELF_DRIVING, TAXI
from travelplanner_bench.models import (
    Accommodation,
    Attraction,
    DistanceInfo,
    Flight,
    Restaurant,
)

log = logging.getLogger(__name__)


class ReferenceDatabase:
    """In-memory database built from a task's reference_information.

    Parses tab-separated or fixed-width content blocks indexed by description type.
    """

    def __init__(self, reference_entries: list[dict[str, str]]) -> None:
        # Flights keyed by (origin_lower, dest_lower, date)
        self.flights: dict[tuple[str, str, str], list[Flight]] = {}
        # Accommodations keyed by city_lower
        self.accommodations: dict[str, list[Accommodation]] = {}
        # Restaurants keyed by city_lower
        self.restaurants: dict[str, list[Restaurant]] = {}
        # Attractions keyed by city_lower
        self.attractions: dict[str, list[Attraction]] = {}
        # Distances keyed by (origin_lower, dest_lower, mode_lower)
        self.distances: dict[tuple[str, str, str], DistanceInfo] = {}
        # Cities keyed by state_lower
        self.cities: dict[str, list[str]] = {}

        # Flat sets for sandbox validation
        self.all_flight_numbers: set[str] = set()
        self.all_restaurant_names: set[str] = set()
        self.all_accommodation_names: set[str] = set()
        self.all_attraction_names: set[str] = set()
        # Entity → set of cities (handles chain names in multiple cities)
        self.restaurant_city: dict[str, set[str]] = {}
        self.accommodation_city: dict[str, set[str]] = {}
        self.attraction_city: dict[str, set[str]] = {}

        self._parse(reference_entries)

    def _parse(self, entries: list[dict[str, str]]) -> None:
        for entry in entries:
            desc = entry.get("Description", "")
            content = entry.get("Content", "")
            if not desc or not content:
                continue

            desc_lower = desc.lower()
            if FLIGHT in desc_lower:
                self._parse_flights(desc, content)
            elif "restaurant" in desc_lower:
                self._parse_restaurants(desc, content)
            elif "accommodation" in desc_lower:
                self._parse_accommodations(desc, content)
            elif "attraction" in desc_lower:
                self._parse_attractions(desc, content)
            elif SELF_DRIVING in desc_lower or TAXI in desc_lower or "driving" in desc_lower:
                self._parse_distance(desc, content)
            elif "cities in" in desc_lower:
                self._parse_cities(desc, content)

    # ------------------------------------------------------------------
    # Content parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_content(content: str) -> list[dict[str, str]]:
        """Parse content, auto-detecting tab-separated or fixed-width format."""
        if "\t" in content:
            return ReferenceDatabase._parse_tsv(content)
        return ReferenceDatabase._parse_fwf(content)

    @staticmethod
    def _parse_tsv(content: str) -> list[dict[str, str]]:
        """Parse tab-separated content into list of dicts using header row."""
        lines = [line for line in content.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return []
        headers = [h.strip() for h in lines[0].split("\t")]
        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            vals = [v.strip() for v in line.split("\t")]
            while len(vals) < len(headers):
                vals.append("")
            row = dict(zip(headers, vals[: len(headers)]))
            rows.append(row)
        return rows

    @staticmethod
    def _parse_fwf(content: str) -> list[dict[str, str]]:
        """Parse fixed-width content (from pandas to_string) using known columns."""
        # Do NOT strip() content before splitting - leading spaces are significant
        # for column position alignment between header and data rows.
        lines = [line for line in content.split("\n") if line.strip()]
        if len(lines) < 2:
            return []
        header = lines[0]

        col_names = ReferenceDatabase._match_known_columns(header)
        if not col_names:
            # Fallback: split header by 2+ spaces
            col_names = [c for c in re.split(r"  +", header.strip()) if c]
            if not col_names:
                return []

        # Find each column name's position in the header
        positions: list[int] = []
        search_from = 0
        for name in col_names:
            idx = header.find(name, search_from)
            if idx == -1:
                return []
            positions.append(idx)
            search_from = idx + len(name)

        # Boundaries = end of each column header text (start of the gap)
        boundaries = [
            pos + len(name)
            for pos, name in zip(positions[:-1], col_names[:-1])
        ]

        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            row: dict[str, str] = {}
            for i, name in enumerate(col_names):
                start = boundaries[i - 1] if i > 0 else 0
                end = boundaries[i] if i < len(boundaries) else len(line)
                val = line[start:min(end, len(line))].strip() if start < len(line) else ""
                # Strip pandas row index from the first column (e.g.
                # "251                  Flying Mango" -> "Flying Mango")
                if i == 0 and re.match(r"^\d+\s{2,}", val):
                    val = re.sub(r"^\d+\s{2,}", "", val)
                row[name] = val
            rows.append(row)
        return rows

    @staticmethod
    def _match_known_columns(header: str) -> list[str] | None:
        """Match a header line against known column sets."""
        for col_set in KNOWN_COLUMN_SETS:
            pos = 0
            matched = True
            for col in col_set:
                idx = header.find(col, pos)
                if idx == -1:
                    matched = False
                    break
                pos = idx + len(col)
            if matched:
                return col_set
        return None

    # ------------------------------------------------------------------
    # Entity-specific parsing
    # ------------------------------------------------------------------

    def _parse_flights(self, desc: str, content: str) -> None:
        rows = self._parse_content(content)
        flights = [Flight.from_raw(row) for row in rows]

        match = re.search(
            r"[Ff]light\w*\s+from\s+(.+?)\s+to\s+(.+?)\s+on\s+(\d{4}-\d{2}-\d{2})",
            desc,
        )
        if match:
            origin, dest, date = match.group(1).strip(), match.group(2).strip(), match.group(3)
            key = (origin.lower(), dest.lower(), date)
            self.flights.setdefault(key, []).extend(flights)
        else:
            for f in flights:
                if f.origin and f.destination and f.date:
                    key = (f.origin.lower(), f.destination.lower(), f.date)
                    self.flights.setdefault(key, []).append(f)

        for f in flights:
            if f.flight_number:
                self.all_flight_numbers.add(f.flight_number)

    def _parse_restaurants(self, desc: str, content: str) -> None:
        rows = self._parse_content(content)
        restaurants = [Restaurant.from_raw(row) for row in rows]

        match = re.search(r"[Rr]estaurants?\s+in\s+(.+)", desc)
        city = match.group(1).strip() if match else ""
        if city:
            self.restaurants.setdefault(city.lower(), []).extend(restaurants)

        for r in restaurants:
            row_city = r.city or city
            if r.name:
                self.all_restaurant_names.add(r.name)
                self.restaurant_city.setdefault(r.name.lower(), set()).add(row_city)

    def _parse_accommodations(self, desc: str, content: str) -> None:
        rows = self._parse_content(content)
        accommodations = [Accommodation.from_raw(row) for row in rows]

        match = re.search(r"[Aa]ccommodations?\s+in\s+(.+)", desc)
        city = match.group(1).strip() if match else ""
        if city:
            self.accommodations.setdefault(city.lower(), []).extend(accommodations)

        for a in accommodations:
            row_city = a.city or city
            if a.name:
                self.all_accommodation_names.add(a.name)
                self.accommodation_city.setdefault(a.name.lower(), set()).add(row_city)

    def _parse_attractions(self, desc: str, content: str) -> None:
        rows = self._parse_content(content)
        attractions = [Attraction.from_raw(row) for row in rows]

        match = re.search(r"[Aa]ttractions?\s+in\s+(.+)", desc)
        city = match.group(1).strip() if match else ""
        if city:
            self.attractions.setdefault(city.lower(), []).extend(attractions)

        for a in attractions:
            row_city = a.city or city
            if a.name:
                self.all_attraction_names.add(a.name)
                self.attraction_city.setdefault(a.name.lower(), set()).add(row_city)

    def _parse_distance(self, desc: str, content: str) -> None:
        match = re.search(
            r"(self-driving|taxi|driving)\s+from\s+(.+?)\s+to\s+(.+)",
            desc,
            re.IGNORECASE,
        )
        if not match:
            return
        mode = match.group(1).strip().lower()
        if mode == "driving":
            mode = SELF_DRIVING
        origin = match.group(2).strip()
        dest = match.group(3).strip()

        # Try table format first (TSV or FWF)
        rows = self._parse_content(content)
        if rows:
            raw = rows[0]
        else:
            # Fallback: parse comma-separated key-value format
            raw: dict[str, str] = {}
            for field in ("duration", "distance", "cost"):
                m = re.search(rf"{field}\s*:\s*([^,]+)", content, re.IGNORECASE)
                if m:
                    raw[field] = m.group(1).strip()

        if raw:
            raw["mode"] = mode
            raw["origin"] = origin
            raw["destination"] = dest
            dist = DistanceInfo.from_raw(raw)
            key = (origin.lower(), dest.lower(), mode)
            self.distances[key] = dist

    def _parse_cities(self, desc: str, content: str) -> None:
        match = re.search(r"[Cc]ities\s+in\s+(.+)", desc)
        state = match.group(1).strip() if match else ""
        if not state:
            return
        city_list = [c.strip() for c in content.strip().split("\n") if c.strip()]
        self.cities[state.lower()] = city_list


# ===========================================================================
# Standalone tool functions
# ===========================================================================


def _fuzzy_city_key(db_dict: dict[str, list[object]], city: str) -> str | None:
    """Find the best matching city key in a dict (case-insensitive, partial match)."""
    city_lower = city.lower().strip()
    if city_lower in db_dict:
        return city_lower
    for key in db_dict:
        if city_lower in key or key in city_lower:
            return key
    return None


def search_flights(
    db: ReferenceDatabase, origin: str, destination: str, date: str
) -> list[Flight]:
    """Return flights matching origin, destination, date."""
    key = (origin.lower().strip(), destination.lower().strip(), date.strip())
    if key in db.flights:
        return db.flights[key]
    # Fuzzy: try partial match on city names
    for fkey, flights in db.flights.items():
        if (
            key[0] in fkey[0] or fkey[0] in key[0]
        ) and (
            key[1] in fkey[1] or fkey[1] in key[1]
        ) and fkey[2] == key[2]:
            return flights
    return []


def search_accommodations(
    db: ReferenceDatabase, city: str
) -> list[Accommodation]:
    """Return accommodations in the given city."""
    key = _fuzzy_city_key(db.accommodations, city)
    return db.accommodations.get(key, []) if key else []


def search_restaurants(
    db: ReferenceDatabase, city: str
) -> list[Restaurant]:
    """Return restaurants in the given city."""
    key = _fuzzy_city_key(db.restaurants, city)
    return db.restaurants.get(key, []) if key else []


def search_attractions(
    db: ReferenceDatabase, city: str
) -> list[Attraction]:
    """Return attractions in the given city."""
    key = _fuzzy_city_key(db.attractions, city)
    return db.attractions.get(key, []) if key else []


def get_distance(
    db: ReferenceDatabase, origin: str, destination: str, mode: str = SELF_DRIVING
) -> DistanceInfo | None:
    """Return distance/duration/cost between two cities."""
    key = (origin.lower().strip(), destination.lower().strip(), mode.lower().strip())
    if key in db.distances:
        return db.distances[key]
    # Fuzzy match
    for dkey, data in db.distances.items():
        if (
            (key[0] in dkey[0] or dkey[0] in key[0])
            and (key[1] in dkey[1] or dkey[1] in key[1])
            and dkey[2] == key[2]
        ):
            return data
    return None


def search_cities(
    db: ReferenceDatabase, state: str
) -> list[str]:
    """Return list of cities in the given state.

    Falls back to inferring cities from available restaurant/accommodation/
    attraction data when no explicit "cities in <state>" entry exists.
    """
    key = state.lower().strip()
    if key in db.cities:
        return db.cities[key]
    for skey, cities in db.cities.items():
        if key in skey or skey in key:
            return cities
    # Fallback: infer destination cities from available data.
    all_cities: set[str] = set()
    all_cities.update(db.restaurants.keys())
    all_cities.update(db.accommodations.keys())
    all_cities.update(db.attractions.keys())
    if all_cities:
        return sorted(all_cities)
    return []
