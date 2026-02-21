"""Framework-specific tool wrappers for search primitives.

Wraps the 6 search functions from tools.py for LangChain and CrewAI.
Tools return JSON strings and auto-accumulate results into GatheredData.
"""

from __future__ import annotations

import json
from typing import Any

from travelplanner_bench.models import (
    Accommodation,
    Attraction,
    DistanceInfo,
    Flight,
    GatheredData,
    Restaurant,
)
from travelplanner_bench.tools import (
    ReferenceDatabase,
    get_distance,
    search_accommodations,
    search_attractions,
    search_cities,
    search_flights,
    search_restaurants,
)


def _serialize(objs: list[Any] | Any | None) -> str:
    """Serialize Pydantic models to JSON string for LLM consumption."""
    if objs is None:
        return json.dumps(None)
    if isinstance(objs, list):
        return json.dumps([o.model_dump() for o in objs], default=str)
    return json.dumps(objs.model_dump(), default=str)


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------


def make_langchain_tools(
    db: ReferenceDatabase,
    gathered: GatheredData,
) -> list:
    """Create LangChain-compatible tools that auto-accumulate into GatheredData.

    Requires: langchain-core
    """
    from langchain_core.tools import tool

    @tool
    def search_flights_tool(origin: str, destination: str, date: str) -> str:
        """Search flights from origin city to destination city on a specific date (YYYY-MM-DD format)."""
        results = search_flights(db, origin, destination, date)
        if results:
            key = f"{origin}->{destination} on {date}"
            gathered.flights[key] = results
        return _serialize(results)

    @tool
    def search_restaurants_tool(city: str) -> str:
        """Search for restaurants in a city."""
        results = search_restaurants(db, city)
        if results:
            gathered.restaurants[city] = results
        return _serialize(results)

    @tool
    def search_accommodations_tool(city: str) -> str:
        """Search for accommodations (hotels, apartments) in a city."""
        results = search_accommodations(db, city)
        if results:
            gathered.accommodations[city] = results
        return _serialize(results)

    @tool
    def search_attractions_tool(city: str) -> str:
        """Search for tourist attractions in a city."""
        results = search_attractions(db, city)
        if results:
            gathered.attractions[city] = results
        return _serialize(results)

    @tool
    def get_distance_tool(origin: str, destination: str, mode: str = "self-driving") -> str:
        """Get distance, duration, and travel cost between two cities. Mode can be 'self-driving' or 'taxi'."""
        result = get_distance(db, origin, destination, mode)
        if result:
            key = f"{origin}->{destination} ({mode})"
            gathered.distances[key] = result
        return _serialize(result)

    @tool
    def search_cities_tool(state: str) -> str:
        """Get list of cities in a US state."""
        results = search_cities(db, state)
        if results:
            gathered.cities[state] = results
        return json.dumps(results)

    return [
        search_flights_tool,
        search_restaurants_tool,
        search_accommodations_tool,
        search_attractions_tool,
        get_distance_tool,
        search_cities_tool,
    ]


# ---------------------------------------------------------------------------
# CrewAI tool wrappers
# ---------------------------------------------------------------------------


def make_crewai_tools(
    db: ReferenceDatabase,
    gathered: GatheredData,
) -> list:
    """Create CrewAI-compatible tools that auto-accumulate into GatheredData.

    Requires: crewai
    """
    from crewai.tools import tool as crewai_tool

    @crewai_tool("Search Flights")
    def search_flights_tool(origin: str, destination: str, date: str) -> str:
        """Search flights from origin city to destination city on a specific date (YYYY-MM-DD format).
        Returns a JSON list of flight objects with fields: flight_number, price, dep_time, arr_time, origin, destination."""
        results = search_flights(db, origin, destination, date)
        if results:
            key = f"{origin}->{destination} on {date}"
            gathered.flights[key] = results
        return _serialize(results)

    @crewai_tool("Search Restaurants")
    def search_restaurants_tool(city: str) -> str:
        """Search for restaurants in a city.
        Returns a JSON list of restaurant objects with fields: name, average_cost, cuisines, rating, city."""
        results = search_restaurants(db, city)
        if results:
            gathered.restaurants[city] = results
        return _serialize(results)

    @crewai_tool("Search Accommodations")
    def search_accommodations_tool(city: str) -> str:
        """Search for accommodations in a city.
        Returns a JSON list with fields: name, price, room_type, house_rules, min_nights, max_occupancy, city."""
        results = search_accommodations(db, city)
        if results:
            gathered.accommodations[city] = results
        return _serialize(results)

    @crewai_tool("Search Attractions")
    def search_attractions_tool(city: str) -> str:
        """Search for tourist attractions in a city.
        Returns a JSON list with fields: name, latitude, longitude, address, phone, website, city."""
        results = search_attractions(db, city)
        if results:
            gathered.attractions[city] = results
        return _serialize(results)

    @crewai_tool("Get Distance")
    def get_distance_tool(origin: str, destination: str, mode: str = "self-driving") -> str:
        """Get distance, duration, and travel cost between two cities. Mode is 'self-driving' or 'taxi'.
        Returns a JSON object with fields: duration, distance, cost, mode, origin, destination."""
        result = get_distance(db, origin, destination, mode)
        if result:
            key = f"{origin}->{destination} ({mode})"
            gathered.distances[key] = result
        return _serialize(result)

    @crewai_tool("Search Cities")
    def search_cities_tool(state: str) -> str:
        """Get list of cities in a US state. Returns a JSON list of city names."""
        results = search_cities(db, state)
        if results:
            gathered.cities[state] = results
        return json.dumps(results)

    return [
        search_flights_tool,
        search_restaurants_tool,
        search_accommodations_tool,
        search_attractions_tool,
        get_distance_tool,
        search_cities_tool,
    ]
