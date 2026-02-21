"""Data models for the TravelPlanner benchmark."""

from __future__ import annotations

from typing import Any, Self

from opensymbolicai.models import GoalContext
from pydantic import BaseModel, Field

from travelplanner_bench.constants import SELF_DRIVING
from travelplanner_bench.utils import parse_cost


# =========================================================================
# Entity models (parsed from reference database)
# =========================================================================


class Flight(BaseModel):
    """A single flight record from the reference database."""

    flight_number: str = ""
    price: float = 0.0
    dep_time: str = ""
    arr_time: str = ""
    elapsed_time: str = ""
    date: str = ""
    origin: str = ""
    destination: str = ""
    distance: str = ""

    _RAW_FIELD_MAP: dict[str, str] = {
        "Flight Number": "flight_number",
        "Price": "price",
        "DepTime": "dep_time",
        "ArrTime": "arr_time",
        "ActualElapsedTime": "elapsed_time",
        "FlightDate": "date",
        "OriginCityName": "origin",
        "DestCityName": "destination",
        "Distance": "distance",
    }

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> Self:
        return cls(
            flight_number=raw.get("Flight Number", "").strip(),
            price=parse_cost(raw.get("Price", "0")),
            dep_time=raw.get("DepTime", "").strip(),
            arr_time=raw.get("ArrTime", "").strip(),
            elapsed_time=raw.get("ActualElapsedTime", "").strip(),
            date=raw.get("FlightDate", "").strip(),
            origin=raw.get("OriginCityName", "").strip(),
            destination=raw.get("DestCityName", "").strip(),
            distance=raw.get("Distance", "").strip(),
        )

    def __getitem__(self, key: str) -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, "")

    def get(self, key: str, default: Any = "") -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, default)


class Restaurant(BaseModel):
    """A single restaurant record from the reference database."""

    name: str = ""
    average_cost: float = 0.0
    cuisines: str = ""
    rating: str = ""
    city: str = ""

    _RAW_FIELD_MAP: dict[str, str] = {
        "Name": "name",
        "Average Cost": "average_cost",
        "Cuisines": "cuisines",
        "Aggregate Rating": "rating",
        "City": "city",
    }

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> Self:
        return cls(
            name=raw.get("Name", "").strip(),
            average_cost=parse_cost(raw.get("Average Cost", "0")),
            cuisines=raw.get("Cuisines", "").strip(),
            rating=raw.get("Aggregate Rating", "").strip(),
            city=raw.get("City", "").strip(),
        )

    def cuisine_set(self) -> set[str]:
        """Return cuisines as a lowercase set for matching."""
        return {c.strip().lower() for c in self.cuisines.split(",") if c.strip()}

    def __getitem__(self, key: str) -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, "")

    def get(self, key: str, default: Any = "") -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, default)


class Accommodation(BaseModel):
    """A single accommodation record from the reference database."""

    name: str = ""
    price: float = 0.0
    room_type: str = ""
    house_rules: str = ""
    min_nights: int = 1
    max_occupancy: int = 1
    review_rate: str = ""
    city: str = ""

    _RAW_FIELD_MAP: dict[str, str] = {
        "NAME": "name",
        "Name": "name",
        "price": "price",
        "room type": "room_type",
        "room_type": "room_type",
        "house_rules": "house_rules",
        "minimum nights": "min_nights",
        "minimum_nights": "min_nights",
        "maximum occupancy": "max_occupancy",
        "review rate number": "review_rate",
        "city": "city",
        "City": "city",
    }

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> Self:
        min_n = 1
        try:
            min_n = int(raw.get("minimum nights", raw.get("minimum_nights", "1")).strip())
        except (ValueError, TypeError):
            pass
        max_o = 1
        try:
            max_o = int(raw.get("maximum occupancy", "1").strip())
        except (ValueError, TypeError):
            pass
        return cls(
            name=raw.get("NAME", raw.get("Name", "")).strip(),
            price=parse_cost(raw.get("price", "0")),
            room_type=raw.get("room type", raw.get("room_type", "")).strip(),
            house_rules=raw.get("house_rules", "").strip(),
            min_nights=min_n,
            max_occupancy=max_o,
            review_rate=raw.get("review rate number", "").strip(),
            city=raw.get("city", raw.get("City", "")).strip(),
        )

    def __getitem__(self, key: str) -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, "")

    def get(self, key: str, default: Any = "") -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, default)


class Attraction(BaseModel):
    """A single attraction record from the reference database."""

    name: str = ""
    latitude: str = ""
    longitude: str = ""
    address: str = ""
    phone: str = ""
    website: str = ""
    city: str = ""

    _RAW_FIELD_MAP: dict[str, str] = {
        "Name": "name",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Address": "address",
        "Phone": "phone",
        "Website": "website",
        "City": "city",
    }

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> Self:
        return cls(
            name=raw.get("Name", "").strip(),
            latitude=raw.get("Latitude", "").strip(),
            longitude=raw.get("Longitude", "").strip(),
            address=raw.get("Address", "").strip(),
            phone=raw.get("Phone", "").strip(),
            website=raw.get("Website", "").strip(),
            city=raw.get("City", "").strip(),
        )

    def __getitem__(self, key: str) -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, "")

    def get(self, key: str, default: Any = "") -> Any:
        attr = self._RAW_FIELD_MAP.get(key, key)
        return getattr(self, attr, default)


class DistanceInfo(BaseModel):
    """Distance, duration, and cost between two cities."""

    duration: str = ""
    distance: str = ""
    cost: float = 0.0
    mode: str = SELF_DRIVING
    origin: str = ""
    destination: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, str]) -> Self:
        return cls(
            duration=raw.get("duration", "").strip(),
            distance=raw.get("distance", "").strip(),
            cost=parse_cost(raw.get("cost", "0")),
            mode=raw.get("mode", SELF_DRIVING).strip(),
            origin=raw.get("origin", "").strip(),
            destination=raw.get("destination", "").strip(),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "")

    def get(self, key: str, default: Any = "") -> Any:
        return getattr(self, key, default)


# =========================================================================
# Transport result models
# =========================================================================


class ValidTransport(BaseModel):
    """Result of filtering transport options by constraint."""

    flights: list[Flight] = Field(default_factory=list)
    distances: list[DistanceInfo] = Field(default_factory=list)


class TransportPlan(BaseModel):
    """Selected transport for an entire trip (outbound + return legs)."""

    outbound_str: str = "-"
    return_str: str = "-"
    outbound_flight: Flight | None = None
    return_flight: Flight | None = None
    driving_costs: list[float] = Field(default_factory=list)
    mode: str = ""


# =========================================================================
# Constraint model
# =========================================================================


class LocalConstraint(BaseModel):
    """Parsed local constraints for a task."""

    cuisine: list[str] | None = None
    room_type: str | None = None
    room_rule: str | None = None
    transportation: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Self:
        cuisine_val = raw.get("cuisine")
        cuisine_list: list[str] | None = None
        if cuisine_val:
            if isinstance(cuisine_val, str):
                cuisine_list = [c.strip() for c in cuisine_val.split(",") if c.strip()]
            elif isinstance(cuisine_val, list):
                cuisine_list = cuisine_val
        return cls(
            cuisine=cuisine_list,
            room_type=raw.get("room_type") or None,
            room_rule=raw.get("room_rule") or None,
            transportation=raw.get("transportation") or None,
        )


# =========================================================================
# Task model
# =========================================================================


class TravelPlannerTask(BaseModel):
    """A single TravelPlanner benchmark task."""

    task_id: str = Field(..., description="Index-based task ID")
    query: str = Field(..., description="Natural language travel request")
    org: str = Field(..., description="Origin city")
    dest: str = Field(..., description="Destination city or state")
    days: int = Field(..., description="Trip duration (3, 5, or 7)")
    date: list[str] = Field(default_factory=list, description="Travel dates")
    level: str = Field(default="easy", description="Difficulty: easy, medium, hard")
    visiting_city_number: int = Field(default=1, description="Number of cities to visit")
    people_number: int = Field(default=1, description="Number of travelers")
    local_constraint: LocalConstraint = Field(
        default_factory=LocalConstraint,
        description="Parsed constraints: cuisine, room_type, room_rule, transportation",
    )
    budget: int = Field(default=0, description="Total budget in dollars")
    reference_information: list[dict[str, str]] = Field(
        default_factory=list, description="Reference info entries with Description/Content",
    )
    annotated_plan: list[dict[str, Any]] | None = Field(
        default=None, description="Ground truth plan (None for test split)",
    )


class DayPlan(BaseModel):
    """A single day in the travel itinerary."""

    days: int = Field(..., description="Day number")
    current_city: str = Field(..., description="'from X to Y' or city name")
    transportation: str = Field(default="-", description="Flight info, taxi, self-driving, or '-'")
    breakfast: str = Field(default="-", description="Restaurant name or '-'")
    attraction: str = Field(default="-", description="Semicolon-separated attraction names or '-'")
    lunch: str = Field(default="-", description="Restaurant name or '-'")
    dinner: str = Field(default="-", description="Restaurant name or '-'")
    accommodation: str = Field(default="-", description="Accommodation name or '-'")


# =========================================================================
# Gathered Data (output of retrieval, input to plan assembler)
# =========================================================================


class GatheredData(BaseModel):
    """Structured container for all retrieved data from the reference database.

    Organized by city for easy access during plan assembly.
    """

    # Flights keyed by route string (e.g., "Chicago->NYC on 2022-03-16")
    flights: dict[str, list[Flight]] = Field(default_factory=dict)
    # Restaurants keyed by city name
    restaurants: dict[str, list[Restaurant]] = Field(default_factory=dict)
    # Accommodations keyed by city name
    accommodations: dict[str, list[Accommodation]] = Field(default_factory=dict)
    # Attractions keyed by city name
    attractions: dict[str, list[Attraction]] = Field(default_factory=dict)
    # Distances keyed by route string (e.g., "Philly->Pittsburgh (self-driving)")
    distances: dict[str, DistanceInfo] = Field(default_factory=dict)
    # Cities keyed by state name
    cities: dict[str, list[str]] = Field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary for prompt inclusion."""
        parts: list[str] = []
        total_flights = sum(len(v) for v in self.flights.values())
        if total_flights:
            routes = ", ".join(self.flights.keys())
            parts.append(f"Flights: {total_flights} across routes [{routes}]")
        for city, rests in self.restaurants.items():
            parts.append(f"Restaurants in {city}: {len(rests)}")
        for city, accs in self.accommodations.items():
            parts.append(f"Accommodations in {city}: {len(accs)}")
        for city, attrs in self.attractions.items():
            parts.append(f"Attractions in {city}: {len(attrs)}")
        if self.distances:
            parts.append(f"Distances: {len(self.distances)} routes")
        for state, city_list in self.cities.items():
            parts.append(f"Cities in {state}: {city_list}")
        return "\n".join(parts) if parts else "No data gathered yet."


# =========================================================================
# Contexts (introspection boundaries)
# =========================================================================


class RetrievalContext(GoalContext):
    """Context for the RetrievalAgent (introspection boundary).

    Tracks what data has been gathered so the evaluator can determine
    when enough information is available.
    """

    # Task metadata needed for evaluator
    org: str = ""
    dest: str = ""
    days: int = 0
    date: list[str] = Field(default_factory=list)
    visiting_city_number: int = 1

    # Gathered data counts (for evaluator / prompt context)
    has_outbound_flights: bool = False
    has_return_flights: bool = False
    destination_cities: list[str] = Field(default_factory=list)
    restaurants_per_city: dict[str, int] = Field(default_factory=dict)
    accommodations_per_city: dict[str, int] = Field(default_factory=dict)
    attractions_per_city: dict[str, int] = Field(default_factory=dict)
    has_distances: bool = False
    has_cities_list: bool = False

    # The actual gathered data (accumulated across iterations)
    gathered: GatheredData = Field(default_factory=GatheredData)


class TravelPlanContext(GoalContext):
    """Context for the top-level TravelPlannerAgent (introspection boundary).

    Tracks orchestration state: have we gathered data? Have we built a plan?
    """

    # Task metadata
    query: str = ""
    org: str = ""
    dest: str = ""
    days: int = 0
    people_number: int = 1
    budget: int = 0
    local_constraint: LocalConstraint = Field(default_factory=LocalConstraint)

    # Phase tracking
    data_gathered: bool = False
    gathered_summary: str = ""
    plan_built: bool = False
    current_plan: list[dict[str, Any]] | None = None
    plan_complete: bool = False
    solver_error: str | None = None


# =========================================================================
# Iteration logging models
# =========================================================================


class StepLog(BaseModel):
    """Log of a single primitive execution step."""

    step: int = 0
    primitive: str = "?"
    args: str = ""
    result: str = ""
    time_seconds: float = 0.0
    success: bool = False
    error: str | None = None


class IterationLog(BaseModel):
    """Log of a single agent iteration (LLM call + execution)."""

    phase: str = ""
    iteration: int | None = None
    attempt: int | None = None
    prompt: str = ""
    response: str = ""
    extracted_code: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    time_seconds: float = 0.0
    steps: list[StepLog] = Field(default_factory=list)
    goal_achieved: bool = False


# =========================================================================
# Result model
# =========================================================================


class TravelPlannerResult(BaseModel):
    """Result of evaluating a single TravelPlanner task."""

    task_id: str = Field(..., description="Task identifier")
    query: str = ""
    level: str = "easy"
    days: int = 3

    # Agent output
    plan: list[dict[str, Any]] | None = None
    plan_delivered: bool = False

    # Commonsense constraint results (8 checks)
    within_sandbox: bool = False
    complete_info: bool = False
    within_current_city: bool = False
    reasonable_city_route: bool = False
    diverse_restaurants: bool = False
    diverse_attractions: bool = False
    non_conflicting_transport: bool = False
    valid_accommodation: bool = False

    # Hard constraint results (None = not applicable)
    budget_ok: bool | None = None
    room_rule_ok: bool | None = None
    room_type_ok: bool | None = None
    cuisine_ok: bool | None = None
    transportation_ok: bool | None = None

    # Aggregate scores
    commonsense_micro: float = 0.0
    commonsense_macro: bool = False
    hard_micro: float = 0.0
    hard_macro: bool = False
    final_pass: bool = False

    # Execution metadata
    model: str = ""
    framework: str = "opensymbolicai"
    iterations: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
