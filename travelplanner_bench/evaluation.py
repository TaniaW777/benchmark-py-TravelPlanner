"""Evaluation for the TravelPlanner benchmark.

Implements 8 commonsense constraint checks, 5 hard constraint checks,
and aggregate scoring following the original TravelPlanner paper.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from travelplanner_bench.constants import (
    ACCOMMODATION,
    ATTRACTION,
    CURRENT_CITY,
    DAY_PLAN_FIELDS,
    FLIGHT,
    MEAL_KEYS,
    NO_DATA,
    SELF_DRIVING,
    TRANSPORTATION,
)
from travelplanner_bench.models import TravelPlannerResult, TravelPlannerTask
from travelplanner_bench.tools import ReferenceDatabase
from travelplanner_bench.utils import (
    extract_name,
    extract_names,
    name_in_set,
    normalize_name,
    parse_cost,
    strip_city_suffix,
)

log = logging.getLogger(__name__)


# ===========================================================================
# Helper utilities
# ===========================================================================


def _get_current_city(day: dict[str, Any]) -> str | None:
    """Extract the current city from a day entry.

    Handles both "CityName" and "from X to Y" formats.
    For "from X to Y", returns Y (the destination city for that day).
    """
    val = day.get(CURRENT_CITY, "")
    if not val or val.strip() == NO_DATA:
        return None
    val = val.strip()
    match = re.match(r"from\s+(.+?)\s+to\s+(.+)", val, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return val


# ===========================================================================
# Commonsense Constraints (8 checks)
# ===========================================================================


def check_within_sandbox(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool:
    """Check that all entities in the plan exist in the reference database."""
    for day in plan:
        # Check transportation (flights)
        transport = day.get(TRANSPORTATION, NO_DATA)
        if transport and transport.strip() != NO_DATA:
            flight_match = re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
            if flight_match:
                fn = flight_match.group(1).strip()
                if fn and not name_in_set(fn, db.all_flight_numbers):
                    log.debug("Sandbox fail: flight %s not found", fn)
                    return False

        # Check restaurants
        for meal_key in MEAL_KEYS:
            name = extract_name(day.get(meal_key, NO_DATA))
            if name and not name_in_set(name, db.all_restaurant_names):
                base_name = strip_city_suffix(name)
                if not name_in_set(base_name, db.all_restaurant_names):
                    log.debug("Sandbox fail: restaurant %r not found", name)
                    return False

        # Check accommodations
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if acc_name and not name_in_set(acc_name, db.all_accommodation_names):
            base_name = strip_city_suffix(acc_name)
            if not name_in_set(base_name, db.all_accommodation_names):
                log.debug("Sandbox fail: accommodation %r not found", acc_name)
                return False

        # Check attractions
        for attr_name in extract_names(day.get(ATTRACTION, NO_DATA)):
            if not name_in_set(attr_name, db.all_attraction_names):
                base_name = strip_city_suffix(attr_name)
                if not name_in_set(base_name, db.all_attraction_names):
                    log.debug("Sandbox fail: attraction %r not found", attr_name)
                    return False

    return True


def check_complete_info(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool:
    """Check that plan has complete information.

    Each day should have reasonable coverage of required fields.
    """
    if len(plan) != task.days:
        return False

    for day in plan:
        fields = DAY_PLAN_FIELDS[1:]  # exclude CURRENT_CITY
        filled = sum(
            1 for f in fields
            if day.get(f, NO_DATA).strip() not in ("", NO_DATA)
        )
        day_num = day.get("days", 0)
        if day_num == task.days:
            # Last day: may only have transportation (return flight)
            if filled < 1:
                return False
        elif day_num == 1:
            # First day: arrival, may skip breakfast
            if filled < 2:
                return False
        else:
            if filled < 3:
                return False

    return True


def check_within_current_city(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool:
    """Check that daily activities are in the designated city."""
    for day in plan:
        city = _get_current_city(day)
        if not city:
            continue

        city_lower = normalize_name(city)

        # Check restaurants
        for meal_key in MEAL_KEYS:
            name = extract_name(day.get(meal_key, NO_DATA))
            if not name:
                continue
            base_name = strip_city_suffix(name)
            db_cities = db.restaurant_city.get(normalize_name(base_name), set())
            if db_cities and city_lower not in {normalize_name(c) for c in db_cities}:
                log.debug(
                    "City fail: restaurant %r in %r, expected %r",
                    base_name, db_cities, city,
                )
                return False

        # Check attractions
        for attr_name in extract_names(day.get(ATTRACTION, NO_DATA)):
            base_name = strip_city_suffix(attr_name)
            db_cities = db.attraction_city.get(normalize_name(base_name), set())
            if db_cities and city_lower not in {normalize_name(c) for c in db_cities}:
                log.debug(
                    "City fail: attraction %r in %r, expected %r",
                    base_name, db_cities, city,
                )
                return False

        # Check accommodation
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if acc_name:
            base_name = strip_city_suffix(acc_name)
            db_cities = db.accommodation_city.get(normalize_name(base_name), set())
            if db_cities and city_lower not in {normalize_name(c) for c in db_cities}:
                log.debug(
                    "City fail: accommodation %r in %r, expected %r",
                    base_name, db_cities, city,
                )
                return False

    return True


def check_reasonable_city_route(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool:
    """Check that the city visiting sequence is reasonable."""
    if not plan:
        return False

    cities_visited: list[str] = []
    last_day = plan[-1]
    for day in plan:
        cc = day.get(CURRENT_CITY, "")
        if not cc:
            return False
        match = re.match(r"from\s+(.+?)\s+to\s+(.+)", cc, re.IGNORECASE)
        if match:
            if not cities_visited:
                cities_visited.append(match.group(1).strip())
            cities_visited.append(match.group(2).strip())
        else:
            cities_visited.append(cc.strip())

    if not cities_visited:
        return False

    # Must start from origin
    if normalize_name(cities_visited[0]) != normalize_name(task.org):
        log.debug("Route fail: starts from %r, expected %r", cities_visited[0], task.org)
        return False

    # Last day must return to origin
    last_city = last_day.get(CURRENT_CITY, "")
    match = re.match(r"from\s+(.+?)\s+to\s+(.+)", last_city, re.IGNORECASE)
    if match:
        if normalize_name(match.group(2).strip()) != normalize_name(task.org):
            log.debug("Route fail: doesn't return to origin")
            return False

    return True


def check_diverse_restaurants(
    plan: list[dict[str, Any]],
) -> bool:
    """Check that no restaurant appears more than once across all days."""
    seen: set[str] = set()
    for day in plan:
        for meal_key in MEAL_KEYS:
            name = extract_name(day.get(meal_key, NO_DATA))
            if not name:
                continue
            key = normalize_name(strip_city_suffix(name))
            if key in seen:
                log.debug("Diversity fail: duplicate restaurant %r", name)
                return False
            seen.add(key)
    return True


def check_diverse_attractions(
    plan: list[dict[str, Any]],
) -> bool:
    """Check that no attraction appears more than once across all days."""
    seen: set[str] = set()
    for day in plan:
        for attr_name in extract_names(day.get(ATTRACTION, NO_DATA)):
            key = normalize_name(strip_city_suffix(attr_name))
            if key in seen:
                log.debug("Diversity fail: duplicate attraction %r", attr_name)
                return False
            seen.add(key)
    return True


def check_non_conflicting_transport(
    plan: list[dict[str, Any]],
) -> bool:
    """Check that flights and self-driving are not both used for inter-city travel."""
    has_flight = False
    has_self_driving = False

    for day in plan:
        transport = day.get(TRANSPORTATION, NO_DATA)
        if not transport or transport.strip() == NO_DATA:
            continue
        t_lower = transport.lower()
        if FLIGHT in t_lower:
            has_flight = True
        if SELF_DRIVING in t_lower or "self driving" in t_lower:
            has_self_driving = True

    if has_flight and has_self_driving:
        log.debug("Transport conflict: both flight and self-driving used")
        return False
    return True


def check_valid_accommodation(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
) -> bool:
    """Check that accommodation minimum nights requirements are met."""
    acc_nights: dict[str, int] = {}
    for day in plan:
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if acc_name:
            key = normalize_name(strip_city_suffix(acc_name))
            acc_nights[key] = acc_nights.get(key, 0) + 1

    for acc_key, nights in acc_nights.items():
        for city_accs in db.accommodations.values():
            for acc in city_accs:
                if normalize_name(acc.name) == acc_key:
                    if nights < acc.min_nights:
                        log.debug(
                            "Accommodation fail: %r stayed %d nights, minimum %d",
                            acc.name, nights, acc.min_nights,
                        )
                        return False
    return True


# ===========================================================================
# Hard Constraints (5 checks)
# ===========================================================================


def check_budget(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that total trip cost is within budget."""
    if not task.budget:
        return None

    total_cost = 0.0
    people = task.people_number or 1

    for day in plan:
        # Flight costs
        transport = day.get(TRANSPORTATION, NO_DATA)
        if transport and transport.strip() != NO_DATA:
            flight_match = re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
            if flight_match:
                fn = flight_match.group(1).strip()
                total_cost += _find_flight_cost(db, fn) * people

            # Self-driving/taxi cost from transport description
            cost_match = re.search(r"[Cc]ost:?\s*\$?([\d,.]+)", transport)
            if cost_match and not flight_match:
                total_cost += parse_cost(cost_match.group(1))

        # Meal costs
        for meal_key in MEAL_KEYS:
            name = extract_name(day.get(meal_key, NO_DATA))
            if name:
                base_name = strip_city_suffix(name)
                total_cost += _find_restaurant_cost(db, base_name) * people

        # Accommodation costs
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if acc_name:
            base_name = strip_city_suffix(acc_name)
            total_cost += _find_accommodation_cost(db, base_name)

    return total_cost <= task.budget


def _find_flight_cost(db: ReferenceDatabase, flight_number: str) -> float:
    """Find cost of a flight by flight number."""
    fn_lower = normalize_name(flight_number)
    for flights in db.flights.values():
        for flight in flights:
            if normalize_name(flight.flight_number) == fn_lower:
                return flight.price
    return 0.0


def _find_restaurant_cost(db: ReferenceDatabase, name: str) -> float:
    """Find average cost of a restaurant by name."""
    n = normalize_name(name)
    for restaurants in db.restaurants.values():
        for rest in restaurants:
            if normalize_name(rest.name) == n:
                return rest.average_cost
    return 0.0


def _find_accommodation_cost(db: ReferenceDatabase, name: str) -> float:
    """Find per-night cost of an accommodation by name."""
    n = normalize_name(name)
    for accs in db.accommodations.values():
        for acc in accs:
            if normalize_name(acc.name) == n:
                return acc.price
    return 0.0


def check_room_rule(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check accommodation house rules compliance."""
    rule = task.local_constraint.room_rule
    if not rule:
        return None

    rule_lower = rule.lower().strip()

    for day in plan:
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if not acc_name:
            continue
        base_name = strip_city_suffix(acc_name)
        house_rules = _find_accommodation_field(db, base_name, "house_rules")
        if house_rules:
            if rule_lower not in house_rules.lower():
                log.debug(
                    "Room rule fail: %r rules=%r, need %r",
                    base_name, house_rules, rule,
                )
                return False
    return True


def check_room_type(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that accommodation room type matches requirement."""
    required_type = task.local_constraint.room_type
    if not required_type:
        return None

    required_lower = required_type.lower().strip()
    is_negation = required_lower.startswith("not ")

    for day in plan:
        acc_name = extract_name(day.get(ACCOMMODATION, NO_DATA))
        if not acc_name:
            continue
        base_name = strip_city_suffix(acc_name)
        actual_type = _find_accommodation_field(db, base_name, "room_type")
        if not actual_type:
            continue
        actual_lower = actual_type.lower().strip()

        if is_negation:
            forbidden = required_lower[4:].strip()
            if forbidden in actual_lower:
                log.debug(
                    "Room type fail: %r is %r, forbidden %r",
                    base_name, actual_type, forbidden,
                )
                return False
        else:
            if required_lower not in actual_lower:
                log.debug(
                    "Room type fail: %r is %r, need %r",
                    base_name, actual_type, required_type,
                )
                return False
    return True


def _find_accommodation_field(
    db: ReferenceDatabase, name: str, field: str
) -> str | None:
    """Find a specific field value for an accommodation by name."""
    n = normalize_name(name)
    for accs in db.accommodations.values():
        for acc in accs:
            if normalize_name(acc.name) == n:
                return getattr(acc, field, None)
    return None


def check_cuisine(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that all required cuisines are represented in meals."""
    required_cuisines = task.local_constraint.cuisine
    if not required_cuisines:
        return None

    required_set = {c.lower().strip() for c in required_cuisines}

    # Collect all cuisines from restaurants in the plan
    found_cuisines: set[str] = set()
    for day in plan:
        for meal_key in MEAL_KEYS:
            name = extract_name(day.get(meal_key, NO_DATA))
            if not name:
                continue
            base_name = strip_city_suffix(name)
            cuisines = _find_restaurant_cuisines(db, base_name)
            for c in cuisines:
                found_cuisines.add(c.lower().strip())

    for req in required_set:
        if not any(req in fc for fc in found_cuisines):
            log.debug("Cuisine fail: %r not found in meals", req)
            return False
    return True


def _find_restaurant_cuisines(db: ReferenceDatabase, name: str) -> list[str]:
    """Find cuisines offered by a restaurant."""
    n = normalize_name(name)
    for restaurants in db.restaurants.values():
        for rest in restaurants:
            if normalize_name(rest.name) == n:
                return [c.strip() for c in rest.cuisines.split(",") if c.strip()]
    return []


def check_transportation_constraint(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool | None:
    """Check hard transportation constraint (e.g., 'no flight', 'no self-driving')."""
    transport_constraint = task.local_constraint.transportation
    if not transport_constraint:
        return None

    constraint_lower = transport_constraint.lower().strip()

    for day in plan:
        transport = day.get(TRANSPORTATION, NO_DATA)
        if not transport or transport.strip() == NO_DATA:
            continue
        t_lower = transport.lower()

        if "no flight" in constraint_lower and FLIGHT in t_lower:
            log.debug("Transport constraint fail: flight used but forbidden")
            return False
        if f"no {SELF_DRIVING}" in constraint_lower and (
            SELF_DRIVING in t_lower or "self driving" in t_lower
        ):
            log.debug("Transport constraint fail: self-driving used but forbidden")
            return False

    return True


# ===========================================================================
# Aggregate Scoring
# ===========================================================================


def evaluate_plan(
    plan: list[dict[str, Any]] | None,
    task: TravelPlannerTask,
    db: ReferenceDatabase,
) -> TravelPlannerResult:
    """Run all constraint checks and compute aggregate scores."""
    result = TravelPlannerResult(
        task_id=task.task_id,
        query=task.query,
        level=task.level,
        days=task.days,
    )

    if plan is None or len(plan) == 0:
        result.plan_delivered = False
        return result

    result.plan_delivered = True
    result.plan = plan

    # Commonsense checks (8)
    result.within_sandbox = check_within_sandbox(plan, db, task)
    result.complete_info = check_complete_info(plan, task)
    result.within_current_city = check_within_current_city(plan, db, task)
    result.reasonable_city_route = check_reasonable_city_route(plan, task)
    result.diverse_restaurants = check_diverse_restaurants(plan)
    result.diverse_attractions = check_diverse_attractions(plan)
    result.non_conflicting_transport = check_non_conflicting_transport(plan)
    result.valid_accommodation = check_valid_accommodation(plan, db)

    commonsense_checks = [
        result.within_sandbox,
        result.complete_info,
        result.within_current_city,
        result.reasonable_city_route,
        result.diverse_restaurants,
        result.diverse_attractions,
        result.non_conflicting_transport,
        result.valid_accommodation,
    ]
    result.commonsense_micro = sum(commonsense_checks) / len(commonsense_checks)
    result.commonsense_macro = all(commonsense_checks)

    # Hard checks (5)
    result.budget_ok = check_budget(plan, db, task)
    result.room_rule_ok = check_room_rule(plan, db, task)
    result.room_type_ok = check_room_type(plan, db, task)
    result.cuisine_ok = check_cuisine(plan, db, task)
    result.transportation_ok = check_transportation_constraint(plan, task)

    hard_checks = [
        v
        for v in [
            result.budget_ok,
            result.room_rule_ok,
            result.room_type_ok,
            result.cuisine_ok,
            result.transportation_ok,
        ]
        if v is not None
    ]

    if hard_checks:
        result.hard_micro = sum(hard_checks) / len(hard_checks)
        result.hard_macro = all(hard_checks)
    else:
        result.hard_micro = 1.0
        result.hard_macro = True

    result.final_pass = result.commonsense_macro and result.hard_macro
    return result


def compute_aggregate_metrics(
    results: list[TravelPlannerResult],
) -> dict[str, Any]:
    """Compute benchmark-level aggregate metrics."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    delivered = [r for r in results if r.plan_delivered]
    delivery_rate = len(delivered) / total

    cs_micro_avg = (
        sum(r.commonsense_micro for r in delivered) / len(delivered)
        if delivered
        else 0.0
    )
    cs_macro_rate = (
        sum(1 for r in delivered if r.commonsense_macro) / len(delivered)
        if delivered
        else 0.0
    )

    hard_micro_avg = (
        sum(r.hard_micro for r in delivered) / len(delivered)
        if delivered
        else 0.0
    )
    hard_macro_rate = (
        sum(1 for r in delivered if r.hard_macro) / len(delivered)
        if delivered
        else 0.0
    )

    final_pass_rate = sum(1 for r in results if r.final_pass) / total

    per_level: dict[str, dict[str, Any]] = {}
    for r in results:
        lvl = r.level
        if lvl not in per_level:
            per_level[lvl] = {
                "total": 0,
                "delivered": 0,
                "commonsense_macro": 0,
                "hard_macro": 0,
                "final_pass": 0,
            }
        per_level[lvl]["total"] += 1
        if r.plan_delivered:
            per_level[lvl]["delivered"] += 1
        if r.commonsense_macro:
            per_level[lvl]["commonsense_macro"] += 1
        if r.hard_macro:
            per_level[lvl]["hard_macro"] += 1
        if r.final_pass:
            per_level[lvl]["final_pass"] += 1

    for lvl, stats in per_level.items():
        n = stats["total"]
        stats["delivery_rate"] = stats["delivered"] / n if n else 0.0
        stats["commonsense_macro_rate"] = stats["commonsense_macro"] / n if n else 0.0
        stats["hard_macro_rate"] = stats["hard_macro"] / n if n else 0.0
        stats["final_pass_rate"] = stats["final_pass"] / n if n else 0.0

    all_times = [r.wall_time_seconds for r in results]

    return {
        "total": total,
        "delivered": len(delivered),
        "delivery_rate": delivery_rate,
        "commonsense_micro_avg": cs_micro_avg,
        "commonsense_macro_rate": cs_macro_rate,
        "hard_micro_avg": hard_micro_avg,
        "hard_macro_rate": hard_macro_rate,
        "final_pass_rate": final_pass_rate,
        "per_level": per_level,
        "timing": {
            "total_seconds": sum(all_times),
            "avg_per_task": sum(all_times) / total if total else 0.0,
        },
        "errors": sum(1 for r in results if r.error),
    }
