"""CrewAI backend: Crew with Researcher + Planner agents."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from travelplanner_bench.backend import BackendResult, TokenUsage
from travelplanner_bench.models import GatheredData, TravelPlannerTask
from travelplanner_bench.tools import (
    ReferenceDatabase,
    get_distance,
    search_accommodations,
    search_attractions,
    search_cities,
    search_flights,
    search_restaurants,
)
from travelplanner_bench.tool_wrappers import make_crewai_tools

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MONKEY-PATCH 1 of 2: litellm.completion
#
# Some open-source models (notably gpt-oss-120b) append stray generation
# tokens like "<|call|>" to tool-call argument JSON. litellm doesn't strip
# these, so json.loads() fails inside CrewAI. This patch intercepts every
# completion response and removes <|...|> tokens from tool-call arguments
# before CrewAI sees them.
#
# Guarded by a flag to prevent double-patching.  Safe to remove once the
# upstream model or litellm handles this.
# ---------------------------------------------------------------------------

def _patch_litellm_for_tool_calls() -> None:
    """Patch litellm completion to clean ``<|call|>`` tokens from tool call args."""
    try:
        import litellm
    except ImportError:
        return

    if getattr(litellm, "_patched_tool_call_clean", False):
        return

    _orig_completion = litellm.completion

    def _clean_response(response: Any) -> Any:
        """Strip stray generation tokens from tool call arguments in-place."""
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if msg is None:
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            for tc in tool_calls:
                func = getattr(tc, "function", None)
                if func and hasattr(func, "arguments"):
                    raw = func.arguments
                    if isinstance(raw, str) and "<|" in raw:
                        cleaned = re.sub(r"\s*<\|[^>]*\|>\s*", "", raw).strip()
                        func.arguments = cleaned
        return response

    def _patched_completion(*args: Any, **kwargs: Any) -> Any:
        response = _orig_completion(*args, **kwargs)
        return _clean_response(response)

    litellm.completion = _patched_completion
    litellm._patched_tool_call_clean = True  # type: ignore[attr-defined]


class CrewAIBackend:
    """CrewAI Crew with Researcher + Planner agents for TravelPlanner benchmark."""

    def __init__(
        self,
        model: str,
        provider: str,
        max_iterations: int = 15,
        **kwargs: object,
    ) -> None:
        self._model = model
        self._provider = provider
        self._max_iterations = max_iterations

    @property
    def framework_name(self) -> str:
        return "crewai"

    def _get_llm_string(self) -> str:
        """Get LLM identifier string for CrewAI."""
        if self._provider == "openai":
            return f"openai/{self._model}"
        elif self._provider == "anthropic":
            return f"anthropic/{self._model}"
        elif self._provider == "fireworks":
            model = self._model
            if not model.startswith("accounts/"):
                model = f"accounts/fireworks/models/{model}"
            return f"fireworks_ai/{model}"
        elif self._provider == "groq":
            return f"groq/{self._model}"
        elif self._provider == "ollama":
            return f"ollama/{self._model}"
        return self._model

    @staticmethod
    def _patch_crewai_task_output() -> None:
        """MONKEY-PATCH 2 of 2: CrewAI TaskOutput.__init__.

        Some litellm/Fireworks models return tool-call lists (not strings) as
        the final task output, which triggers a Pydantic validation error in
        CrewAI's TaskOutput(raw=...).  This patch coerces non-string ``raw``
        values to str before Pydantic validates them.

        Guarded by a flag to prevent double-patching.  Safe to remove once
        CrewAI handles non-string raw values natively.
        """
        try:
            from crewai.tasks.task_output import TaskOutput as _TO

            if getattr(_TO, "_patched_raw_fix", False):
                return

            _orig_init = _TO.__init__

            def _patched_init(self_to: Any, *args: Any, **kwargs: Any) -> None:
                if "raw" in kwargs and not isinstance(kwargs["raw"], str):
                    kwargs["raw"] = str(kwargs["raw"])
                _orig_init(self_to, *args, **kwargs)

            _TO.__init__ = _patched_init  # type: ignore[method-assign]
            _TO._patched_raw_fix = True  # type: ignore[attr-defined]
        except Exception:
            pass

    def solve(self, task: TravelPlannerTask) -> BackendResult:
        from crewai import Agent, Crew, Process, Task

        self._patch_crewai_task_output()
        _patch_litellm_for_tool_calls()

        token_usage = TokenUsage()
        raw_logs: list[dict[str, Any]] = []
        start = time.perf_counter()

        try:
            db = ReferenceDatabase(task.reference_information)
            gathered = GatheredData()
            tools = make_crewai_tools(db, gathered)
            llm_string = self._get_llm_string()

            # Pre-resolve city names and routes
            city_names = search_cities(db, task.dest)
            cities_to_search = city_names if city_names else [task.dest]

            # Build explicit tool call list
            tool_calls_list = self._build_tool_calls_list(task, db, cities_to_search)

            # Scale iterations to the number of required tool calls
            num_tool_calls = len(tool_calls_list)
            researcher_max_iter = max(self._max_iterations, num_tool_calls * 2 + 5)
            planner_max_iter = max(self._max_iterations, task.days * 2 + 5)

            # --- Phase 1: Researcher Crew ---
            researcher = Agent(
                role="Travel Data Researcher",
                goal=(
                    "Call EVERY search tool listed in the task description. "
                    "Do NOT skip any tool calls. Do NOT make up data. "
                    "Only report data returned by the tools."
                ),
                backstory=(
                    "You are a meticulous travel researcher who always calls "
                    "every required search tool before reporting results."
                ),
                tools=tools,
                llm=llm_string,
                max_iter=researcher_max_iter,
                verbose=False,
            )

            research_description = self._build_research_task_description(
                task, tool_calls_list
            )
            raw_logs.append({"phase": "researcher_prompt", "description": research_description})

            research_task = Task(
                description=research_description,
                expected_output=(
                    "A structured summary of ALL data returned by the tools: "
                    "flight numbers and prices per route, restaurant names and "
                    "costs per city, accommodation names and prices per city, "
                    "attraction names per city, and distances."
                ),
                agent=researcher,
            )

            research_crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                process=Process.sequential,
                verbose=False,
            )
            retrieval_start = time.perf_counter()
            research_result = research_crew.kickoff()
            retrieval_time = time.perf_counter() - retrieval_start

            # Extract researcher token usage
            research_tokens = self._extract_crew_tokens(research_result)
            raw_logs.extend(self._capture_crew_logs(research_result, gathered))

            # Fill any gaps left by the researcher agent
            gaps = self._fill_retrieval_gaps(db, gathered, task, cities_to_search)
            if gaps:
                raw_logs.append({"phase": "gap_fill", "filled": gaps})

            # Log gathered data
            raw_logs.append({
                "phase": "tool_calls_summary",
                "gathered_data": self._gathered_summary(gathered),
            })

            # --- Phase 2: Planner Crew with structured data ---
            planner = Agent(
                role="Travel Planner",
                goal=(
                    f"Create a valid {task.days}-day travel plan from {task.org} "
                    f"to {task.dest} for {task.people_number} people within a "
                    f"${task.budget} budget. Use ONLY entity names from the "
                    f"data provided. Do NOT invent any names or numbers."
                ),
                backstory=(
                    "You are an expert travel planner. You ONLY use data provided "
                    "in the task description — you never make up flight numbers, "
                    "restaurant names, or other details."
                ),
                llm=llm_string,
                max_iter=planner_max_iter,
                verbose=False,
            )

            # Build planning prompt with structured data from gathered
            planning_description = self._build_planning_task_with_data(
                gathered, task, db
            )
            raw_logs.append({"phase": "planner_prompt", "description": planning_description})

            planning_task = Task(
                description=planning_description,
                expected_output=(
                    "A JSON array of day-plan objects. Each object has keys: "
                    "days, current_city, transportation, breakfast, attraction, "
                    "lunch, dinner, accommodation. ALL entity names must come "
                    "from the data provided in the task description."
                ),
                agent=planner,
            )

            planning_crew = Crew(
                agents=[planner],
                tasks=[planning_task],
                process=Process.sequential,
                verbose=False,
            )
            assembly_start = time.perf_counter()
            planning_result = planning_crew.kickoff()
            assembly_time = time.perf_counter() - assembly_start

            # Extract planner token usage
            planner_tokens = self._extract_crew_tokens(planning_result)

            # Populate phase-level token tracking
            token_usage.retrieval_input_tokens = research_tokens.input_tokens
            token_usage.retrieval_output_tokens = research_tokens.output_tokens
            token_usage.retrieval_llm_calls = research_tokens.llm_calls
            token_usage.retrieval_time_seconds = retrieval_time
            token_usage.assembly_input_tokens = planner_tokens.input_tokens
            token_usage.assembly_output_tokens = planner_tokens.output_tokens
            token_usage.assembly_llm_calls = planner_tokens.llm_calls
            token_usage.assembly_time_seconds = assembly_time

            # Aggregate totals
            token_usage.input_tokens = (
                research_tokens.input_tokens + planner_tokens.input_tokens
            )
            token_usage.output_tokens = (
                research_tokens.output_tokens + planner_tokens.output_tokens
            )
            token_usage.llm_calls = (
                research_tokens.llm_calls + planner_tokens.llm_calls
            )
            token_usage.time_seconds = retrieval_time + assembly_time
            token_usage.compute_totals()

            # Parse the plan from planner output
            output_text = str(planning_result)
            plan = self._parse_plan_json(output_text)

            elapsed = time.perf_counter() - start

            return BackendResult(
                framework="crewai",
                plan=plan,
                token_usage=token_usage,
                iterations=token_usage.llm_calls,
                wall_time_seconds=elapsed,
                raw_logs=raw_logs,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            log.exception("CrewAI solve failed: %s", e)
            return BackendResult(
                framework="crewai",
                wall_time_seconds=elapsed,
                error=str(e),
                raw_logs=raw_logs,
            )

    @staticmethod
    def _build_tool_calls_list(
        task: TravelPlannerTask,
        db: ReferenceDatabase,
        cities: list[str],
    ) -> list[str]:
        """Build explicit list of tool calls the researcher must make."""
        calls: list[str] = []
        n = 1
        for orig, dest, date in db.flights:
            calls.append(
                f'{n}. Search Flights(origin="{orig.title()}", '
                f'destination="{dest.title()}", date="{date}")'
            )
            n += 1
        for orig, dest, mode in db.distances:
            calls.append(
                f'{n}. Get Distance(origin="{orig.title()}", '
                f'destination="{dest.title()}", mode="{mode}")'
            )
            n += 1
        for city in cities:
            c = city.title()
            calls.append(f'{n}. Search Restaurants(city="{c}")')
            n += 1
            calls.append(f'{n}. Search Accommodations(city="{c}")')
            n += 1
            calls.append(f'{n}. Search Attractions(city="{c}")')
            n += 1
        return calls

    @staticmethod
    def _build_research_task_description(
        task: TravelPlannerTask,
        tool_calls: list[str],
    ) -> str:
        """Build research task description with explicit tool calls."""
        parts = [
            f"Gather all travel data for a {task.days}-day trip:",
            f"  Origin: {task.org}",
            f"  Destination: {task.dest}",
            f"  Dates: {task.date}",
            "",
            f"You MUST make ALL {len(tool_calls)} tool calls listed below.",
            "Do NOT skip any. Do NOT make up data. Call them NOW:",
            "",
        ]
        parts.extend(tool_calls)
        parts.extend([
            "",
            "After calling ALL tools, summarize the data returned by each tool.",
        ])
        return "\n".join(parts)

    @staticmethod
    def _fill_retrieval_gaps(
        db: ReferenceDatabase,
        gathered: GatheredData,
        task: TravelPlannerTask,
        cities: list[str],
    ) -> list[str]:
        """Programmatically fill data the researcher agent failed to gather."""
        filled: list[str] = []

        for city in cities:
            if city not in gathered.restaurants:
                results = search_restaurants(db, city)
                if results:
                    gathered.restaurants[city] = results
                    filled.append(f"restaurants:{city}")
            if city not in gathered.accommodations:
                results = search_accommodations(db, city)
                if results:
                    gathered.accommodations[city] = results
                    filled.append(f"accommodations:{city}")
            if city not in gathered.attractions:
                results = search_attractions(db, city)
                if results:
                    gathered.attractions[city] = results
                    filled.append(f"attractions:{city}")

        for orig, dest, date in db.flights:
            key = f"{orig.title()}->{dest.title()} on {date}"
            if key not in gathered.flights:
                results = search_flights(db, orig, dest, date)
                if results:
                    gathered.flights[key] = results
                    filled.append(f"flights:{key}")

        for orig, dest, mode in db.distances:
            key = f"{orig.title()}->{dest.title()} ({mode})"
            if key not in gathered.distances:
                result = get_distance(db, orig, dest, mode)
                if result:
                    gathered.distances[key] = result
                    filled.append(f"distance:{key}")

        if filled:
            log.info("Filled %d retrieval gaps: %s", len(filled), filled)
        return filled

    @staticmethod
    def _gathered_summary(gathered: GatheredData) -> dict[str, Any]:
        """Build a summary dict of gathered data for logging."""
        summary: dict[str, Any] = {}
        if gathered.flights:
            summary["flights"] = {k: len(v) for k, v in gathered.flights.items()}
        if gathered.restaurants:
            summary["restaurants"] = {k: len(v) for k, v in gathered.restaurants.items()}
        if gathered.accommodations:
            summary["accommodations"] = {k: len(v) for k, v in gathered.accommodations.items()}
        if gathered.attractions:
            summary["attractions"] = {k: len(v) for k, v in gathered.attractions.items()}
        if gathered.distances:
            summary["distances"] = list(gathered.distances.keys())
        return summary

    @staticmethod
    def _recommend_transport_mode(
        gathered: GatheredData, db: ReferenceDatabase
    ) -> str:
        """Determine whether to recommend flights or self-driving."""
        required_routes = set(db.flights.keys())
        if not required_routes:
            return "self-driving"
        for orig, dest, date in required_routes:
            key = f"{orig.title()}->{dest.title()} on {date}"
            flights = gathered.flights.get(key, [])
            if not flights:
                return "self-driving"
        return "flight"

    @staticmethod
    def _build_planning_task_with_data(
        gathered: GatheredData, task: TravelPlannerTask, db: ReferenceDatabase | None = None
    ) -> str:
        """Build planning task description with structured gathered data.

        This is the same format used by the LangChain assembly prompt —
        a clean, explicit list of all entity names the planner can use.
        This prevents the planner from hallucinating names.
        """
        parts = [
            f"Create a {task.days}-day travel plan using ONLY the data below.",
            f"Origin: {task.org}, Destination: {task.dest}",
            f"People: {task.people_number}, Budget: ${task.budget}",
            "",
            "IMPORTANT: Use ONLY entity names, flight numbers, costs, and details "
            "that appear in the data below. Do NOT invent or hallucinate any data.",
        ]

        constraints = []
        if task.local_constraint.room_type:
            constraints.append(f"Room type: {task.local_constraint.room_type}")
        if task.local_constraint.room_rule:
            constraints.append(f"Room rule: {task.local_constraint.room_rule}")
        if task.local_constraint.cuisine:
            constraints.append(f"Required cuisines: {', '.join(task.local_constraint.cuisine)}")
        if task.local_constraint.transportation:
            constraints.append(f"Transportation: {task.local_constraint.transportation}")
        if constraints:
            parts.append(f"Constraints: {'; '.join(constraints)}")

        # --- Flights ---
        parts.append("\n=== FLIGHTS ===")
        if gathered.flights:
            for route, flights in gathered.flights.items():
                sorted_f = sorted(flights, key=lambda f: f.price)
                parts.append(f"\nRoute: {route} ({len(flights)} flights)")
                for f in sorted_f[:5]:
                    parts.append(
                        f"  Flight {f.flight_number}: ${f.price}, "
                        f"dep {f.dep_time}, arr {f.arr_time}"
                    )
        else:
            parts.append("  (no flights found)")

        # --- Distances ---
        parts.append("\n=== DISTANCES ===")
        if gathered.distances:
            for route, dist in gathered.distances.items():
                parts.append(
                    f"  {route}: {dist.duration}, {dist.distance}, cost ${dist.cost}"
                )
        else:
            parts.append("  (no distance data)")

        # --- Restaurants ---
        parts.append("\n=== RESTAURANTS ===")
        for city, rests in gathered.restaurants.items():
            sorted_r = sorted(rests, key=lambda r: r.average_cost)
            parts.append(f"\n{city} ({len(rests)} restaurants, cheapest first):")
            for r in sorted_r[:15]:
                parts.append(f"  - {r.name} (${r.average_cost}, {r.cuisines})")

        # --- Accommodations ---
        parts.append("\n=== ACCOMMODATIONS ===")
        for city, accs in gathered.accommodations.items():
            sorted_a = sorted(accs, key=lambda a: a.price)
            parts.append(f"\n{city} ({len(accs)} accommodations, cheapest first):")
            for a in sorted_a[:10]:
                parts.append(
                    f"  - {a.name} (${a.price}/night, {a.room_type}, "
                    f"rules: {a.house_rules}, min nights: {a.min_nights})"
                )

        # --- Attractions ---
        parts.append("\n=== ATTRACTIONS ===")
        for city, attrs in gathered.attractions.items():
            parts.append(f"\n{city} ({len(attrs)} attractions):")
            for a in attrs[:10]:
                parts.append(f"  - {a.name}")

        # Determine recommended transport mode
        if db is not None:
            transport_mode = CrewAIBackend._recommend_transport_mode(gathered, db)
        else:
            transport_mode = "flight"
        if transport_mode == "self-driving":
            transport_rule = (
                "- TRANSPORT MODE: Use ONLY self-driving for ALL inter-city travel. "
                "Do NOT use any flights. This is because not all routes have "
                "flight service."
            )
        else:
            transport_rule = (
                "- TRANSPORT MODE: Use ONLY flights for ALL inter-city travel. "
                "Do NOT use self-driving. Pick the cheapest flight for each route."
            )

        parts.extend([
            "",
            "=== OUTPUT FORMAT ===",
            "Return a JSON array of day objects. Each day must have exactly these keys:",
            '  "days": <day number (int)>',
            '  "current_city": "<city name>" or "from <origin> to <destination>"',
            '  "transportation": "<transport details>" or "-"',
            '  "breakfast": "<Restaurant Name, City>" or "-"',
            '  "attraction": "<Attraction1, City;Attraction2, City>" or "-"',
            '  "lunch": "<Restaurant Name, City>" or "-"',
            '  "dinner": "<Restaurant Name, City>" or "-"',
            '  "accommodation": "<Accommodation Name, City>" or "-"',
            "",
            "RULES:",
            transport_rule,
            "- ANY travel day (current_city = 'from X to Y'): include transport, breakfast = '-', all lunch/dinner/attraction/accommodation MUST use DESTINATION city (Y) entities ONLY",
            "- Last day: current_city = 'from <last_city> to <origin>', include transport ONLY; set breakfast/lunch/dinner/attraction/accommodation = '-'",
            "- Non-travel days: current_city = city name, transportation = '-'",
            "- Use CHEAPEST options to stay within budget",
            "- Each restaurant must appear at most ONCE across all days",
            "- Each attraction must appear at most ONCE",
            "- Flight format: 'Flight Number: <num>, from <orig> to <dest>, Departure Time: <dep>, Arrival Time: <arr>'",
            "- Driving format: 'Self-driving, from <orig> to <dest>, Duration: <dur>, Distance: <dist>, Cost: <cost>'",
            "- ALL entity names must come from the data above. Do not invent names.",
            "",
            "Return ONLY the JSON array, no other text.",
        ])
        return "\n".join(parts)

    @staticmethod
    def _extract_crew_tokens(crew_result: Any) -> TokenUsage:
        """Extract token usage from CrewAI result."""
        usage = TokenUsage()

        metrics = getattr(crew_result, "token_usage", None)
        if metrics:
            usage.input_tokens = getattr(metrics, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(metrics, "completion_tokens", 0) or 0
            usage.total_tokens = getattr(metrics, "total_tokens", 0) or 0

        if not usage.total_tokens:
            raw_usage = getattr(crew_result, "usage_metrics", None)
            if raw_usage and isinstance(raw_usage, dict):
                usage.input_tokens = raw_usage.get("prompt_tokens", 0)
                usage.output_tokens = raw_usage.get("completion_tokens", 0)
                usage.total_tokens = raw_usage.get("total_tokens", 0)

        if usage.total_tokens > 0:
            usage.llm_calls = max(1, usage.total_tokens // 2000)
        usage.compute_totals()
        return usage

    @staticmethod
    def _parse_plan_json(content: str) -> list[dict[str, Any]] | None:
        """Extract and parse JSON plan from CrewAI output."""
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _capture_crew_logs(
        crew_result: Any, gathered: GatheredData,
    ) -> list[dict[str, Any]]:
        """Capture LLM inputs/outputs from CrewAI task results."""
        logs: list[dict[str, Any]] = []

        tasks_output = getattr(crew_result, "tasks_output", None)
        if tasks_output:
            for i, task_out in enumerate(tasks_output):
                phase = f"crew_task_{i}_output"
                raw = getattr(task_out, "raw", str(task_out))
                logs.append({
                    "phase": phase,
                    "output": raw if isinstance(raw, str) else str(raw),
                })

        return logs
