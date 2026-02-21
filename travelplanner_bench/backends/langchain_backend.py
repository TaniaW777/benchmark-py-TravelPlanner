"""LangChain backend: ReAct agent for retrieval + structured output for assembly."""

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
from travelplanner_bench.tool_wrappers import make_langchain_tools

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MONKEY-PATCH: ChatOpenAI subclass for malformed tool-call arguments
#
# Some open-source models (notably gpt-oss-120b) append stray generation
# tokens like "<|call|>" to tool-call argument JSON, which causes LangChain's
# default parser to drop the tool calls entirely (they end up in
# invalid_tool_calls). This subclass overrides _generate() to strip the
# tokens and recover valid tool_calls.
#
# Only used when provider == "fireworks".  Safe to remove once the upstream
# model handles this.
# ---------------------------------------------------------------------------

def _clean_tool_call_args(raw: str) -> str:
    """Strip stray generation tokens (e.g. ``<|call|>``) from arguments JSON."""
    cleaned = re.sub(r"\s*<\|[^>]*\|>\s*", "", raw).strip()
    return cleaned


def _make_cleaned_chat_openai(**kwargs: Any) -> Any:
    """Create a ChatOpenAI subclass that cleans ``<|call|>`` from tool call args.

    Subclassing ChatOpenAI is the idiomatic LangChain approach — the result is
    a proper Runnable that works with ``create_react_agent``, ``|`` piping, etc.
    """
    from langchain_openai import ChatOpenAI

    class _CleanedChatOpenAI(ChatOpenAI):  # type: ignore[misc]
        """ChatOpenAI subclass that recovers tool calls with malformed JSON args."""

        def _generate(
            self,
            messages: list,
            stop: list[str] | None = None,
            run_manager: Any | None = None,
            **kw: Any,
        ) -> Any:
            result = super()._generate(
                messages, stop=stop, run_manager=run_manager, **kw
            )
            for gen in result.generations:
                self._recover_tool_calls(gen.message)
            return result

        @staticmethod
        def _recover_tool_calls(msg: Any) -> None:
            """If LangChain failed to parse tool_calls, recover from raw data."""
            if getattr(msg, "tool_calls", None):
                return
            raw_tcs = (getattr(msg, "additional_kwargs", None) or {}).get(
                "tool_calls"
            )
            if not raw_tcs:
                return
            tool_calls = []
            for tc in raw_tcs:
                func = tc.get("function", {})
                name = func.get("name") or tc.get("name")
                raw_args = func.get("arguments", "{}")
                try:
                    args = json.loads(_clean_tool_call_args(raw_args))
                except json.JSONDecodeError:
                    continue
                tool_calls.append({
                    "name": name,
                    "args": args,
                    "id": tc.get("id", ""),
                    "type": "tool_call",
                })
            if tool_calls:
                msg.tool_calls = tool_calls
                if hasattr(msg, "invalid_tool_calls"):
                    msg.invalid_tool_calls = []

    return _CleanedChatOpenAI(**kwargs)


class LangChainBackend:
    """LangChain ReAct agent for TravelPlanner benchmark."""

    def __init__(
        self,
        model: str,
        provider: str,
        max_iterations: int = 25,
        **kwargs: object,
    ) -> None:
        self._model = model
        self._provider = provider
        self._max_iterations = max_iterations

    @property
    def framework_name(self) -> str:
        return "langchain"

    def _make_llm(self) -> Any:
        """Create a LangChain chat model based on provider."""
        if self._provider in ("openai", "fireworks"):
            model = self._model
            if self._provider == "fireworks":
                import os
                if not model.startswith("accounts/"):
                    model = f"accounts/fireworks/models/{model}"
                base_url = "https://api.fireworks.ai/inference/v1"
                api_key = os.environ.get("FIREWORKS_API_KEY", "")
                return _make_cleaned_chat_openai(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=0,
                )
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0)
        elif self._provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=self._model, temperature=0)
        else:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=self._model, temperature=0)

    def solve(self, task: TravelPlannerTask) -> BackendResult:
        from langgraph.prebuilt import create_react_agent

        token_usage = TokenUsage()
        raw_logs: list[dict[str, Any]] = []
        start = time.perf_counter()

        try:
            db = ReferenceDatabase(task.reference_information)
            gathered = GatheredData()
            tools = make_langchain_tools(db, gathered)
            llm = self._make_llm()

            # --- Phase 1: Retrieval via ReAct agent ---
            retrieval_start = time.perf_counter()
            retrieval_prompt = self._build_retrieval_prompt(task, db)
            raw_logs.append({"phase": "retrieval_prompt", "description": retrieval_prompt})

            kickoff_msg, num_tool_calls = self._build_retrieval_kickoff(task, db)
            agent = create_react_agent(llm, tools)
            # Each tool call needs ~3 steps (AI msg + tool msg + next AI).
            # Ensure enough headroom for all required calls.
            recursion_limit = max(self._max_iterations * 2, num_tool_calls * 3 + 10)
            config = {"recursion_limit": recursion_limit}

            result = agent.invoke(
                {"messages": [
                    ("system", retrieval_prompt),
                    ("user", kickoff_msg),
                ]},
                config,
            )

            # Extract token usage and log messages
            messages = result.get("messages", [])
            retrieval_tokens = self._extract_tokens_from_messages(messages)
            token_usage.retrieval_input_tokens = retrieval_tokens["input"]
            token_usage.retrieval_output_tokens = retrieval_tokens["output"]
            token_usage.retrieval_llm_calls = retrieval_tokens["calls"]
            token_usage.retrieval_time_seconds = time.perf_counter() - retrieval_start

            raw_logs.append({
                "phase": "retrieval_messages",
                "output": self._format_messages(messages),
            })

            # Fill any gaps left by the ReAct agent (it may stop early)
            gaps = self._fill_retrieval_gaps(db, gathered, task)
            if gaps:
                raw_logs.append({"phase": "gap_fill", "filled": gaps})

            # Log what tools actually gathered
            raw_logs.append({
                "phase": "tool_calls_summary",
                "gathered_data": self._gathered_summary(gathered),
            })

            # --- Phase 2: Plan assembly from gathered data ---
            assembly_start = time.perf_counter()
            assembly_prompt = self._build_assembly_prompt(gathered, task, db)
            raw_logs.append({"phase": "assembly_prompt", "description": assembly_prompt})

            plan = self._assemble_plan(llm, gathered, task, db)
            assembly_time = time.perf_counter() - assembly_start

            # Rough estimate: 4 chars per token
            est_input = len(assembly_prompt) // 4
            est_output = len(json.dumps(plan, default=str)) // 4 if plan else 0
            token_usage.assembly_input_tokens = est_input
            token_usage.assembly_output_tokens = est_output
            token_usage.assembly_llm_calls = 1
            token_usage.assembly_time_seconds = assembly_time

            # Aggregate totals
            token_usage.input_tokens = (
                token_usage.retrieval_input_tokens + token_usage.assembly_input_tokens
            )
            token_usage.output_tokens = (
                token_usage.retrieval_output_tokens + token_usage.assembly_output_tokens
            )
            token_usage.llm_calls = (
                token_usage.retrieval_llm_calls + token_usage.assembly_llm_calls
            )
            token_usage.time_seconds = (
                token_usage.retrieval_time_seconds + token_usage.assembly_time_seconds
            )
            token_usage.compute_totals()

            elapsed = time.perf_counter() - start
            iterations = token_usage.retrieval_llm_calls

            return BackendResult(
                framework="langchain",
                plan=plan,
                token_usage=token_usage,
                iterations=iterations,
                wall_time_seconds=elapsed,
                raw_logs=raw_logs,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            log.exception("LangChain solve failed: %s", e)
            return BackendResult(
                framework="langchain",
                wall_time_seconds=elapsed,
                error=str(e),
                raw_logs=raw_logs,
            )

    def _build_retrieval_prompt(
        self, task: TravelPlannerTask, db: ReferenceDatabase
    ) -> str:
        """Build system prompt for retrieval agent."""
        return (
            "You are a travel data researcher. You have access to search tools "
            "that query a reference database. You MUST call EVERY tool listed in "
            "the user's instructions. Do NOT respond with text until you have "
            "completed ALL tool calls. Do NOT make up any data — only use data "
            "returned by the tools."
        )

    def _build_retrieval_kickoff(
        self, task: TravelPlannerTask, db: ReferenceDatabase
    ) -> tuple[str, int]:
        """Build the user message and return (message, num_tool_calls)."""
        city_names = search_cities(db, task.dest)

        # Build explicit tool call instructions
        calls: list[str] = []
        call_num = 1

        # Flights
        for orig, dest, date in db.flights:
            calls.append(
                f"{call_num}. search_flights_tool(origin=\"{orig.title()}\", "
                f"destination=\"{dest.title()}\", date=\"{date}\")"
            )
            call_num += 1

        # Distances
        for orig, dest, mode in db.distances:
            calls.append(
                f"{call_num}. get_distance_tool(origin=\"{orig.title()}\", "
                f"destination=\"{dest.title()}\", mode=\"{mode}\")"
            )
            call_num += 1

        # Per-city searches
        cities_to_search = city_names if city_names else [task.dest]
        for city in cities_to_search:
            c = city.title()
            calls.append(f"{call_num}. search_restaurants_tool(city=\"{c}\")")
            call_num += 1
            calls.append(f"{call_num}. search_accommodations_tool(city=\"{c}\")")
            call_num += 1
            calls.append(f"{call_num}. search_attractions_tool(city=\"{c}\")")
            call_num += 1

        parts = [
            f"Gather all travel data for this trip: {task.query}",
            f"Origin: {task.org}, Destination: {task.dest}, Dates: {task.date}",
            "",
            f"Make these {call_num - 1} tool calls (call ALL of them, do not skip any):",
            "",
        ]
        parts.extend(calls)
        parts.extend([
            "",
            "Call the tools NOW. Start with the first one.",
        ])
        return "\n".join(parts), call_num - 1

    @staticmethod
    def _fill_retrieval_gaps(
        db: ReferenceDatabase,
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> list[str]:
        """Programmatically fill any data the ReAct agent failed to gather.

        The gpt-oss-120b model frequently stops calling tools before completing
        all required calls. This ensures the assembly prompt always has complete
        data by falling back to direct tool invocations for any gaps.
        """
        filled: list[str] = []
        city_names = search_cities(db, task.dest)
        cities = city_names if city_names else [task.dest]

        # Fill missing per-city data
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

        # Fill missing flights
        for orig, dest, date in db.flights:
            key = f"{orig.title()}->{dest.title()} on {date}"
            if key not in gathered.flights:
                results = search_flights(db, orig, dest, date)
                if results:
                    gathered.flights[key] = results
                    filled.append(f"flights:{key}")

        # Fill missing distances
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
    def _recommend_transport_mode(
        gathered: GatheredData, db: ReferenceDatabase
    ) -> str:
        """Determine whether to recommend flights or self-driving.

        The non_conflicting_transport constraint requires that ALL inter-city
        legs use the same transport mode. If any required flight route is
        missing, recommend self-driving for everything.
        """
        # Check if all flight routes have results
        required_routes = set(db.flights.keys())
        if not required_routes:
            return "self-driving"

        for orig, dest, date in required_routes:
            key = f"{orig.title()}->{dest.title()} on {date}"
            flights = gathered.flights.get(key, [])
            if not flights:
                return "self-driving"
        return "flight"

    def _build_assembly_prompt(
        self,
        gathered: GatheredData,
        task: TravelPlannerTask,
        db: ReferenceDatabase | None = None,
    ) -> str:
        """Build prompt for plan assembly using ONLY gathered data."""
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
            constraints.append(f"Cuisine: {', '.join(task.local_constraint.cuisine)}")
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
        transport_mode = (
            self._recommend_transport_mode(gathered, db) if db else "flight"
        )
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

    def _assemble_plan(
        self,
        llm: Any,
        gathered: GatheredData,
        task: TravelPlannerTask,
        db: ReferenceDatabase | None = None,
    ) -> list[dict[str, Any]] | None:
        """Use LLM to generate a plan from gathered data."""
        prompt = self._build_assembly_prompt(gathered, task, db)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)

                plan = self._parse_plan_json(content)
                if plan:
                    return plan

                log.warning(
                    "LangChain assembly attempt %d: failed to parse plan",
                    attempt + 1,
                )
            except Exception as e:
                log.warning(
                    "LangChain assembly attempt %d failed: %s", attempt + 1, e
                )

        return None

    @staticmethod
    def _parse_plan_json(content: str) -> list[dict[str, Any]] | None:
        """Extract and parse JSON plan from LLM response."""
        # Try direct parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try finding JSON array in content
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
    def _extract_tokens_from_messages(messages: list) -> dict[str, int]:
        """Extract token counts from LangGraph message history."""
        total_input = 0
        total_output = 0
        llm_calls = 0

        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                llm_calls += 1
            # Also check response_metadata for OpenAI-style usage
            resp_meta = getattr(msg, "response_metadata", None)
            if resp_meta and "token_usage" in resp_meta and not usage:
                tu = resp_meta["token_usage"]
                total_input += tu.get("prompt_tokens", 0)
                total_output += tu.get("completion_tokens", 0)
                llm_calls += 1

        return {"input": total_input, "output": total_output, "calls": llm_calls}

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format LangGraph messages into a readable string for logging."""
        parts: list[str] = []
        for msg in messages:
            msg_type = type(msg).__name__
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = json.dumps(content, default=str)
            tool_calls = getattr(msg, "tool_calls", None)
            text = f"[{msg_type}] {content}"
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    text += f"\n  -> tool_call: {name}({json.dumps(args, default=str)})"
            parts.append(text)
        return "\n".join(parts)

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
