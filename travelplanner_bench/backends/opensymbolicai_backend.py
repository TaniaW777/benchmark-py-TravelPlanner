"""OpenSymbolicAI backend: wraps the existing TravelPlannerAgent."""

from __future__ import annotations

import time

from travelplanner_bench.backend import BackendResult, TokenUsage
from travelplanner_bench.models import TravelPlannerTask
from travelplanner_bench.token_tracking import extract_opensymbolicai_tokens


class OpenSymbolicAIBackend:
    """Wraps the existing TravelPlannerAgent to conform to AgentBackend protocol."""

    def __init__(
        self,
        model: str,
        provider: str,
        max_iterations: int = 10,
        **kwargs: object,
    ) -> None:
        from opensymbolicai.llm import LLMConfig, Provider

        provider_map = {
            "ollama": Provider.OLLAMA,
            "openai": Provider.OPENAI,
            "anthropic": Provider.ANTHROPIC,
            "fireworks": Provider.FIREWORKS,
            "groq": Provider.GROQ,
        }
        if provider == "fireworks" and not model.startswith("accounts/"):
            model = f"accounts/fireworks/models/{model}"
        self._llm_config = LLMConfig(provider=provider_map[provider], model=model)
        self._max_iterations = max_iterations
        self._model = model

    @property
    def framework_name(self) -> str:
        return "opensymbolicai"

    def solve(self, task: TravelPlannerTask) -> BackendResult:
        from travelplanner_bench.agent import TravelPlannerAgent

        agent = TravelPlannerAgent(
            llm=self._llm_config, max_iterations=self._max_iterations
        )

        start = time.perf_counter()
        try:
            plan, iterations, iteration_logs = agent.solve(task)
            elapsed = time.perf_counter() - start

            token_usage = extract_opensymbolicai_tokens(iteration_logs)

            return BackendResult(
                framework="opensymbolicai",
                plan=plan,
                token_usage=token_usage,
                iterations=iterations,
                wall_time_seconds=elapsed,
                raw_logs=[il.model_dump() for il in iteration_logs],
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            return BackendResult(
                framework="opensymbolicai",
                wall_time_seconds=elapsed,
                error=str(e),
            )
