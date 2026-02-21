"""Backend registry for framework comparison."""

from __future__ import annotations

from typing import Any

from travelplanner_bench.backend import AgentBackend

_BACKENDS: dict[str, str] = {
    "opensymbolicai": "travelplanner_bench.backends.opensymbolicai_backend.OpenSymbolicAIBackend",
    "langchain": "travelplanner_bench.backends.langchain_backend.LangChainBackend",
    "crewai": "travelplanner_bench.backends.crewai_backend.CrewAIBackend",
}


def get_backend(
    framework: str,
    model: str,
    provider: str,
    **kwargs: Any,
) -> AgentBackend:
    """Instantiate a backend by framework name.

    Args:
        framework: One of "opensymbolicai", "langchain", "crewai".
        model: LLM model name/ID.
        provider: LLM provider (openai, anthropic, fireworks, etc.).
        **kwargs: Extra kwargs passed to the backend constructor.

    Returns:
        An AgentBackend instance.
    """
    if framework not in _BACKENDS:
        raise ValueError(
            f"Unknown framework {framework!r}. "
            f"Available: {', '.join(_BACKENDS)}"
        )
    module_path, class_name = _BACKENDS[framework].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(model=model, provider=provider, **kwargs)


def available_frameworks() -> list[str]:
    """Return list of registered framework names."""
    return list(_BACKENDS.keys())
