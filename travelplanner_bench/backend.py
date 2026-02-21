"""Agent backend protocol and shared result models for framework comparison."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Unified token tracking across all frameworks."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    time_seconds: float = 0.0
    # Phase breakdown
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    retrieval_llm_calls: int = 0
    retrieval_time_seconds: float = 0.0
    assembly_input_tokens: int = 0
    assembly_output_tokens: int = 0
    assembly_llm_calls: int = 0
    assembly_time_seconds: float = 0.0

    def compute_totals(self) -> None:
        """Recompute total_tokens from input/output."""
        self.total_tokens = self.input_tokens + self.output_tokens


class BackendResult(BaseModel):
    """Unified result from any agent backend."""

    framework: str = ""
    plan: list[dict[str, Any]] | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    raw_logs: list[dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class AgentBackend(Protocol):
    """Protocol that all framework backends must implement."""

    @property
    def framework_name(self) -> str: ...

    def solve(self, task: Any) -> BackendResult: ...
