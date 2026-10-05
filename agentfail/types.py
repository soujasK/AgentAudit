"""
Core type definitions and data structures for AgentFail-Bench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureCategory(str, Enum):
    """Taxonomy of failure modes for autonomous multi-turn agents."""
    CYCLIC_LOOP = "CYCLIC_LOOP"
    SCHEMA_DRIFT_HALLUCINATION = "SCHEMA_DRIFT_HALLUCINATION"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    CONTEXT_EXHAUSTION = "CONTEXT_EXHAUSTION"
    TIMEOUT_CRASH = "TIMEOUT_CRASH"
    NONE = "NONE"  # Successful trajectory


class AgentStrategy(str, Enum):
    """Execution strategy profile."""
    BASELINE_NAIVE = "BASELINE_NAIVE"
    GUARDED_OPTIMIZED = "GUARDED_OPTIMIZED"


@dataclass(frozen=True)
class PricingModel:
    """Token cost model per 1M tokens in USD."""
    model_name: str
    prompt_cost_per_mtok: float
    completion_cost_per_mtok: float

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total dollar cost for given token counts."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_cost_per_mtok
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_cost_per_mtok
        return round(prompt_cost + completion_cost, 6)


# Standard 2026 Frontier Model Pricing Models
MODEL_PRICING: Dict[str, PricingModel] = {
    "gpt-4o": PricingModel(
        model_name="gpt-4o",
        prompt_cost_per_mtok=2.50,
        completion_cost_per_mtok=10.00,
    ),
    "claude-3-5-sonnet": PricingModel(
        model_name="claude-3-5-sonnet",
        prompt_cost_per_mtok=3.00,
        completion_cost_per_mtok=15.00,
    ),
    "mock-frontier": PricingModel(
        model_name="mock-frontier",
        prompt_cost_per_mtok=2.75,
        completion_cost_per_mtok=12.50,
    ),
}


@dataclass
class ToolCall:
    """Standardized tool invocation payload from agent."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCall:
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
            call_id=data.get("call_id", ""),
        )

    def signature_key(self) -> str:
        """Deterministic fingerprint for oscillation and loop detection."""
        args_serialized = json.dumps(self.arguments, sort_keys=True, default=str)
        return f"{self.name}:{args_serialized}"


@dataclass
class Observation:
    """Environment response returned after tool execution."""
    status: str  # "SUCCESS", "ERROR", "RATE_LIMITED", "POLICY_BLOCKED"
    content: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Environment step return tuple conforming to RL interface."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class Turn:
    """A single dialogue and action turn in a multi-turn trajectory."""
    turn_id: int
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    observation: Optional[Observation] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class Trajectory:
    """A full multi-turn rollout trajectory for an agent on a scenario."""
    scenario_id: str
    agent_strategy: AgentStrategy
    model: str
    prompt: str
    turns: List[Turn] = field(default_factory=list)
    success: bool = False
    reward: float = 0.0
    failure_category: FailureCategory = FailureCategory.NONE
    failure_reason: Optional[str] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    divergence_step: Optional[int] = None
    state_diff: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "agent_strategy": self.agent_strategy.value,
            "model": self.model,
            "prompt": self.prompt,
            "turns": [t.to_dict() for t in self.turns],
            "success": self.success,
            "reward": self.reward,
            "failure_category": self.failure_category.value,
            "failure_reason": self.failure_reason,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "divergence_step": self.divergence_step,
            "state_diff": self.state_diff,
        }


@dataclass
class DPOPreferencePair:
    """Dataset sample formatted for Direct Preference Optimization (DPO)."""
    id: str
    scenario: str
    prompt: str
    divergence_step: int
    failure_category: str
    chosen: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class BenchmarkMetrics:
    """Summary metrics aggregated across comparative benchmark runs."""
    strategy: str
    total_runs: int
    successful_runs: int
    success_rate_pct: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_total_tokens: float
    mean_cost_usd: float
    mean_latency_ms: float
    cyclic_loop_incidence_pct: float
    schema_drift_incidence_pct: float
    policy_violation_incidence_pct: float
    timeout_crash_incidence_pct: float
    dpo_export_yield: int = 0
