"""
AgentFail-Bench: Deterministic Stress-Testing & Trajectory DPO Harness for Multi-Turn Frontier Agents.

A production-grade RL sandbox and verification engine for autonomous enterprise agents.
"""

from agentfail.types import (
    ToolCall,
    Observation,
    StepResult,
    Trajectory,
    Turn,
    FailureCategory,
    BenchmarkMetrics,
    DPOPreferencePair,
    AgentStrategy,
)
from agentfail.environment import EnterpriseSandbox
from agentfail.verifier import (
    DeterministicVerifier,
    FailureTaxonomyClassifier,
    StateDiffOracle,
    ASTValidator,
)
from agentfail.runner import AgentRunner, CircuitBreakerGate
from agentfail.synthesizer import DPOSynthesizer

__version__ = "1.0.0"
__all__ = [
    "EnterpriseSandbox",
    "DeterministicVerifier",
    "FailureTaxonomyClassifier",
    "StateDiffOracle",
    "ASTValidator",
    "AgentRunner",
    "CircuitBreakerGate",
    "DPOSynthesizer",
    "ToolCall",
    "Observation",
    "StepResult",
    "Trajectory",
    "Turn",
    "FailureCategory",
    "BenchmarkMetrics",
    "DPOPreferencePair",
    "AgentStrategy",
]
