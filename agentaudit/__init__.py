"""
AgentAudit: Deterministic Stress-Testing & Trajectory DPO Harness for Multi-Turn Frontier Agents.

A production-grade RL sandbox and verification engine for autonomous enterprise agents.
"""

from agentaudit.types import (
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
from agentaudit.environment import EnterpriseSandbox
from agentaudit.verifier import (
    DeterministicVerifier,
    FailureTaxonomyClassifier,
    StateDiffOracle,
    ASTValidator,
)
from agentaudit.runner import AgentRunner, CircuitBreakerGate
from agentaudit.synthesizer import DPOSynthesizer

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
