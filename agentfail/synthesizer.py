"""
Module 4: RL/DPO Signal Extractor (agentfail/synthesizer.py)

Converts paired trajectories (Guarded vs. Baseline Failures) into standardized
Direct Preference Optimization (DPO) and RLHF training datasets.
Computes divergence points, token deltas, cost savings, and outputs train-ready JSONL.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentfail.types import DPOPreferencePair, FailureCategory, Trajectory, Turn


class DPOSynthesizer:
    """
    Synthesizes standardized DPO preference datasets from multi-turn agent rollouts.
    Aligns winning (chosen) trajectories with divergent/failing (rejected) trajectories.
    """

    def __init__(self, min_token_delta: int = 100):
        self.min_token_delta = min_token_delta

    def synthesize_pairs(
        self,
        baseline_trajectories: List[Trajectory],
        guarded_trajectories: List[Trajectory],
    ) -> List[DPOPreferencePair]:
        """
        Pair corresponding trajectories by scenario ID and synthesize DPO preference records.
        """
        # Index guarded trajectories by scenario_id
        guarded_map: Dict[str, Trajectory] = {
            t.scenario_id: t for t in guarded_trajectories if t.success
        }

        dpo_pairs: List[DPOPreferencePair] = []

        for b_traj in baseline_trajectories:
            # We want baseline failures paired with guarded successes
            if b_traj.success:
                continue

            g_traj = guarded_map.get(b_traj.scenario_id)
            if not g_traj:
                continue

            pair = self._create_preference_pair(g_traj, b_traj)
            if pair:
                dpo_pairs.append(pair)

        return dpo_pairs

    def _create_preference_pair(
        self,
        chosen_traj: Trajectory,
        rejected_traj: Trajectory,
    ) -> Optional[DPOPreferencePair]:
        """Construct a single DPO pair with computed divergence step and metrics."""
        chosen_tokens = chosen_traj.total_prompt_tokens + chosen_traj.total_completion_tokens
        rejected_tokens = rejected_traj.total_prompt_tokens + rejected_traj.total_completion_tokens
        token_delta = rejected_tokens - chosen_tokens
        cost_saved = round(max(0.0, rejected_traj.total_cost_usd - chosen_traj.total_cost_usd), 6)

        # Detect divergence step
        divergence_step = rejected_traj.divergence_step or self._compute_divergence_step(chosen_traj, rejected_traj)

        # Format turns into standardized chat completion dialogue
        chosen_turns = self._format_turn_sequence(chosen_traj.turns)
        rejected_turns = self._format_turn_sequence(rejected_traj.turns)

        pair_id = f"dpo_{chosen_traj.scenario_id}_{chosen_traj.model}"

        metrics = {
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "token_delta": token_delta,
            "cost_saved_usd": cost_saved,
            "chosen_reward": chosen_traj.reward,
            "rejected_reward": rejected_traj.reward,
            "latency_delta_ms": round(rejected_traj.total_latency_ms - chosen_traj.total_latency_ms, 2),
            "chosen_step_count": len(chosen_traj.turns),
            "rejected_step_count": len(rejected_traj.turns),
        }

        return DPOPreferencePair(
            id=pair_id,
            scenario=chosen_traj.scenario_id,
            prompt=chosen_traj.prompt,
            divergence_step=divergence_step,
            failure_category=rejected_traj.failure_category.value,
            chosen=chosen_turns,
            rejected=rejected_turns,
            metrics=metrics,
        )

    @staticmethod
    def _compute_divergence_step(chosen: Trajectory, rejected: Trajectory) -> int:
        """Find the 1-based index where the two trajectories first execute different actions."""
        min_len = min(len(chosen.turns), len(rejected.turns))
        for i in range(min_len):
            c_calls = chosen.turns[i].tool_calls
            r_calls = rejected.turns[i].tool_calls

            if not c_calls and not r_calls:
                continue

            c_sig = c_calls[0].signature_key() if c_calls else ""
            r_sig = r_calls[0].signature_key() if r_calls else ""

            if c_sig != r_sig:
                return i + 1

        return 1

    @staticmethod
    def _format_turn_sequence(turns: List[Turn]) -> List[Dict[str, Any]]:
        """
        Format turns into standardized message sequence for DPO training:
        [{"role": "assistant", "content": ..., "tool_calls": [...]}, {"role": "tool", "content": ...}]
        """
        sequence: List[Dict[str, Any]] = []
        for t in turns:
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": t.content,
            }
            if t.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.call_id or f"call_{t.turn_id}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in t.tool_calls
                ]
            sequence.append(assistant_msg)

            if t.observation:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": t.tool_calls[0].call_id if t.tool_calls else f"call_{t.turn_id}",
                    "name": t.tool_calls[0].name if t.tool_calls else "environment",
                    "content": t.observation.content,
                }
                sequence.append(tool_msg)

        return sequence

    def export_jsonl(self, pairs: List[DPOPreferencePair], output_path: str) -> int:
        """
        Write preference pairs to JSONL file.
        Returns count of records written.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair.to_jsonl() + "\n")

        return len(pairs)
