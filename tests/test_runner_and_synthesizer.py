"""
Unit tests for Runner, CircuitBreakerGate, and DPOSynthesizer (agentfail/runner.py, agentfail/synthesizer.py).
"""

import os
import tempfile
import unittest
from agentfail.runner import AgentRunner, CircuitBreakerGate
from agentfail.scenarios import get_default_scenarios
from agentfail.synthesizer import DPOSynthesizer
from agentfail.types import AgentStrategy, FailureCategory, Observation, ToolCall


class TestRunnerAndSynthesizer(unittest.TestCase):
    def test_circuit_breaker_consecutive_loop(self):
        cb = CircuitBreakerGate(max_identical_actions=3)
        call = ToolCall(name="sql_query", arguments={"query": "SELECT 1"})
        calls = [call, call, call]
        obs = Observation(status="SUCCESS", content="ok")

        should_abort, cat, reason = cb.check_turn(calls, obs, cumulative_tokens=1000)
        self.assertTrue(should_abort)
        self.assertEqual(cat, FailureCategory.CYCLIC_LOOP)

    def test_circuit_breaker_token_ceiling(self):
        cb = CircuitBreakerGate(max_token_ceiling=5000)
        should_abort, cat, reason = cb.check_turn([], None, cumulative_tokens=5500)
        self.assertTrue(should_abort)
        self.assertEqual(cat, FailureCategory.CONTEXT_EXHAUSTION)

    def test_comparative_benchmark_and_synthesis(self):
        runner = AgentRunner(model="gpt-4o")
        scenarios = get_default_scenarios()[:2]  # Test first 2 scenarios
        b_trajs, g_trajs, b_metrics, g_metrics = runner.run_comparative_benchmark(scenarios)

        self.assertEqual(len(b_trajs), 2)
        self.assertEqual(len(g_trajs), 2)
        self.assertTrue(all(t.success for t in g_trajs))
        self.assertTrue(all(not t.success for t in b_trajs))

        # Test DPO synthesis
        synthesizer = DPOSynthesizer()
        pairs = synthesizer.synthesize_pairs(b_trajs, g_trajs)
        self.assertEqual(len(pairs), 2)

        p = pairs[0]
        self.assertIn("chosen", p.to_dict())
        self.assertIn("rejected", p.to_dict())
        self.assertIn("token_delta", p.metrics)
        self.assertGreater(len(p.chosen), 0)
        self.assertGreater(len(p.rejected), 0)

        # Test export
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            count = synthesizer.export_jsonl(pairs, temp_path)
            self.assertEqual(count, 2)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
