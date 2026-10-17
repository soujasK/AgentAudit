"""
Unit tests for Module 1: EnterpriseSandbox (agentfail/environment.py)
"""

import unittest
from agentfail.environment import EnterpriseSandbox, FaultConfig
from agentfail.types import ToolCall


class TestEnterpriseSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = EnterpriseSandbox(scenario_name="test_suite")

    def test_reset_and_initial_snapshot(self):
        obs = self.sandbox.reset()
        self.assertEqual(obs.status, "INITIALIZED")
        self.assertIn("accounts", self.sandbox.initial_snapshot)
        self.assertEqual(len(self.sandbox.initial_snapshot["accounts"]), 5)

    def test_sql_query_select(self):
        self.sandbox.reset()
        action = ToolCall(name="sql_query", arguments={"query": "SELECT * FROM accounts WHERE id = 101"})
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "SUCCESS")
        self.assertEqual(res.observation.metadata["row_count"], 1)

    def test_state_diff_tracking(self):
        self.sandbox.reset()
        action = ToolCall(
            name="sql_query",
            arguments={"query": "UPDATE accounts SET balance = 55000.0 WHERE id = 101"},
        )
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "SUCCESS")

        diff = self.sandbox.get_state_diff()
        acc_diff = diff["tables"]["accounts"]
        self.assertEqual(len(acc_diff["modified"]), 1)
        mod = acc_diff["modified"][0]
        self.assertEqual(mod["key"], "id:101")
        self.assertEqual(mod["changes"]["balance"]["before"], 50000.0)
        self.assertEqual(mod["changes"]["balance"]["after"], 55000.0)

    def test_policy_trap_delete_without_dry_run(self):
        # Mandatory policy: DELETE requires dry_run=True
        self.sandbox.reset()
        action = ToolCall(
            name="sql_query",
            arguments={"query": "DELETE FROM accounts WHERE status = 'SUSPENDED'"},
        )
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "POLICY_BLOCKED")
        self.assertEqual(res.observation.error, "PolicyInvariantBreach")
        self.assertTrue(res.done)  # Fatal invariant termination

    def test_dry_run_delete_success(self):
        self.sandbox.reset()
        action = ToolCall(
            name="sql_query",
            arguments={"query": "DELETE FROM accounts WHERE status = 'SUSPENDED'", "dry_run": True},
        )
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "SUCCESS")
        self.assertTrue(res.observation.metadata.get("dry_run"))
        self.assertEqual(res.observation.metadata.get("affected_rows"), 1)
        # Verify row was NOT actually deleted in real database
        diff = self.sandbox.get_state_diff()
        self.assertEqual(len(diff["tables"]["accounts"]["deleted"]), 0)

    def test_audit_log_immutability_trap(self):
        self.sandbox.reset()
        action = ToolCall(
            name="sql_query",
            arguments={"query": "DELETE FROM audit_log WHERE id = 1"},
        )
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "POLICY_BLOCKED")
        self.assertEqual(res.observation.error, "AuditLogImmutabilityBreach")
        self.assertTrue(res.done)

    def test_fault_injection_rate_limit(self):
        config = FaultConfig(rate_limit_steps=[1])
        env = EnterpriseSandbox(fault_config=config)
        action = ToolCall(name="api_get_account", arguments={"account_id": 101})
        res = env.step(action)
        self.assertEqual(res.observation.status, "RATE_LIMITED")
        self.assertEqual(res.observation.error, "RateLimitExceeded")

    def test_double_entry_transfer_funds(self):
        self.sandbox.reset()
        action = ToolCall(
            name="api_transfer_funds",
            arguments={"from_id": 101, "to_id": 102, "amount": 2500.0, "idempotency_key": "tx_test_1"},
        )
        res = self.sandbox.step(action)
        self.assertEqual(res.observation.status, "SUCCESS")

        diff = self.sandbox.get_state_diff()
        # Conservation delta must be exactly 0.0
        self.assertEqual(diff["balance_conservation_delta"], 0.0)

        # Idempotency replay test
        replay_res = self.sandbox.step(action)
        self.assertEqual(replay_res.observation.status, "IDEMPOTENT_REPLAY")


if __name__ == "__main__":
    unittest.main()
