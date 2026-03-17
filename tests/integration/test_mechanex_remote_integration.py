import os
import unittest

import requests

from mechanex import Mechanex


class TestMechanexRemoteIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("MECHANEX_INTEGRATION_BASE_URL", "").strip()
        cls.api_key = os.getenv("MECHANEX_INTEGRATION_API_KEY", "").strip()
        cls.run_remote = os.getenv("MECHANEX_INTEGRATION_RUN_REMOTE", "").lower() in ("1", "true", "yes")

    def test_docs_endpoint_reachable(self):
        if not self.base_url:
            self.skipTest("Set MECHANEX_INTEGRATION_BASE_URL to run integration tests.")
        resp = requests.get(f"{self.base_url.rstrip('/')}/docs", timeout=20)
        self.assertLess(resp.status_code, 500)

    def test_policy_run_remote_smoke(self):
        if not self.run_remote:
            self.skipTest("Set MECHANEX_INTEGRATION_RUN_REMOTE=1 to enable remote smoke test.")
        if not self.base_url or not self.api_key:
            self.skipTest("Set MECHANEX_INTEGRATION_BASE_URL and MECHANEX_INTEGRATION_API_KEY.")

        mx = Mechanex(base_url=self.base_url).set_execution_mode("remote")
        mx.set_key(self.api_key)
        policy = {
            "task_profile": {"name": "Strict JSON extraction"},
            "objective": {"name": "maximize json validity"},
            "sampling": {"method": "guided-generation", "temperature": 0.2, "top_p": 0.9},
            "constraints": {
                "json_mode": True,
                "json_schema": {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                },
            },
            "verifiers": {"enabled": ["syntax", "json_schema"], "repair_on_failure": True},
            "optimization": {"best_of_n": 1},
        }

        result = mx.policy.run(
            prompt="Return JSON only with one field 'answer' and value 'ok'.",
            policy=policy,
            include_trace=True,
        )
        self.assertIn("output", result)
        self.assertIn("trace", result)


if __name__ == "__main__":
    unittest.main()
