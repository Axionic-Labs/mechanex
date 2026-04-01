"""API Contract Validation

Validates that the SDK's endpoint paths and request/response field
assumptions stay in sync with the backend's OpenAPI spec. If the backend
renames an endpoint or removes a field the SDK depends on, these tests fail.

The spec fixture is updated via: gh api repos/Axionic-Labs/axionic-mvp-backend/contents/openapi.json
"""
import json
import os
import unittest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SPEC_PATH = FIXTURES_DIR / "backend-openapi.json"


def _load_spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


class TestOpenAPISpecFixture(unittest.TestCase):
    """Verify the spec fixture is present and parseable."""

    def test_spec_exists(self):
        self.assertTrue(SPEC_PATH.exists(), f"Missing {SPEC_PATH}")

    def test_spec_has_paths(self):
        spec = _load_spec()
        self.assertIn("paths", spec)
        self.assertGreater(len(spec["paths"]), 0)

    def test_spec_has_schemas(self):
        spec = _load_spec()
        schemas = spec.get("components", {}).get("schemas", {})
        self.assertGreater(len(schemas), 0)


class TestEndpointPaths(unittest.TestCase):
    """Verify that every endpoint the SDK calls exists in the backend spec."""

    @classmethod
    def setUpClass(cls):
        cls.spec = _load_spec()
        cls.paths = cls.spec["paths"]

    # --- Auth & Account ---

    def test_auth_signup(self):
        self.assertIn("/auth/signup", self.paths)
        self.assertIn("post", self.paths["/auth/signup"])

    def test_auth_login(self):
        self.assertIn("/auth/login", self.paths)
        self.assertIn("post", self.paths["/auth/login"])

    def test_auth_refresh(self):
        self.assertIn("/auth/refresh", self.paths)
        self.assertIn("post", self.paths["/auth/refresh"])

    def test_auth_whoami(self):
        self.assertIn("/auth/whoami", self.paths)
        self.assertIn("get", self.paths["/auth/whoami"])

    def test_auth_api_keys_list(self):
        self.assertIn("/auth/api-keys", self.paths)
        self.assertIn("get", self.paths["/auth/api-keys"])

    def test_auth_api_keys_create(self):
        self.assertIn("/auth/api-keys", self.paths)
        self.assertIn("post", self.paths["/auth/api-keys"])

    def test_auth_api_keys_balance(self):
        self.assertIn("/auth/api-keys/balance", self.paths)
        self.assertIn("get", self.paths["/auth/api-keys/balance"])

    def test_auth_change_password(self):
        self.assertIn("/auth/change-password", self.paths)
        self.assertIn("post", self.paths["/auth/change-password"])

    def test_auth_delete_account(self):
        self.assertIn("/auth/delete-account", self.paths)
        self.assertIn("delete", self.paths["/auth/delete-account"])

    # --- Models & Graph ---

    def test_models_list(self):
        self.assertIn("/models", self.paths)
        self.assertIn("get", self.paths["/models"])

    def test_graph(self):
        self.assertIn("/graph", self.paths)
        self.assertIn("get", self.paths["/graph"])

    # --- Generation & Steering ---

    def test_generate(self):
        self.assertIn("/generate", self.paths)
        self.assertIn("post", self.paths["/generate"])

    def test_steering_generate(self):
        self.assertIn("/steering/generate", self.paths)
        self.assertIn("post", self.paths["/steering/generate"])

    def test_steering_run(self):
        self.assertIn("/steering/run", self.paths)
        self.assertIn("post", self.paths["/steering/run"])

    def test_steering_generate_pairs(self):
        self.assertIn("/steering/generate-pairs", self.paths)
        self.assertIn("post", self.paths["/steering/generate-pairs"])

    def test_steering_evaluate(self):
        self.assertIn("/steering/evaluate", self.paths)
        self.assertIn("post", self.paths["/steering/evaluate"])

    # --- Behaviors / SAE ---

    def test_behaviors_create(self):
        self.assertIn("/behaviors/create", self.paths)
        self.assertIn("post", self.paths["/behaviors/create"])

    def test_behaviors_list(self):
        self.assertIn("/behaviors", self.paths)
        self.assertIn("get", self.paths["/behaviors"])

    def test_sae_generate(self):
        self.assertIn("/sae/generate", self.paths)
        self.assertIn("post", self.paths["/sae/generate"])

    # --- Payments ---

    def test_payments_subscriptions(self):
        self.assertIn("/payments/subscriptions", self.paths)
        self.assertIn("get", self.paths["/payments/subscriptions"])

    def test_payments_checkout(self):
        self.assertIn("/payments/checkout", self.paths)
        self.assertIn("post", self.paths["/payments/checkout"])

    # --- Policies ---

    def test_policies_save(self):
        self.assertIn("/policies/save", self.paths)
        self.assertIn("post", self.paths["/policies/save"])

    def test_policies_list(self):
        self.assertIn("/policies", self.paths)
        self.assertIn("get", self.paths["/policies"])

    def test_policies_run(self):
        self.assertIn("/policies/run", self.paths)
        self.assertIn("post", self.paths["/policies/run"])

    # --- OpenAI-compatible ---

    def test_openai_chat_completions(self):
        self.assertIn("/v1/chat/completions", self.paths)
        self.assertIn("post", self.paths["/v1/chat/completions"])

    def test_openai_completions(self):
        self.assertIn("/v1/completions", self.paths)
        self.assertIn("post", self.paths["/v1/completions"])


class TestResponseSchemas(unittest.TestCase):
    """Verify that backend response schemas the SDK depends on still exist
    and have the fields the SDK reads."""

    @classmethod
    def setUpClass(cls):
        cls.spec = _load_spec()
        cls.schemas = cls.spec.get("components", {}).get("schemas", {})

    def _assert_schema_has_fields(self, schema_name, expected_fields):
        self.assertIn(schema_name, self.schemas,
                      f"Schema '{schema_name}' missing from backend spec")
        props = self.schemas[schema_name].get("properties", {})
        for field in expected_fields:
            self.assertIn(field, props,
                          f"Field '{field}' missing from schema '{schema_name}'")

    # SDK reads these fields from API responses

    def test_auth_user_schema(self):
        self._assert_schema_has_fields("AuthUser", ["id", "email"])

    def test_auth_session_schema(self):
        self._assert_schema_has_fields("AuthSession",
                                       ["access_token", "refresh_token"])

    def test_api_key_item_schema(self):
        self._assert_schema_has_fields("ApiKeyItem", ["id", "key", "name"])

    def test_hosted_model_schema(self):
        self._assert_schema_has_fields("HostedModel",
                                       ["id", "name", "base_url", "is_active"])

    def test_model_graph_response_schema(self):
        self._assert_schema_has_fields("ModelGraphResponse", ["graph"])

    def test_generate_pairs_response_schema(self):
        self._assert_schema_has_fields("GeneratePairsResponse",
                                       ["pairs", "total_pairs"])

    def test_evaluate_response_schema(self):
        self._assert_schema_has_fields("EvaluateResponse",
                                       ["cosine_metrics", "judge_evaluation"])

    def test_behavior_response_schema(self):
        self._assert_schema_has_fields("BehaviorResponse",
                                       ["id", "model_id"])

    def test_sampling_response_schema(self):
        self._assert_schema_has_fields("SamplingResponse",
                                       ["output", "method_used"])

    def test_policy_run_response_schema(self):
        self._assert_schema_has_fields("PolicyRunResponse",
                                       ["output", "accepted"])


if __name__ == "__main__":
    unittest.main()
