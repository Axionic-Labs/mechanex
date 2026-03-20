import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from mechanex import Mechanex
from mechanex.errors import MechanexError


class _DummyHooksContext:
    def __init__(self, model, hooks):
        self.model = model
        self.hooks = hooks or []

    def __enter__(self):
        self.model.last_hooks = self.hooks
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyLocalModel:
    class _Cfg:
        device = "cpu"

    cfg = _Cfg()

    def __init__(self):
        self.last_hooks = []
        self.calls: List[Dict[str, Any]] = []

    def hooks(self, fwd_hooks=None):
        return _DummyHooksContext(self, fwd_hooks)

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return f"{prompt} :: generated"

    def to_string(self, out):
        return [str(out)]


def _fresh_local_client() -> Mechanex:
    return Mechanex(local_model=_DummyLocalModel()).set_execution_mode("local")


class TestExecutionModes(unittest.TestCase):
    def test_default_auto_without_local_or_auth_uses_remote(self):
        mx = Mechanex()
        self.assertFalse(mx.should_use_local())

    def test_auto_with_local_and_no_auth_uses_local(self):
        mx = Mechanex(local_model=_DummyLocalModel())
        mx.api_key = None
        mx.access_token = None
        self.assertTrue(mx.should_use_local())

    def test_auto_with_local_and_api_key_prefers_remote(self):
        mx = Mechanex(local_model=_DummyLocalModel())
        mx.api_key = "key"
        self.assertFalse(mx.should_use_local())

    def test_auto_with_local_and_access_token_prefers_remote(self):
        mx = Mechanex(local_model=_DummyLocalModel())
        mx.access_token = "token"
        self.assertFalse(mx.should_use_local())

    def test_remote_mode_forces_remote(self):
        mx = Mechanex(local_model=_DummyLocalModel()).set_execution_mode("remote")
        self.assertFalse(mx.should_use_local())

    def test_local_mode_requires_local_model(self):
        mx = Mechanex().set_execution_mode("local")
        with self.assertRaises(MechanexError):
            _ = mx.should_use_local()

    def test_set_execution_mode_rejects_invalid_mode(self):
        mx = Mechanex()
        with self.assertRaises(MechanexError):
            mx.set_execution_mode("edge")


class TestInlinePolicyBuilder(unittest.TestCase):
    def test_build_inline_policy_includes_sampling_constraints_verifiers(self):
        mx = _fresh_local_client()
        policy = mx.generation._build_inline_policy(
            sampling_method="guided-generation",
            temperature=0.6,
            top_k=20,
            top_p=0.8,
            min_p=0.1,
            typical_p=0.9,
            ads_subset_size=8,
            ads_beta=0.2,
            regex_pattern="hello",
            grammar="regex:^.*$",
            json_schema={"type": "object", "required": ["a"]},
            draft_model="draft",
            ensemble_models=["m1", "m2"],
            steering_vector="vec-id",
            steering_strength=0.7,
            best_of_n=2,
        )
        self.assertEqual(policy["sampling"]["method"], "ensemble-sampling")
        self.assertTrue(policy["constraints"]["json_mode"])
        self.assertIn("json_schema", policy["constraints"])
        self.assertIn("regex_pattern", policy["constraints"])
        self.assertIn("grammar", policy["constraints"])
        self.assertIn("json_schema", policy["verifiers"]["enabled"])
        self.assertTrue(policy["steering"]["enabled"])
        self.assertEqual(policy["steering"]["vector_id"], "vec-id")
        self.assertEqual(policy["optimization"]["best_of_n"], 2)

    def test_build_inline_policy_supports_inline_vector(self):
        mx = _fresh_local_client()
        policy = mx.generation._build_inline_policy(
            sampling_method="top-k",
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            min_p=None,
            typical_p=None,
            ads_subset_size=None,
            ads_beta=None,
            regex_pattern=None,
            grammar=None,
            json_schema=None,
            draft_model=None,
            ensemble_models=None,
            steering_vector={"0": [0.1, 0.2]},
            steering_strength=0.3,
            best_of_n=1,
        )
        self.assertTrue(policy["steering"]["enabled"])
        self.assertIn("vector", policy["steering"])

    def test_build_inline_policy_enables_steering_with_strength_only(self):
        mx = _fresh_local_client()
        policy = mx.generation._build_inline_policy(
            sampling_method="top-k",
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            min_p=None,
            typical_p=None,
            ads_subset_size=None,
            ads_beta=None,
            regex_pattern=None,
            grammar=None,
            json_schema=None,
            draft_model=None,
            ensemble_models=None,
            steering_vector=None,
            steering_strength=0.1,
            best_of_n=1,
        )
        self.assertTrue(policy["steering"]["enabled"])
        self.assertEqual(policy["steering"]["strength"], 0.1)


class TestGenerationLocalBehavior(unittest.TestCase):
    def test_local_generate_top_k_uses_top_k_param(self):
        mx = _fresh_local_client()
        out = mx.generation.generate("hello", sampling_method="top-k", top_k=7, max_tokens=10)
        self.assertIn("hello", out)
        self.assertEqual(mx.local_model.calls[-1]["kwargs"]["top_k"], 7)

    def test_local_generate_top_p_uses_top_p_param(self):
        mx = _fresh_local_client()
        out = mx.generation.generate("hello", sampling_method="top-p", top_p=0.7, max_tokens=10)
        self.assertIn("hello", out)
        self.assertEqual(mx.local_model.calls[-1]["kwargs"]["top_p"], 0.7)

    def test_local_generate_greedy_sets_top_k_one(self):
        mx = _fresh_local_client()
        out = mx.generation.generate("hello", sampling_method="greedy", max_tokens=10)
        self.assertIn("hello", out)
        self.assertEqual(mx.local_model.calls[-1]["kwargs"]["top_k"], 1)

    def test_local_generate_blocks_ads(self):
        mx = _fresh_local_client()
        with self.assertRaises(MechanexError):
            mx.generation.generate("hello", sampling_method="ads")

    def test_local_generate_blocks_unknown_sampling_method(self):
        mx = _fresh_local_client()
        with self.assertRaises(MechanexError):
            mx.generation.generate("hello", sampling_method="beam-search")

    def test_local_generate_applies_steering_hooks_with_vector_id(self):
        mx = _fresh_local_client()
        mx._local_vectors["v1"] = {0: [0.1, 0.2]}
        _ = mx.generation.generate(
            "hello",
            sampling_method="top-k",
            steering_vector="v1",
            steering_strength=0.5,
            max_tokens=10,
        )
        self.assertTrue(mx.local_model.last_hooks)


class TestPolicyLocalExecution(unittest.TestCase):
    def test_policy_run_local_without_policy_raises(self):
        mx = _fresh_local_client()
        with self.assertRaises(MechanexError):
            mx.policy.run(prompt="hello")

    def test_policy_save_get_list_local_roundtrip(self):
        mx = _fresh_local_client()
        pid = mx.policy.save({"name": "p1", "sampling": {"method": "top-k"}})
        self.assertIsInstance(pid, str)
        data = mx.policy.get(pid)
        self.assertEqual(data["name"], "p1")
        listed = mx.policy.list()
        self.assertTrue(any(item["id"] == pid for item in listed))

    def test_policy_run_local_accepts_policy_id(self):
        mx = _fresh_local_client()
        pid = mx.policy.save({"name": "run-id", "sampling": {"method": "top-k"}})
        result = mx.policy.run(prompt="hello", policy_id=pid)
        self.assertIn("output", result)

    def test_policy_evaluate_local_returns_metrics(self):
        mx = _fresh_local_client()
        policy = {"name": "eval", "sampling": {"method": "top-k"}}
        result = mx.policy.evaluate(prompts=["a", "b", "c"], policy=policy)
        self.assertEqual(result["num_prompts"], 3)
        self.assertIn("success_rate", result)

    def test_policy_compare_local_returns_result_list(self):
        mx = _fresh_local_client()
        policies = [{"name": "a", "sampling": {"method": "top-k"}}, {"name": "b", "sampling": {"method": "top-p"}}]
        result = mx.policy.compare(prompt="x", policies=policies)
        self.assertEqual(len(result["results"]), 2)

    def test_policy_run_local_blocks_steering_perceptrons(self):
        mx = _fresh_local_client()
        policy = {
            "sampling": {"method": "top-k"},
            "steering": {"enabled": True, "method": "steering-perceptrons"},
        }
        with self.assertRaises(MechanexError):
            mx.policy.run(prompt="x", policy=policy)

    def test_policy_run_local_applies_steering_vector(self):
        mx = _fresh_local_client()
        mx._local_vectors["v2"] = {0: [0.1, -0.1]}
        policy = {
            "sampling": {"method": "top-k"},
            "steering": {"enabled": True, "vector_id": "v2", "strength": 0.3},
            "constraints": {},
            "verifiers": {"enabled": []},
            "optimization": {"best_of_n": 1},
        }
        result = mx.policy.run(prompt="x", policy=policy)
        self.assertIn("output", result)
        self.assertTrue(mx.local_model.last_hooks)


class TestPolicyPresets(unittest.TestCase):
    def test_strict_json_extraction_preset(self):
        schema = {"type": "object", "required": ["field"]}
        policy = Mechanex().policy.strict_json_extraction(schema)
        self.assertTrue(policy["constraints"]["json_mode"])
        self.assertIn("json_schema", policy["constraints"])

    def test_fast_tool_router_preset(self):
        policy = Mechanex().policy.fast_tool_router()
        self.assertEqual(policy["sampling"]["method"], "speculative-decoding")
        self.assertIn("tool_name", policy["constraints"]["required_fields"])

    def test_diverse_chatbot_preset(self):
        policy = Mechanex().policy.diverse_chatbot()
        self.assertEqual(policy["sampling"]["method"], "ads")
        self.assertGreaterEqual(policy["sampling"]["num_candidates"], 1)

    def test_ensemble_vote_preset(self):
        policy = Mechanex().policy.ensemble_vote(["m1", "m2"])
        self.assertEqual(policy["sampling"]["method"], "ensemble-sampling")
        self.assertEqual(policy["sampling"]["ensemble_models"], ["m1", "m2"])


class TestRoutingPaths(unittest.TestCase):
    def test_generate_remote_policy_path_uses_policies_run(self):
        mx = Mechanex().set_execution_mode("remote")
        mx.api_key = "k"
        with patch.object(mx.generation, "_post", return_value={"output": "ok"}) as post_mock:
            out = mx.generation.generate("hi", policy={"sampling": {"method": "top-k"}})
        self.assertEqual(out, "ok")
        self.assertEqual(post_mock.call_args[0][0], "/policies/run")

    def test_generate_remote_simple_path_uses_generate_sse(self):
        mx = Mechanex().set_execution_mode("remote")
        mx.api_key = "k"
        with patch.object(mx.generation, "_post_sse", return_value={"output": "ok"}) as sse_mock:
            out = mx.generation.generate("hi", sampling_method="top-k")
        self.assertEqual(out, "ok")
        self.assertEqual(sse_mock.call_args[0][0], "/generate")

    def test_generate_remote_requires_auth(self):
        mx = Mechanex().set_execution_mode("remote")
        with self.assertRaises(MechanexError):
            mx.generation.generate("hi", sampling_method="top-k")

    def test_policy_remote_requires_auth_for_save(self):
        mx = Mechanex().set_execution_mode("remote")
        with self.assertRaises(MechanexError):
            mx.policy.save({"sampling": {"method": "top-k"}})


class TestSSEParserMatrix(unittest.TestCase):
    pass


def _make_sse_parse_test(lines, expected_count, expected_status=None):
    def _test(self):
        events = Mechanex._parse_sse_events(lines)
        self.assertEqual(len(events), expected_count)
        if expected_status is not None and events:
            self.assertEqual(events[-1].get("status"), expected_status)

    return _test


_sse_cases = [
    ([b"data: {\"status\":\"processing\"}", b""], 1, "processing"),
    ([b"data:{\"status\":\"complete\"}", b""], 1, "complete"),
    ([b": ping", b"data: {\"status\":\"complete\"}", b""], 1, "complete"),
    ([b"event: x", b"data: {\"status\":\"complete\"}", b""], 1, "complete"),
    ([b"data: {\"status\":\"a\"}", b"", b"data: {\"status\":\"b\"}", b""], 2, "b"),
    ([b"data:{\"status\":\"complete\",", b"data:\"result\":{}}", b""], 1, "complete"),
    ([b"data: not-json", b""], 0, None),
    ([b"", b"", b""], 0, None),
    ([b"data: {\"status\":\"error\",\"message\":\"x\"}", b""], 1, "error"),
    ([b"data: {\"status\":\"complete\"}"], 1, "complete"),
]

for idx, case in enumerate(_sse_cases):
    setattr(
        TestSSEParserMatrix,
        f"test_sse_case_{idx:03d}",
        _make_sse_parse_test(*case),
    )


class TestConstraintMatrix(unittest.TestCase):
    pass


def _make_constraint_test(text, constraints, expected_ok):
    def _test(self):
        mx = _fresh_local_client()
        ok, _ = mx.policy._check_constraints(text, constraints)
        self.assertEqual(ok, expected_ok)

    return _test


_constraint_cases = [
    ("{\"a\":1}", {"json_mode": True}, True),
    ("not json", {"json_mode": True}, False),
    ("{\"a\":1}", {"json_schema": {"type": "object", "required": ["a"]}}, True),
    ("{\"a\":1}", {"json_schema": {"type": "object", "required": ["b"]}}, False),
    ("hello world", {"regex_pattern": "hello"}, True),
    ("hello world", {"regex_pattern": "^world"}, False),
    ("alpha beta", {"grammar": "\"alpha\" \"beta\""}, True),
    ("beta alpha", {"grammar": "\"alpha\" \"beta\""}, False),
    ("safe text", {"forbidden_terms": ["bad"]}, True),
    ("contains bad token", {"forbidden_terms": ["bad"]}, False),
]

# Add broad edge combinations.
_constraint_cases.extend(
    [
        ("{\"x\":1}", {"json_mode": True, "required_fields": ["x"]}, True),
        ("{\"x\":1}", {"json_mode": True, "required_fields": ["y"]}, False),
        ("{\"x\":\"v\"}", {"json_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}, True),
        ("{\"x\":1}", {"json_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}, False),
        ("abc123", {"regex_pattern": r"\d+"}, True),
        ("abc", {"regex_pattern": r"\d+"}, False),
        ("A B C", {"grammar": "regex:^A\\sB\\sC$"}, True),
        ("A C B", {"grammar": "regex:^A\\sB\\sC$"}, False),
        ("good", {"forbidden_terms": ["evil", "mal"]}, True),
        ("evil appears", {"forbidden_terms": ["evil", "mal"]}, False),
    ]
)

for idx, case in enumerate(_constraint_cases * 3):  # 60 tests
    setattr(
        TestConstraintMatrix,
        f"test_constraint_case_{idx:03d}",
        _make_constraint_test(*case),
    )


class TestVerifierMatrix(unittest.TestCase):
    pass


def _make_verifier_test(text, constraints, verifiers, expected_ok):
    def _test(self):
        mx = _fresh_local_client()
        ok, _, _ = mx.policy._run_verifiers(text, constraints, verifiers)
        self.assertEqual(ok, expected_ok)

    return _test


_verifier_cases = [
    ("{\"a\":1}", {"json_mode": True}, {"enabled": ["syntax"]}, True),
    ("bad", {"json_mode": True}, {"enabled": ["syntax"]}, False),
    (
        "{\"a\":1}",
        {"json_schema": {"type": "object", "required": ["a"]}},
        {"enabled": ["json_schema"]},
        True,
    ),
    (
        "{\"a\":1}",
        {"json_schema": {"type": "object", "required": ["b"]}},
        {"enabled": ["json_schema"]},
        False,
    ),
    ("hello 123", {"regex_pattern": r"\d+"}, {"enabled": ["regex"]}, True),
    ("hello", {"regex_pattern": r"\d+"}, {"enabled": ["regex"]}, True),
]

for idx, case in enumerate(_verifier_cases * 4):  # 24 tests
    setattr(
        TestVerifierMatrix,
        f"test_verifier_case_{idx:03d}",
        _make_verifier_test(*case),
    )


class TestSamplingMethodMatrix(unittest.TestCase):
    pass


def _make_sampling_matrix_test(method: str):
    def _test(self):
        mx = _fresh_local_client()
        if method in ("ads", "adaptive-determinantal-sampling"):
            with self.assertRaises(MechanexError):
                mx.generation.generate("p", sampling_method=method, max_tokens=3)
            return
        if method in ("constrained-beam-search", "speculative-decoding", "ssd", "guided-generation", "ensemble-sampling"):
            out = mx.generation.generate("p", sampling_method=method, max_tokens=3, best_of_n=1)
            self.assertIn("p", out)
            return
        out = mx.generation.generate("p", sampling_method=method, max_tokens=3)
        self.assertIn("p", out)

    return _test


_sampling_methods = [
    "top-k",
    "top-p",
    "greedy",
    "min-p",
    "typical",
    "ads",
    "adaptive-determinantal-sampling",
    "constrained-beam-search",
    "speculative-decoding",
    "ssd",
    "guided-generation",
    "ensemble-sampling",
]

for idx, method in enumerate(_sampling_methods * 2):  # 24 tests
    setattr(
        TestSamplingMethodMatrix,
        f"test_sampling_method_{idx:03d}_{method.replace('-', '_')}",
        _make_sampling_matrix_test(method),
    )


if __name__ == "__main__":
    unittest.main()
