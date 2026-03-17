import unittest
from unittest.mock import patch

import requests

from mechanex import Mechanex
from mechanex.errors import MechanexError


class _FakeSSEResponse:
    def __init__(self, status_code=200, lines=None, json_payload=None):
        self.status_code = status_code
        self._lines = lines or []
        self._json_payload = json_payload or {}
        self.text = ""
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        return self._json_payload

    def close(self):
        self.closed = True


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

    def hooks(self, fwd_hooks=None):
        return _DummyHooksContext(self, fwd_hooks)

    def generate(self, prompt, **kwargs):
        return f"{prompt} :: local"

    def to_string(self, out):
        return [str(out)]


class TestSSEParsing(unittest.TestCase):
    def test_parse_sse_events_with_standard_and_multiline_data(self):
        lines = [
            b": keep-alive",
            b"event: status",
            b"data: {\"status\":\"processing\"}",
            b"",
            b"data:{\"status\":\"complete\",",
            b"data:\"result\":{\"output\":\"ok\"}}",
            b"",
        ]
        events = Mechanex._parse_sse_events(lines)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["status"], "processing")
        self.assertEqual(events[1]["status"], "complete")
        self.assertEqual(events[1]["result"]["output"], "ok")

    def test_post_sse_returns_complete_result_payload(self):
        mx = Mechanex()
        mx.api_key = "test-key"
        resp = _FakeSSEResponse(
            lines=[
                b"data: {\"status\":\"processing\"}",
                b"",
                b"data: {\"status\":\"complete\",\"result\":{\"output\":\"done\"}}",
                b"",
            ]
        )
        with patch("mechanex.client.requests.post", return_value=resp):
            result = mx._post_sse("/generate", {"prompt": "hello"})
        self.assertEqual(result["output"], "done")
        self.assertTrue(resp.closed)


class TestLocalPolicyRuntime(unittest.TestCase):
    def test_local_policy_applies_steering_vectors(self):
        local_model = _DummyLocalModel()
        mx = Mechanex(local_model=local_model).set_execution_mode("local")
        mx._local_vectors["vec-1"] = {0: [0.1, 0.2, 0.3]}

        policy = {
            "name": "steered-local",
            "sampling": {"method": "top-k", "top_k": 20, "temperature": 0.7},
            "steering": {"enabled": True, "vector_id": "vec-1", "strength": 0.5, "method": "caa"},
            "constraints": {},
            "verifiers": {"enabled": []},
            "optimization": {"best_of_n": 1},
        }

        result = mx.policy.run(prompt="hello", policy=policy, include_trace=True)
        self.assertIn("hello", result["output"])
        self.assertTrue(local_model.last_hooks)

    def test_local_policy_rejects_ads(self):
        mx = Mechanex(local_model=_DummyLocalModel()).set_execution_mode("local")
        policy = {"sampling": {"method": "ads"}}
        with self.assertRaises(MechanexError):
            mx.policy.run(prompt="hello", policy=policy)

    def test_local_policy_rejects_steering_perceptrons(self):
        mx = Mechanex(local_model=_DummyLocalModel()).set_execution_mode("local")
        policy = {
            "sampling": {"method": "top-k"},
            "steering": {"method": "steering-perceptrons", "enabled": True},
        }
        with self.assertRaises(MechanexError):
            mx.policy.run(prompt="hello", policy=policy)


if __name__ == "__main__":
    unittest.main()
