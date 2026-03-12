from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .base import _BaseModule
from .errors import MechanexError

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


class PolicyModule(_BaseModule):
    """
    Runtime policy API for inference-time control.
    Supports both remote and local execution.
    """

    def save(self, policy: Dict[str, Any]) -> str:
        if self._use_local():
            pid = policy.get("id") or str(uuid.uuid4())
            payload = dict(policy)
            payload["id"] = pid
            self._client._local_policies[pid] = payload
            return pid

        self._require_remote_auth()
        resp = self._post("/policies/save", {"policy": policy})
        policy_id = resp.get("policy_id")
        if not policy_id:
            raise MechanexError(f"Failed to save policy: {resp}")
        return policy_id

    def list(self) -> List[Dict[str, Any]]:
        if self._use_local():
            return list(self._client._local_policies.values())

        self._require_remote_auth()
        resp = self._get("/policies")
        return resp if isinstance(resp, list) else resp.get("policies", [])

    def get(self, policy_id: str) -> Dict[str, Any]:
        if self._use_local():
            item = self._client._local_policies.get(policy_id)
            if item is None:
                raise MechanexError(f"Policy '{policy_id}' not found in local session.")
            return item

        self._require_remote_auth()
        return self._get(f"/policies/{policy_id}")

    def run(
        self,
        prompt: str,
        policy: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        model: str = "default",
        max_new_tokens: int = 128,
        include_trace: bool = True,
    ) -> Dict[str, Any]:
        if policy is None and policy_id is None:
            raise MechanexError("Provide either `policy` or `policy_id`.")

        if self._use_local():
            resolved = self._resolve_local_policy(policy=policy, policy_id=policy_id)
            return self._run_local_policy(
                prompt=prompt,
                policy=resolved,
                max_new_tokens=max_new_tokens,
                include_trace=include_trace,
            )

        self._require_remote_auth()
        payload = {
            "prompt": prompt,
            "policy_id": policy_id,
            "policy": policy,
            "model": model,
            "max_new_tokens": max_new_tokens,
            "include_trace": include_trace,
        }
        return self._post("/policies/run", payload)

    def compare(
        self,
        prompt: str,
        policies: List[Dict[str, Any]],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        if self._use_local():
            results = [
                self._run_local_policy(
                    prompt=prompt,
                    policy=p,
                    max_new_tokens=max_new_tokens,
                    include_trace=True,
                )
                for p in policies
            ]
            return {"results": results}

        self._require_remote_auth()
        payload = {
            "prompt": prompt,
            "policies": policies,
            "max_new_tokens": max_new_tokens,
        }
        return self._post("/policies/compare", payload)

    def evaluate(
        self,
        prompts: List[str],
        policy: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        if policy is None and policy_id is None:
            raise MechanexError("Provide either `policy` or `policy_id`.")

        if self._use_local():
            resolved = self._resolve_local_policy(policy=policy, policy_id=policy_id)
            outputs = [
                self._run_local_policy(
                    prompt=p,
                    policy=resolved,
                    max_new_tokens=max_new_tokens,
                    include_trace=True,
                )
                for p in prompts
            ]
            pass_count = sum(1 for o in outputs if o.get("accepted"))
            fail_count = len(outputs) - pass_count
            avg_latency_ms = sum(o.get("latency_ms", 0) for o in outputs) / max(1, len(outputs))
            avg_tokens = sum(o.get("tokens", 0) for o in outputs) / max(1, len(outputs))
            return {
                "num_prompts": len(prompts),
                "success_rate": pass_count / max(1, len(prompts)),
                "avg_latency_ms": avg_latency_ms,
                "avg_tokens": avg_tokens,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "outputs": outputs,
            }

        self._require_remote_auth()
        payload = {
            "prompts": prompts,
            "policy_id": policy_id,
            "policy": policy,
            "max_new_tokens": max_new_tokens,
        }
        return self._post("/policies/evaluate", payload)

    def _require_remote_auth(self) -> None:
        if not (self._client.api_key or self._client.access_token):
            raise MechanexError(
                "Remote execution requires authentication. Use mx.set_key(...) "
                "or switch to local mode with mx.set_execution_mode('local')."
            )

    def _use_local(self) -> bool:
        return self._client.should_use_local()

    def _resolve_local_policy(self, policy: Optional[Dict[str, Any]], policy_id: Optional[str]) -> Dict[str, Any]:
        if policy is not None:
            return policy
        if policy_id is None:
            raise MechanexError("Local policy execution requires a policy or policy_id.")
        resolved = self._client._local_policies.get(policy_id)
        if resolved is None:
            raise MechanexError(f"Policy '{policy_id}' not found in local session.")
        return resolved

    def _run_local_policy(
        self,
        prompt: str,
        policy: Dict[str, Any],
        max_new_tokens: int,
        include_trace: bool,
    ) -> Dict[str, Any]:
        sampling = policy.get("sampling", {}) or {}
        steering = policy.get("steering", {}) or {}
        constraints = policy.get("constraints", {}) or {}
        verifiers = policy.get("verifiers", {}) or {}
        optimization = policy.get("optimization", {}) or {}

        method = (sampling.get("method") or "top-k").lower()
        if method in ("ads", "adaptive-determinantal-sampling"):
            raise MechanexError("ADS is not supported for local policy execution.")
        if method == "steering-perceptrons":
            raise MechanexError("Steering perceptrons are not supported for local policy execution.")
        steering_method = str(steering.get("method") or "").strip().lower()
        if steering_method == "steering-perceptrons":
            raise MechanexError("Steering perceptrons are not supported for local policy execution.")

        candidate_count = max(1, int(optimization.get("best_of_n", 1)), int(sampling.get("num_candidates", 1)))
        candidates = []

        for idx in range(candidate_count):
            text, latency_ms = self._local_generate_once(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                sampling=sampling,
                steering=steering,
                slot=idx,
            )
            tokens = len((text or "").split())
            constraints_ok, constraint_reason = self._check_constraints(text, constraints)
            verifier_ok, verifier_score, verifier_reason = self._run_verifiers(text, constraints, verifiers)
            accepted = constraints_ok and verifier_ok
            score = verifier_score
            if not constraints_ok:
                score -= 1.0

            candidates.append(
                {
                    "output": text,
                    "accepted": accepted,
                    "score": score,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "constraints_ok": constraints_ok,
                    "verifier_ok": verifier_ok,
                    "reason": " | ".join([r for r in [constraint_reason, verifier_reason] if r]) or "accepted",
                    "sampling_method": method,
                }
            )

        accepted = [c for c in candidates if c["accepted"]]
        winner = sorted(accepted or candidates, key=lambda x: (x["score"], -x["latency_ms"]), reverse=True)[0]

        return {
            "output": winner["output"],
            "accepted": winner["accepted"],
            "policy_id": policy.get("id"),
            "score": winner["score"],
            "latency_ms": winner["latency_ms"],
            "tokens": winner["tokens"],
            "constraints_ok": winner["constraints_ok"],
            "verifier_ok": winner["verifier_ok"],
            "trace": {
                "policy_name": policy.get("name"),
                "candidates": candidates,
                "selection": winner,
            } if include_trace else None,
        }

    def _local_generate_once(
        self,
        prompt: str,
        max_new_tokens: int,
        sampling: Dict[str, Any],
        steering: Optional[Dict[str, Any]] = None,
        slot: int = 0,
    ) -> Tuple[str, int]:
        local_model = getattr(self._client, "local_model", None)
        if local_model is None:
            raise MechanexError("No local model loaded. Call mx.load_model(...) first.")

        method = (sampling.get("method") or "top-k").lower()
        top_k = sampling.get("top_k", 50)
        top_p = sampling.get("top_p", 0.9)
        temperature = float(sampling.get("temperature", 0.7))
        diversity_strength = float(sampling.get("diversity_strength", 0.0) or 0.0)
        if diversity_strength > 0 and slot > 0:
            temperature = min(2.0, max(0.05, temperature + (slot % 3) * 0.1 * diversity_strength))

        if method in ("min-p", "typical", "guided-generation"):
            method = "top-p"
        if method in ("constrained-beam-search", "speculative-decoding", "ssd"):
            method = "top-k"
        if method == "ensemble-sampling":
            method = "top-p"

        if method not in ("top-k", "top-p", "greedy"):
            method = "top-k"

        kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "verbose": False}
        if method == "top-k":
            kwargs["top_k"] = top_k
        elif method == "top-p":
            kwargs["top_p"] = top_p
        elif method == "greedy":
            kwargs["top_k"] = 1

        # Temperature support varies by local model adapters.
        kwargs["temperature"] = temperature

        import torch

        steering = steering or {}
        vectors = None
        vector_inline = steering.get("vector")
        vector_id = steering.get("vector_id")
        strength = float(steering.get("strength", 0.0) or 0.0)
        if vector_inline is not None:
            vectors = vector_inline
        elif vector_id is not None:
            vectors = getattr(self._client, "_local_vectors", {}).get(vector_id)

        fwd_hooks = []
        if vectors and strength != 0:
            for layer, vec in vectors.items():
                layer_ref = str(layer)
                if layer_ref.isdigit():
                    hook_name = f"blocks.{int(layer_ref)}.hook_resid_pre"
                elif "." in layer_ref:
                    hook_name = layer_ref
                else:
                    hook_name = f"blocks.{layer_ref}.hook_resid_pre"
                if not isinstance(vec, torch.Tensor):
                    vec = torch.tensor(vec)
                vec = vec.to(local_model.cfg.device)
                delta = vec * strength
                def hook_fn(x, hook, delta_vec=delta):
                    x[:, :, :] = x[:, :, :] + delta_vec
                    return x
                fwd_hooks.append((hook_name, hook_fn))

        t0 = time.time()
        try:
            if fwd_hooks and hasattr(local_model, "hooks"):
                with local_model.hooks(fwd_hooks=fwd_hooks):
                    out = local_model.generate(prompt, **kwargs)
            else:
                out = local_model.generate(prompt, **kwargs)
        except TypeError:
            kwargs.pop("temperature", None)
            if fwd_hooks and hasattr(local_model, "hooks"):
                with local_model.hooks(fwd_hooks=fwd_hooks):
                    out = local_model.generate(prompt, **kwargs)
            else:
                out = local_model.generate(prompt, **kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        text = out if isinstance(out, str) else local_model.to_string(out)[0]
        return text, latency_ms

    def _check_constraints(self, text: str, constraints: Dict[str, Any]) -> Tuple[bool, str]:
        if constraints.get("json_mode") or constraints.get("json_schema") or constraints.get("required_fields"):
            ok, obj = self._extract_json(text)
            if not ok:
                return False, "constraint: invalid json"
            for field in constraints.get("required_fields", []):
                if field not in obj:
                    return False, f"constraint: missing required field '{field}'"

            schema = constraints.get("json_schema")
            if schema:
                valid, err = self._validate_json_schema(obj, schema)
                if not valid:
                    return False, f"constraint: schema violation ({err})"

        regex_pattern = constraints.get("regex_pattern")
        if regex_pattern and re.search(regex_pattern, text, flags=re.DOTALL) is None:
            return False, "constraint: regex mismatch"

        grammar = constraints.get("grammar")
        if grammar and not self._grammar_check(text, grammar):
            return False, "constraint: grammar mismatch"

        for term in constraints.get("forbidden_terms", []):
            if term and term.lower() in text.lower():
                return False, f"constraint: forbidden term '{term}'"

        return True, ""

    def _run_verifiers(
        self,
        text: str,
        constraints: Dict[str, Any],
        verifiers: Dict[str, Any],
    ) -> Tuple[bool, float, str]:
        enabled = [str(v).lower() for v in (verifiers.get("enabled") or [])]
        score = 1.0
        reasons = []

        if ("syntax" in enabled) or constraints.get("json_mode"):
            ok, _ = self._extract_json(text)
            if constraints.get("json_mode") and not ok:
                score -= 0.7
                reasons.append("syntax_verifier: invalid json")

        if "json_schema" in enabled and constraints.get("json_schema"):
            ok, obj = self._extract_json(text)
            if not ok:
                score -= 0.7
                reasons.append("json_schema_verifier: output is not valid json")
            else:
                valid, err = self._validate_json_schema(obj, constraints["json_schema"])
                if not valid:
                    score -= 0.7
                    reasons.append(f"json_schema_verifier: {err}")

        if "regex" in enabled and constraints.get("regex_pattern"):
            if re.search(constraints["regex_pattern"], text, flags=re.DOTALL) is None:
                score -= 0.4
                reasons.append("regex_verifier: pattern mismatch")

        return score >= 0.5, score, " | ".join(reasons)

    def _extract_json(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        txt = (text or "").strip()
        if not txt:
            return False, {}
        try:
            obj = json.loads(txt)
            return isinstance(obj, dict), obj if isinstance(obj, dict) else {}
        except Exception:
            pass

        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(txt[start:end + 1])
                return isinstance(obj, dict), obj if isinstance(obj, dict) else {}
            except Exception:
                return False, {}
        return False, {}

    def _validate_json_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
        if jsonschema is not None:
            try:
                jsonschema.validate(instance=obj, schema=schema)
                return True, ""
            except Exception as exc:
                return False, str(exc)

        required = schema.get("required", [])
        missing = [k for k in required if k not in obj]
        if missing:
            return False, f"missing fields: {', '.join(missing)}"
        return True, ""

    def _grammar_check(self, text: str, grammar: str) -> bool:
        if grammar.lower().startswith("regex:"):
            pattern = grammar.split(":", 1)[1].strip()
            return re.search(pattern, text, flags=re.DOTALL) is not None

        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", grammar)
        required = [a or b for a, b in tokens if (a or b)]
        if not required:
            return True

        cursor = 0
        lowered = text.lower()
        for token in required:
            idx = lowered.find(token.lower(), cursor)
            if idx < 0:
                return False
            cursor = idx + len(token)
        return True

    @staticmethod
    def strict_json_extraction(
        schema: Dict[str, Any],
        name: str = "strict_json_small_v1",
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "task_profile": {"name": "Strict JSON extraction"},
            "objective": {"name": "maximize json validity"},
            "sampling": {"method": "guided-generation", "temperature": 0.2, "top_p": 0.9},
            "constraints": {"json_mode": True, "json_schema": schema},
            "verifiers": {"enabled": ["syntax", "json_schema"], "repair_on_failure": True},
            "optimization": {"best_of_n": 2, "retry_on_failure": 2},
        }

    @staticmethod
    def fast_tool_router(name: str = "fast_tool_router_qwen") -> Dict[str, Any]:
        return {
            "name": name,
            "task_profile": {"name": "Reliable tool router"},
            "objective": {"name": "maximize correctness under latency budget", "latency_budget_ms": 1200},
            "sampling": {
                "method": "speculative-decoding",
                "temperature": 0.1,
                "top_k": 30,
                "speculative_steps": 6,
            },
            "constraints": {"json_mode": True, "required_fields": ["tool_name", "arguments"]},
            "verifiers": {"enabled": ["syntax", "tool_args"], "repair_on_failure": True},
            "optimization": {"best_of_n": 1, "retry_on_failure": 1, "fallback_methods": ["top-k"]},
        }

    @staticmethod
    def diverse_chatbot(name: str = "diverse_chatbot_ads") -> Dict[str, Any]:
        return {
            "name": name,
            "task_profile": {"name": "Concise customer support"},
            "objective": {"name": "maximize quality"},
            "sampling": {
                "method": "ads",
                "temperature": 0.9,
                "top_p": 0.95,
                "ads_subset_size": 16,
                "ads_beta": 0.35,
                "num_candidates": 3,
                "diversity_strength": 0.7,
            },
            "constraints": {"forbidden_terms": []},
            "verifiers": {"enabled": []},
            "optimization": {"best_of_n": 3, "retry_on_failure": 0},
        }

    @staticmethod
    def ensemble_vote(
        models: List[str],
        name: str = "ensemble_vote_v1",
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "task_profile": {"name": "Low-hallucination analyst mode"},
            "objective": {"name": "reduce hallucinations"},
            "sampling": {
                "method": "ensemble-sampling",
                "ensemble_models": models,
                "voting": "majority",
                "temperature": 0.4,
                "top_p": 0.9,
            },
            "constraints": {},
            "verifiers": {"enabled": ["factuality"], "repair_on_failure": False},
            "optimization": {"best_of_n": 1},
        }
