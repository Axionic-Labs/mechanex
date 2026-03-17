from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from itertools import product
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
        # Filter out None values (Fixes BUG-2 in manual report)
        payload = {k: v for k, v in payload.items() if v is not None}
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
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._post("/policies/evaluate", payload)

    def publish_preset(
        self,
        name: str,
        policy: Dict[str, Any],
        visibility: str = "private",
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        if self._use_local():
            preset_id = str(uuid.uuid4())
            self._client._local_policy_presets[preset_id] = {
                "id": preset_id,
                "name": name,
                "policy": policy,
                "visibility": visibility,
                "tags": tags or [],
                "description": description,
            }
            return preset_id

        self._require_remote_auth()
        payload = {
            "name": name,
            "policy": policy,
            "visibility": visibility,
            "tags": tags or [],
            "description": description,
        }
        resp = self._post("/policies/presets/publish", payload)
        preset_id = resp.get("preset_id")
        if not preset_id:
            raise MechanexError(f"Failed to publish preset: {resp}")
        return preset_id

    def list_presets(self, include_public: bool = True) -> List[Dict[str, Any]]:
        if self._use_local():
            return list(self._client._local_policy_presets.values())

        self._require_remote_auth()
        suffix = "true" if include_public else "false"
        resp = self._get(f"/policies/presets?include_public={suffix}")
        return resp if isinstance(resp, list) else []

    def clone_preset(self, preset_id: str) -> str:
        if self._use_local():
            preset = self._client._local_policy_presets.get(preset_id)
            if not preset:
                raise MechanexError(f"Preset '{preset_id}' not found in local session.")
            return self.save(preset.get("policy") or {})

        self._require_remote_auth()
        resp = self._post(f"/policies/presets/{preset_id}/clone", {})
        policy_id = resp.get("policy_id")
        if not policy_id:
            raise MechanexError(f"Failed to clone preset: {resp}")
        return policy_id

    def auto_tune(
        self,
        prompts: List[str],
        base_policy: Optional[Dict[str, Any]] = None,
        base_policy_id: Optional[str] = None,
        search_space: Optional[Dict[str, List[Any]]] = None,
        max_trials: int = 24,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        if self._use_local():
            if base_policy is not None:
                policy = base_policy
            elif base_policy_id:
                policy = self._resolve_local_policy(policy=None, policy_id=base_policy_id)
            else:
                policy = {"sampling": {"method": "top-k", "temperature": 0.7, "top_p": 0.9}, "optimization": {"best_of_n": 1}}

            ss = search_space or {}
            methods = ss.get("sampling.method") or [policy.get("sampling", {}).get("method", "top-k")]
            temps = ss.get("sampling.temperature") or [policy.get("sampling", {}).get("temperature", 0.7)]
            top_ps = ss.get("sampling.top_p") or [policy.get("sampling", {}).get("top_p", 0.9)]
            best_of_ns = ss.get("optimization.best_of_n") or [policy.get("optimization", {}).get("best_of_n", 1)]

            trials: List[Dict[str, Any]] = []
            for idx, (method, temp, top_p, best_of_n) in enumerate(product(methods, temps, top_ps, best_of_ns)):
                if idx >= max(1, int(max_trials)):
                    break
                candidate = json.loads(json.dumps(policy))
                candidate.setdefault("sampling", {})
                candidate.setdefault("optimization", {})
                candidate["sampling"]["method"] = method
                candidate["sampling"]["temperature"] = float(temp)
                candidate["sampling"]["top_p"] = float(top_p)
                candidate["optimization"]["best_of_n"] = int(best_of_n)

                outputs = [
                    self._run_local_policy(
                        prompt=p,
                        policy=candidate,
                        max_new_tokens=max_new_tokens,
                        include_trace=False,
                    )
                    for p in prompts
                ]
                pass_count = sum(1 for o in outputs if o.get("accepted"))
                fail_count = len(outputs) - pass_count
                avg_score = sum(float(o.get("score", 0.0)) for o in outputs) / max(1, len(outputs))
                avg_latency = sum(float(o.get("latency_ms", 0.0)) for o in outputs) / max(1, len(outputs))
                trials.append(
                    {
                        "policy": candidate,
                        "success_rate": pass_count / max(1, len(outputs)),
                        "avg_score": avg_score,
                        "avg_latency_ms": avg_latency,
                        "pass_count": pass_count,
                        "fail_count": fail_count,
                    }
                )

            if not trials:
                raise MechanexError("No auto-tune trials executed.")

            best = sorted(trials, key=lambda t: (t["success_rate"], t["avg_score"], -t["avg_latency_ms"]), reverse=True)[0]
            return {"best_policy": best["policy"], "best_trial": best, "trials": trials}

        self._require_remote_auth()
        payload = {
            "prompts": prompts,
            "base_policy_id": base_policy_id,
            "base_policy": base_policy,
            "search_space": search_space or {},
            "max_trials": max_trials,
            "max_new_tokens": max_new_tokens,
        }
        return self._post("/policies/autotune", payload)

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
        steering = self._apply_steering_preset(policy).get("steering", {}) or {}
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
        retry_rounds = max(0, int(optimization.get("retry_on_failure", 0)))
        confidence_retry = bool(optimization.get("confidence_triggered_regeneration", False))
        confidence_threshold = float(optimization.get("confidence_threshold", 0.5))
        if confidence_retry:
            retry_rounds = max(retry_rounds, 1)

        candidates: List[Dict[str, Any]] = []
        retry_events: List[Dict[str, Any]] = []
        active_sampling = dict(sampling)

        winner: Dict[str, Any] = {
            "output": "",
            "accepted": False,
            "score": 0.0,
            "latency_ms": 0,
            "tokens": 0,
            "constraints_ok": False,
            "verifier_ok": False,
            "reason": "no candidates",
            "sampling_method": method,
        }
        for round_idx in range(retry_rounds + 1):
            round_candidates: List[Dict[str, Any]] = []
            for idx in range(candidate_count):
                text, latency_ms = self._local_generate_once(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    sampling=active_sampling,
                    steering=steering,
                    slot=idx,
                    round_idx=round_idx,
                )
                tokens = len((text or "").split())
                constraints_ok, constraint_reason = self._check_constraints(text, constraints)
                verifier_ok, verifier_score, verifier_reason = self._run_verifiers(text, constraints, verifiers)
                accepted = constraints_ok and verifier_ok
                score = verifier_score
                if not constraints_ok:
                    score -= 1.0

                round_candidates.append(
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
                        "round": round_idx,
                        "slot": idx,
                    }
                )

            candidates.extend(round_candidates)
            accepted = [c for c in candidates if c["accepted"]]
            winner = sorted(accepted or candidates, key=lambda x: (x["score"], -x["latency_ms"]), reverse=True)[0]
            needs_confidence_retry = confidence_retry and float(winner.get("score", 0.0)) < confidence_threshold
            if winner["accepted"] and not needs_confidence_retry:
                break
            if round_idx >= retry_rounds:
                break
            retry_events.append(
                {
                    "round": round_idx + 1,
                    "trigger": "confidence" if needs_confidence_retry else "accepted=false",
                    "winner_score": winner.get("score", 0.0),
                }
            )
            active_sampling = self._sampling_for_retry(active_sampling, round_idx + 1)

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
                "retry_events": retry_events,
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
        round_idx: int = 0,
    ) -> Tuple[str, int]:
        local_model = getattr(self._client, "local_model", None)
        if local_model is None:
            raise MechanexError("No local model loaded. Call mx.load_model(...) first.")

        method = (sampling.get("method") or "top-k").lower()
        top_k = sampling.get("top_k", 50)
        top_p = sampling.get("top_p", 0.9)
        temperature = float(sampling.get("temperature", 0.7))
        if bool(sampling.get("adaptive_temperature")):
            schedule = sampling.get("adaptive_temperature_schedule") or []
            temperature = float(
                self._scheduled_value(
                    schedule,
                    index=(round_idx * max(1, int(sampling.get("num_candidates", 1)))) + slot,
                    default=temperature,
                )
            )
        diversity_strength = float(sampling.get("diversity_strength", 0.0) or 0.0)
        if diversity_strength > 0 and slot > 0:
            temperature = min(2.0, max(0.05, temperature + (slot % 3) * 0.1 * diversity_strength))

        top_p = sampling.get("top_p", 0.9)
        if bool(sampling.get("adaptive_top_p")):
            p_schedule = sampling.get("adaptive_top_p_schedule") or []
            top_p = float(
                self._scheduled_value(
                    p_schedule,
                    index=(round_idx * max(1, int(sampling.get("num_candidates", 1)))) + slot,
                    default=top_p,
                )
            )

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

    def _scheduled_value(self, values: List[Any], index: int, default: float) -> float:
        if not values:
            return float(default)
        if index < 0:
            index = 0
        if index >= len(values):
            index = len(values) - 1
        try:
            return float(values[index])
        except Exception:
            return float(default)

    def _sampling_for_retry(self, sampling: Dict[str, Any], retry_round: int) -> Dict[str, Any]:
        out = dict(sampling)
        out["temperature"] = max(0.05, float(out.get("temperature", 0.7)) - (0.08 * retry_round))
        if out.get("top_p") is not None:
            out["top_p"] = max(0.2, float(out.get("top_p", 0.9)) - (0.04 * retry_round))
        return out

    def _apply_steering_preset(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        out = json.loads(json.dumps(policy))
        steering = out.setdefault("steering", {})
        objective = out.setdefault("objective", {"name": "quality"})
        verifiers = out.setdefault("verifiers", {})
        enabled = [str(v).lower() for v in (verifiers.get("enabled") or [])]
        constraints = out.setdefault("constraints", {})
        preset = str(steering.get("preset") or "").strip().lower()
        if not preset:
            return out

        steering["enabled"] = True
        strength = float(steering.get("strength", 0.0) or 0.0)
        if preset == "brevity":
            steering["strength"] = max(strength, 0.35)
            if "concise" not in str(objective.get("name", "")).lower():
                objective["name"] = "concise response quality"
        elif preset == "truthfulness":
            steering["strength"] = max(strength, 0.45)
            if "factuality" not in enabled:
                enabled.append("factuality")
        elif preset == "json_compliance":
            constraints["json_mode"] = True
            if "syntax" not in enabled:
                enabled.append("syntax")
            if constraints.get("json_schema") and "json_schema" not in enabled:
                enabled.append("json_schema")
        elif preset == "format_rigidity":
            steering["strength"] = max(strength, 0.5)
            if constraints.get("regex_pattern") and "regex" not in enabled:
                enabled.append("regex")
        elif preset == "refusal_strength":
            steering["strength"] = max(strength, 0.65)
        elif preset == "safety_style":
            steering["strength"] = max(strength, 0.55)
            if "factuality" not in enabled:
                enabled.append("factuality")
        elif preset == "domain_specific":
            steering["strength"] = max(strength, 0.4)
        elif preset == "tone_control":
            steering["strength"] = max(strength, 0.25)

        verifiers["enabled"] = enabled
        return out

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

        if "code_compile" in enabled:
            ok, err = self._code_compile_check(text, verifiers.get("code_language", "python"))
            if not ok:
                score -= 0.8
                reasons.append(f"code_compile_verifier: {err}")

        if "unit_tests" in enabled and verifiers.get("code_unit_tests"):
            ok, err = self._code_unit_test_check(
                text=text,
                language=verifiers.get("code_language", "python"),
                tests=verifiers.get("code_unit_tests") or [],
                timeout_ms=int(verifiers.get("unit_test_timeout_ms", 1500)),
            )
            if not ok:
                score -= 0.9
                reasons.append(f"unit_tests_verifier: {err}")

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

    def _extract_code_text(self, text: str) -> str:
        txt = (text or "").strip()
        block = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", txt, flags=re.DOTALL)
        if block:
            return block.group(1).strip()
        return txt

    def _code_compile_check(self, text: str, language: str = "python") -> Tuple[bool, str]:
        lang = (language or "python").strip().lower()
        if lang != "python":
            return False, f"unsupported language '{language}'"
        code = self._extract_code_text(text)
        try:
            compile(code, "<candidate>", "exec")
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _code_unit_test_check(
        self,
        text: str,
        language: str,
        tests: List[str],
        timeout_ms: int,
    ) -> Tuple[bool, str]:
        lang = (language or "python").strip().lower()
        if lang != "python":
            return False, f"unsupported language '{language}'"
        code = self._extract_code_text(text)
        runner = [sys.executable, "-I", "-c", code + "\n\n" + "\n\n".join(tests)]
        try:
            result = subprocess.run(
                runner,
                capture_output=True,
                text=True,
                timeout=max(0.1, timeout_ms / 1000.0),
            )
        except subprocess.TimeoutExpired:
            return False, "unit tests timed out"
        except Exception as exc:
            return False, str(exc)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unit tests failed").strip()
            return False, err[:500]
        return True, ""

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

    @staticmethod
    def steering_preset(
        preset: str,
        name: Optional[str] = None,
        sampling_method: str = "top-k",
    ) -> Dict[str, Any]:
        preset_name = (preset or "brevity").strip().lower()
        return {
            "name": name or f"steering_{preset_name}",
            "task_profile": {"name": "Steering preset policy"},
            "objective": {"name": "quality"},
            "sampling": {"method": sampling_method, "temperature": 0.7, "top_p": 0.9},
            "steering": {"enabled": True, "preset": preset_name, "strength": 0.35},
            "constraints": {},
            "verifiers": {"enabled": []},
            "optimization": {"best_of_n": 1, "retry_on_failure": 1},
        }
