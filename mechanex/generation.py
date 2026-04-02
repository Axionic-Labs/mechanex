from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import _BaseModule
from .errors import MechanexError


class GenerationModule(_BaseModule):
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        sampling_method: str = "top-k",
        top_k: int = 50,
        top_p: float = 0.9,
        steering_strength: float = 0,
        steering_vector=None,
        temperature: float = 0.7,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        ads_subset_size: Optional[int] = None,
        ads_beta: Optional[float] = None,
        regex_pattern: Optional[str] = None,
        grammar: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        draft_model: Optional[str] = None,
        ensemble_models: Optional[List[str]] = None,
        best_of_n: int = 1,
        adaptive_temperature: bool = False,
        adaptive_temperature_schedule: Optional[List[float]] = None,
        adaptive_top_p: bool = False,
        adaptive_top_p_schedule: Optional[List[float]] = None,
        steering_preset: Optional[str] = None,
        confidence_triggered_regeneration: bool = False,
        confidence_threshold: float = 0.5,
        code_unit_tests: Optional[List[str]] = None,
        policy: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        include_trace: bool = False,
    ) -> str:
        """
        Generates text with runtime control policies.
        Supports remote and local execution.
        """
        advanced_methods = {
            "ads",
            "adaptive-determinantal-sampling",
            "constrained-beam-search",
            "speculative-decoding",
            "ssd",
            "guided-generation",
            "ensemble-sampling",
        }
        needs_policy_path = (
            policy is not None
            or policy_id is not None
            or sampling_method in advanced_methods
            or min_p is not None
            or typical_p is not None
            or regex_pattern is not None
            or grammar is not None
            or json_schema is not None
            or draft_model is not None
            or bool(ensemble_models)
            or best_of_n > 1
            or adaptive_temperature
            or bool(adaptive_temperature_schedule)
            or adaptive_top_p
            or bool(adaptive_top_p_schedule)
            or steering_preset is not None
            or confidence_triggered_regeneration
            or bool(code_unit_tests)
        )
        use_local = self._client.should_use_local()

        try:
            if not use_local:
                if not (self._client.api_key or self._client.access_token):
                    raise MechanexError(
                        "Remote execution requires authentication. "
                        "Use mx.set_key(...) or mx.set_execution_mode('local')."
                    )
                if needs_policy_path:
                    runtime_policy = policy or self._build_inline_policy(
                        sampling_method=sampling_method,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        typical_p=typical_p,
                        ads_subset_size=ads_subset_size,
                        ads_beta=ads_beta,
                        regex_pattern=regex_pattern,
                        grammar=grammar,
                        json_schema=json_schema,
                        draft_model=draft_model,
                        ensemble_models=ensemble_models,
                        steering_vector=steering_vector,
                        steering_strength=steering_strength,
                        best_of_n=best_of_n,
                        adaptive_temperature=adaptive_temperature,
                        adaptive_temperature_schedule=adaptive_temperature_schedule,
                        adaptive_top_p=adaptive_top_p,
                        adaptive_top_p_schedule=adaptive_top_p_schedule,
                        steering_preset=steering_preset,
                        confidence_triggered_regeneration=confidence_triggered_regeneration,
                        confidence_threshold=confidence_threshold,
                        code_unit_tests=code_unit_tests,
                    )
                    payload: Dict[str, Any] = {
                        "prompt": prompt,
                        "policy_id": policy_id,
                        "policy": None if policy_id else runtime_policy,
                        "max_new_tokens": max_tokens,
                        "include_trace": include_trace,
                    }
                    response = self._post("/policies/run", payload)
                    return response.get("output", "")

                payload = {
                    "prompt": prompt,
                    "sampling_method": sampling_method,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "top_p": top_p,
                    "steering_vector_id": steering_vector if isinstance(steering_vector, str) else None,
                    "steering_strength": steering_strength,
                }
                response = self._post_sse("/generate", payload)
                return response.get("output", "")
            else:
                if needs_policy_path:
                    runtime_policy = policy or self._build_inline_policy(
                        sampling_method=sampling_method,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        typical_p=typical_p,
                        ads_subset_size=ads_subset_size,
                        ads_beta=ads_beta,
                        regex_pattern=regex_pattern,
                        grammar=grammar,
                        json_schema=json_schema,
                        draft_model=draft_model,
                        ensemble_models=ensemble_models,
                        steering_vector=steering_vector,
                        steering_strength=steering_strength,
                        best_of_n=best_of_n,
                        adaptive_temperature=adaptive_temperature,
                        adaptive_temperature_schedule=adaptive_temperature_schedule,
                        adaptive_top_p=adaptive_top_p,
                        adaptive_top_p_schedule=adaptive_top_p_schedule,
                        steering_preset=steering_preset,
                        confidence_triggered_regeneration=confidence_triggered_regeneration,
                        confidence_threshold=confidence_threshold,
                        code_unit_tests=code_unit_tests,
                    )
                    response = self._client.policy.run(
                        prompt=prompt,
                        policy=None if policy_id else runtime_policy,
                        policy_id=policy_id,
                        max_new_tokens=max_tokens,
                        include_trace=include_trace,
                    )
                    return response.get("output", "")
        except Exception as e:
            if not getattr(self._client, "local_model", None) or not use_local:
                raise e

        return self._local_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            sampling_method=sampling_method,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            steering_strength=steering_strength,
            steering_vector=steering_vector,
        )

    def _build_inline_policy(
        self,
        sampling_method: str,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: Optional[float],
        typical_p: Optional[float],
        ads_subset_size: Optional[int],
        ads_beta: Optional[float],
        regex_pattern: Optional[str],
        grammar: Optional[str],
        json_schema: Optional[Dict[str, Any]],
        draft_model: Optional[str],
        ensemble_models: Optional[List[str]],
        steering_vector,
        steering_strength: float,
        best_of_n: int,
        adaptive_temperature: bool = False,
        adaptive_temperature_schedule: Optional[List[float]] = None,
        adaptive_top_p: bool = False,
        adaptive_top_p_schedule: Optional[List[float]] = None,
        steering_preset: Optional[str] = None,
        confidence_triggered_regeneration: bool = False,
        confidence_threshold: float = 0.5,
        code_unit_tests: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        method = sampling_method
        if ensemble_models:
            method = "ensemble-sampling"

        constraints: Dict[str, Any] = {}
        if json_schema is not None:
            constraints["json_mode"] = True
            constraints["json_schema"] = json_schema
        if regex_pattern is not None:
            constraints["regex_pattern"] = regex_pattern
        if grammar is not None:
            constraints["grammar"] = grammar

        sampling: Dict[str, Any] = {
            "method": method,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        if min_p is not None:
            sampling["min_p"] = min_p
        if typical_p is not None:
            sampling["typical_p"] = typical_p
        if ads_subset_size is not None:
            sampling["ads_subset_size"] = ads_subset_size
        if ads_beta is not None:
            sampling["ads_beta"] = ads_beta
        if draft_model is not None:
            sampling["draft_model"] = draft_model
        if ensemble_models:
            sampling["ensemble_models"] = ensemble_models
        if adaptive_temperature:
            sampling["adaptive_temperature"] = True
        if adaptive_temperature_schedule:
            sampling["adaptive_temperature_schedule"] = adaptive_temperature_schedule
            sampling["adaptive_temperature"] = True
        if adaptive_top_p:
            sampling["adaptive_top_p"] = True
        if adaptive_top_p_schedule:
            sampling["adaptive_top_p_schedule"] = adaptive_top_p_schedule
            sampling["adaptive_top_p"] = True

        verifiers: Dict[str, Any] = {"enabled": [], "repair_on_failure": True}
        if json_schema is not None:
            verifiers["enabled"] = ["syntax", "json_schema"]
        elif regex_pattern is not None:
            verifiers["enabled"] = ["regex"]
        if code_unit_tests:
            verifiers["enabled"] = list(dict.fromkeys([*verifiers["enabled"], "unit_tests"]))
            verifiers["code_language"] = "python"
            verifiers["code_unit_tests"] = code_unit_tests

        steering: Dict[str, Any] = {"enabled": False, "strength": steering_strength}
        if steering_preset:
            steering["preset"] = steering_preset
            steering["enabled"] = True
        if isinstance(steering_vector, str):
            steering["enabled"] = True
            steering["vector_id"] = steering_vector
        elif isinstance(steering_vector, dict):
            steering["enabled"] = True
            steering["vector"] = steering_vector
        elif steering_strength != 0:
            steering["enabled"] = True

        return {
            "task_profile": {"name": "custom runtime generation"},
            "objective": {"name": "quality"},
            "sampling": sampling,
            "steering": steering,
            "constraints": constraints,
            "verifiers": verifiers,
            "optimization": {
                "best_of_n": max(1, best_of_n),
                "confidence_triggered_regeneration": confidence_triggered_regeneration,
                "confidence_threshold": confidence_threshold,
            },
        }

    def _local_generate(
        self,
        prompt: str,
        max_tokens: int,
        sampling_method: str,
        top_k: int,
        top_p: float,
        temperature: float,
        steering_strength: float,
        steering_vector,
    ) -> str:
        local_model = getattr(self._client, "local_model", None)
        if local_model is None:
            raise MechanexError("No local model available for fallback.")

        if sampling_method in ("ads", "adaptive-determinantal-sampling"):
            raise MechanexError("ADS is not supported for local execution.")

        supported_local_methods = ["top-k", "top-p", "greedy", "min-p", "typical", None]
        if sampling_method not in supported_local_methods:
            raise MechanexError(f"Sampling method '{sampling_method}' is not supported for local models.")

        import torch

        vectors = None
        if steering_vector is not None:
            if isinstance(steering_vector, str):
                vectors = getattr(self._client, "_local_vectors", {}).get(steering_vector)
            elif isinstance(steering_vector, dict):
                vectors = steering_vector

        fwd_hooks = []
        if vectors and steering_strength != 0:
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
                delta = vec * steering_strength
                def hook_fn(x, hook, delta_vec=delta):
                    x[:, :, :] = x[:, :, :] + delta_vec
                    return x
                fwd_hooks.append((hook_name, hook_fn))

        if sampling_method in ("min-p", "typical"):
            sampling_method = "top-p"

        gen_kwargs = {"max_new_tokens": max_tokens, "verbose": False, "temperature": temperature}
        if sampling_method == "top-k":
            gen_kwargs["top_k"] = top_k
        elif sampling_method == "top-p":
            gen_kwargs["top_p"] = top_p
        elif sampling_method == "greedy":
            gen_kwargs["top_k"] = 1

        try:
            if fwd_hooks:
                with local_model.hooks(fwd_hooks=fwd_hooks):
                    output = local_model.generate(prompt, **gen_kwargs)
            else:
                output = local_model.generate(prompt, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("temperature", None)
            if fwd_hooks:
                with local_model.hooks(fwd_hooks=fwd_hooks):
                    output = local_model.generate(prompt, **gen_kwargs)
            else:
                output = local_model.generate(prompt, **gen_kwargs)

        return output if isinstance(output, str) else local_model.to_string(output)[0]
