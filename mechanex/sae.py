import json
import numpy as np
import torch
from typing import List, Optional, Dict, Any, Union
from .base import _BaseModule
from .errors import AuthenticationError, MechanexError, APIError

class SAEModule(_BaseModule):
    """
    Module for SAE-based steering and behavior management.
    """

    def __init__(self, client):
        super().__init__(client)
        self.sae_release = "gpt2-small-res-jb" # Default
        self._local_saes = {}

    def _get_local_model(self):
        local_model = getattr(self._client, 'local_model', None)
        if local_model is None:
            raise MechanexError("No local model set. Use mx.set_local_model(model) to enable local computation.")
        return local_model

    def _resolve_layer_node(self, idx: int) -> str:
        local_model = self._get_local_model()
        # Find the best match for this layer index in the model
        
        # If model has hook_points (TransformerLens)
        if hasattr(local_model, "hook_dict"):
            candidates = [n for n in local_model.hook_dict.keys() if f"blocks.{idx}." in n or n.endswith(f".{idx}")]
            if not candidates:
                return f"blocks.{idx}.hook_resid_pre"
            
            # Prioritize resid_pre
            for c in candidates:
                if "resid_pre" in c: return c
            return candidates[0]
        
        return f"blocks.{idx}.hook_resid_pre"

    def create_behavior(
        self,
        behavior_name: str,
        prompts: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        description: str = "",
        steering_vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Define a new behavior. Falls back to local computation if remote fails.
        """
        try:
            if self._client.api_key:
                payload = {
                "behavior_name": behavior_name,
                "prompts": prompts,
                "positive_answers": positive_answers,
                "negative_answers": negative_answers,
                "description": description
            }
            if steering_vector_id:
                payload["steering_vector_id"] = steering_vector_id
                
            return self._post("/behaviors/create", payload)
        except (AuthenticationError, MechanexError, APIError) as e:
            local_model = getattr(self._client, 'local_model', None)
            if local_model is not None:
                print(f"Remote behavior creation failed ({str(e)}). Computing locally with sae-lens...")
                return self._compute_behavior_locally(behavior_name, prompts, positive_answers, negative_answers)
            raise e

    def create_behavior_from_jsonl(
        self,
        behavior_name: str,
        dataset_path: str,
        description: str = "",
        steering_vector_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Helper to create a behavior from a .jsonl file.
        """
        prompts, positive_answers, negative_answers = [], [], []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "prompt" in data: prompts.append(data["prompt"])
                if "positive_answer" in data: positive_answers.append(data["positive_answer"])
                if "negative_answer" in data: negative_answers.append(data["negative_answer"])
        
        return self.create_behavior(
            behavior_name=behavior_name,
            prompts=prompts,
            positive_answers=positive_answers,
            negative_answers=negative_answers,
            description=description,
            steering_vector_id=steering_vector_id
        )

    def _compute_behavior_locally(self, name, prompts, pos, neg):
        from sae_lens import SAE
        local_model = self._get_local_model()
        
        layer_idx = int(getattr(local_model.cfg, "n_layers", 12) * 2 / 3)
        hook_name = self._resolve_layer_node(layer_idx)
        
        release = self.sae_release
        sae_id = hook_name 
        
        print(f"Loading SAE for {hook_name} ({release})...")
        sae, cfg_dict, sparsity = SAE.from_pretrained(release, sae_id)
        sae.to(local_model.cfg.device)
        
        diff = self._compute_sae_diff_locally(local_model, sae, hook_name, prompts, pos, neg)
        
        res = {
            "id": f"local-{name}",
            "behavior_name": name,
            "sae_baseline": diff,
            "hook_name": hook_name,
            "sae_release": release
        }
        if not hasattr(self._client, "_local_behaviors"):
            self._client._local_behaviors = {}
        self._client._local_behaviors[name] = res
        return res

    def _compute_sae_diff_locally(self, model, sae, hook_name, prompts, pos_answers, neg_answers):
        diffs = []
        for p, pos, neg in zip(prompts, pos_answers, neg_answers):
            pos_tokens = model.to_tokens(p + pos)
            _, cache = model.run_with_cache(pos_tokens, names_filter=[hook_name])
            pos_acts = cache[hook_name]
            pos_sae_acts = sae.encode(pos_acts).mean(dim=1) 
            
            neg_tokens = model.to_tokens(p + neg)
            _, cache = model.run_with_cache(neg_tokens, names_filter=[hook_name])
            neg_acts = cache[hook_name]
            neg_sae_acts = sae.encode(neg_acts).mean(dim=1)
            
            diffs.append((pos_sae_acts - neg_sae_acts).detach().cpu().numpy())
            
        return np.mean(diffs, axis=0)

    def list_behaviors(self) -> List[Dict[str, Any]]:
        """Returns behaviors. Combines remote and local if available."""
        remote_behaviors = []
        try:
            if self._client.api_key:
                remote_behaviors = self._get("/behaviors")
        except:
            pass
            
        local_behaviors = list(getattr(self._client, "_local_behaviors", {}).values())
        return remote_behaviors + local_behaviors

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        behavior_names: Optional[List[str]] = None,
        auto_correct: bool = True,
        force_steering: Optional[List[str]] = None
    ) -> str:
        """
        Generation with SAE monitoring. Falls back to local if needed.
        """
        try:
            if self._client.api_key:
                payload = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "behavior_names": behavior_names,
                "auto_correct": auto_correct,
                "force_steering": force_steering
            }
            response = self._post("/sae/generate", payload)
            return response.get("text", "")
        except (AuthenticationError, MechanexError, APIError):
            local_model = getattr(self._client, 'local_model', None)
            if local_model is not None:
                output = local_model.generate(prompt, max_new_tokens=max_new_tokens)
                return output if isinstance(output, str) else local_model.to_string(output)[0]
            raise
