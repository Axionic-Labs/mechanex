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
                return self._compute_behavior_locally(behavior_name, prompts, positive_answers, negative_answers, steering_vector_id)
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

    def _resolve_sae_id(self, release: str, layer_idx: int) -> Optional[str]:
        """Resolves the SAE ID from the mapping file using list index."""
        try:
            import yaml
            from pathlib import Path
            
            yaml_path = Path(__file__).parent / "utils" / "sae_mapping.yaml"
            if not yaml_path.exists():
                return None
            
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            if release not in data:
                return None
                
            saes = data[release].get("saes", [])
            # User requested to use indices to reference the key
            if 0 <= layer_idx < len(saes):
                return saes[layer_idx]["id"]
            
            return None
        except Exception as e:
            print(f"Warning: Error resolving SAE ID: {e}")
            return None

    def _compute_behavior_locally(self, name, prompts, pos, neg, steering_vector_id=None):
        from sae_lens import SAE
        local_model = self._get_local_model()
        
        layer_idx = int(getattr(local_model.cfg, "n_layers", 12) * 2 / 3)
        hook_name = self._resolve_layer_node(layer_idx)
        
        release = self.sae_release
        
        # Resolve correct SAE ID (might be differnt from hook_name)
        sae_id = self._resolve_sae_id(release, layer_idx)
        if not sae_id:
            sae_id = hook_name # Fallback

        # Check for cached SAE
        if not hasattr(self, "_local_saes"):
            self._local_saes = {}
        
        if sae_id in self._local_saes:
            sae = self._local_saes[sae_id]
        else:
            print(f"Loading SAE for {hook_name} (ID: {sae_id}, Release: {release})...")
            # If explicit path in mapping, might need other handling, but sae_lens usually expects simple ID
            sae, cfg_dict, sparsity = SAE.from_pretrained(release, sae_id)
            sae.to(local_model.cfg.device)
            self._local_saes[sae_id] = sae
        
        # Compute differences
        sae_diffs = []
        resid_diffs = []
        
        for p, pos_txt, neg_txt in zip(prompts, pos, neg):
            # Positive
            pos_tokens = local_model.to_tokens(p + pos_txt)
            _, pos_cache = local_model.run_with_cache(pos_tokens, names_filter=[hook_name])
            pos_resid = pos_cache[hook_name]
            
            # Negative
            neg_tokens = local_model.to_tokens(p + neg_txt)
            _, neg_cache = local_model.run_with_cache(neg_tokens, names_filter=[hook_name])
            neg_resid = neg_cache[hook_name]
            
            # SAE Diff
            pos_sae = sae.encode(pos_resid).mean(dim=1)
            neg_sae = sae.encode(neg_resid).mean(dim=1)
            sae_diffs.append((pos_sae - neg_sae).detach())
            
            # Residual Diff
            pos_r = pos_resid.mean(dim=1)
            neg_r = neg_resid.mean(dim=1)
            resid_diffs.append((pos_r - neg_r).detach())
            
        sae_baseline = torch.stack(sae_diffs).mean(dim=0).cpu().numpy()
        steering_vec = torch.stack(resid_diffs).mean(dim=0).cpu().numpy()
        
        res = {
            "id": f"local-{name}",
            "behavior_name": name,
            "sae_baseline": sae_baseline,
            "steering_vector": steering_vec,
            "steering_vector_id": steering_vector_id,
            "hook_name": hook_name,
            "sae_id": sae_id,
            "sae_release": release
        }
        if not hasattr(self._client, "_local_behaviors"):
            self._client._local_behaviors = {}
        self._client._local_behaviors[name] = res
        return res

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
                "force_steering": force_steering
            }
            response = self._post("/sae/generate", payload)
            return response.get("text", "")
        except (AuthenticationError, MechanexError, APIError):

            local_model = getattr(self._client, 'local_model', None)
            if local_model is not None:
                # Basic generation without behaviors
                if not behavior_names:
                    output = local_model.generate(prompt, max_new_tokens=max_new_tokens)
                    return output if isinstance(output, str) else local_model.to_string(output)[0]

                # SAE-monitored generation
                fwd_hooks = []
                
                # Check for cached SAEs
                if not hasattr(self, "_local_saes"):
                    self._local_saes = {}

                for b_name in behavior_names:
                    # Find behavior
                    behavior = self._client._local_behaviors.get(b_name)
                    if not behavior:
                        print(f"Warning: Local behavior '{b_name}' not found. Skipping.")
                        continue
                    
                    hook_name = behavior["hook_name"]
                    sae_id = behavior.get("sae_id", hook_name)
                    sae_release = behavior["sae_release"]
                    
                    # Load SAE
                    if sae_id in self._local_saes:
                        sae = self._local_saes[sae_id]
                    else:
                        from sae_lens import SAE
                        print(f"Loading SAE for {hook_name} (ID: {sae_id})...")
                        sae, _, _ = SAE.from_pretrained(sae_release, sae_id)
                        sae.to(local_model.cfg.device)
                        self._local_saes[sae_id] = sae

                    # Prepare vectors
                    sae_baseline = torch.tensor(behavior["sae_baseline"]).to(local_model.cfg.device)
                    # Use provided steering vector ID if available (highest priority)
                    sv_id = behavior.get("steering_vector_id")
                    steering_vec = None
                    
                    if sv_id:
                        try:
                            # Get vectors dict {layer: tensor}
                            vectors = self._client.steering.get_vectors(sv_id)
                            # Find vector for current layer? Or apply to all?
                            # For simplicity in this hook, we only apply if the layer matches
                            # or we'd need multiple hooks.
                            # Usually SAE and steering operate on the same blocks.
                            # We'll try to find a vector for the layer corresponding to hook_name
                            try:
                                # Extract layer index from hook_name "blocks.X.hook_resid_pre"
                                layer_idx = int(hook_name.split(".")[1])
                                if layer_idx in vectors:
                                    steering_vec = vectors[layer_idx].to(local_model.cfg.device)
                            except:
                                pass
                        except:
                            print(f"Warning: Could not load steering vector {sv_id}")
                    
                    # Fallback to behavior-computed residual vector
                    if steering_vec is None and "steering_vector" in behavior:
                        steering_vec = torch.tensor(behavior["steering_vector"]).to(local_model.cfg.device)

                    # Define Hook
                    def sae_hook(activations, hook, sae=sae, baseline=sae_baseline, s_vec=steering_vec):
                        # activations: [batch, pos, d_model]
                        # sae.encode requires input [..., d_model]
                        
                        # 1. Encode
                        latents = sae.encode(activations) 
                        
                        # 2. Similarity (use last token for generation steering?)
                        # But generate() passes the whole sequence? 
                        # TransformerLens generate usually caches past KV, so activations 
                        # might just be the new token?
                        # Actually, run_with_hooks in generate context works on chunks.
                        # We calculate similarity on the last position
                        current_latent = latents[:, -1, :] 
                        
                        # Cosine Similarity
                        sim = torch.nn.functional.cosine_similarity(current_latent, baseline, dim=-1)
                        
                        # Remap [-1, 1] -> [0, 1]
                        score = (sim + 1) / 2
                        
                        # 3. Auto-correct
                        if score.item() > 0.5:
                            # print(f"SAE Triggered ({score.item():.2f}). Correcting...")
                            if s_vec is not None:
                                # Subtract the steering vector (assuming it represents the unwanted behavior)
                                # We broadcast s_vec to batch logic if needed
                                activations[:, -1, :] -= s_vec
                                
                        return activations

                    fwd_hooks.append((hook_name, sae_hook))

                # Generate with hooks
                output = local_model.generate(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    fwd_hooks=fwd_hooks
                )
                return output if isinstance(output, str) else local_model.to_string(output)[0]
            raise
