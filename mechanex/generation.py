from typing import Optional
from .base import _BaseModule

class GenerationModule(_BaseModule):
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 128, 
        sampling_method: str = "top-k", 
        top_k: int = 50,
        top_p: float = 0.9,
        steering_strength: float = 0, 
        steering_vector=None
    ) -> str:
        """
        Runs a standard generation. Falls back to local model if available.
        """
        if not self._client.api_key:
            from .errors import MechanexError
            raise MechanexError("API key required. Call mx.set_key() first.")

        try:
            # Handle remote request if possible
            if self._client.api_key:
                payload = {
                    "prompt": prompt,
                    "sampling_method": sampling_method,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "top_p": top_p,
                    "steering_vector_id": steering_vector if isinstance(steering_vector, str) else None,
                    "steering_strength": steering_strength
                }
                response = self._post("/generate", payload)
                return response.get("output", "")
        except Exception as e:
            # Only suppress error if we have a local model to fall back to
            if not getattr(self._client, 'local_model', None):
                if sampling_method == "ads":
                    from .errors import MechanexError
                    raise MechanexError("Add balance to your API key in order to use this feature.")
                raise e

        # Local Fallback
        local_model = getattr(self._client, 'local_model', None)
        if local_model is not None:
            # 1. Validate sampling method for local model
            supported_local_methods = ["top-k", "top-p", "greedy", None]
            if sampling_method == "ads":
                from .errors import MechanexError
                raise MechanexError("Add balance to your API key in order to use this feature.")
            
            if sampling_method not in supported_local_methods:
                from .errors import MechanexError
                raise MechanexError(f"Sampling method '{sampling_method}' is not supported for local models.")

            import torch
            from .utils.steering_opt import make_steering_hook_tflens
            
            # ... (rest of local logic remains the same)
            # Resolve steering vectors
            vectors = None
            if steering_vector is not None:
                if isinstance(steering_vector, str):
                    vectors = getattr(self._client, '_local_vectors', {}).get(steering_vector)
                elif isinstance(steering_vector, dict):
                    vectors = steering_vector
            
            # Apply steering if vectors are found
            fwd_hooks = []
            if vectors and steering_strength != 0:
                for layer, vec in vectors.items():
                    # Ensure vec is a tensor and on the right device
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.tensor(vec)
                    vec = vec.to(local_model.cfg.device)
                    
                    hook_fn = make_steering_hook_tflens(vec * steering_strength)
                    fwd_hooks.append((f"blocks.{layer}.hook_resid_pre", hook_fn))

            # Generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "verbose": False
            }
            if sampling_method == "top-k":
                gen_kwargs["top_k"] = top_k
            elif sampling_method == "top-p":
                gen_kwargs["top_p"] = top_p
            elif sampling_method == "greedy":
                gen_kwargs["top_k"] = 1 # transformer_lens greedy is top_k=1

            if fwd_hooks:
                with local_model.hooks(fwd_hooks=fwd_hooks):
                    output = local_model.generate(prompt, **gen_kwargs)
            else:
                output = local_model.generate(prompt, **gen_kwargs)
                
            return output if isinstance(output, str) else local_model.to_string(output)[0]
        
        from .errors import MechanexError
        raise MechanexError("No local model available for fallback.")
