from typing import List, Optional
from .base import _BaseModule

class AttributionModule(_BaseModule):
    """Module for attribution patching APIs."""
    def compute_scores(self, clean_prompt: str, corrupted_prompt: str, target_module_paths: Optional[List[str]] = None) -> dict:
        """
        Computes attribution scores by patching the model.
        Corresponds to the /attribution-patching/scores endpoint.
        """
        if not self._client.model_name and not self._client.local_model:
            from .errors import MechanexError
            raise MechanexError("No model loaded. Call mx.load_model() or mx.set_local_model() first.")

        return self._post("/attribution-patching/scores", {
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "target_module_paths": target_module_paths or []
        })
