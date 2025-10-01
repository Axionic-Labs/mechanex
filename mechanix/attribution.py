from typing import List, Optional
from .base import _BaseModule

class AttributionModule(_BaseModule):
    """Module for attribution patching APIs."""
    def compute_scores(self, clean_prompt: str, corrupted_prompt: str, target_module_paths: Optional[List[str]] = None) -> dict:
        """
        Computes attribution scores by patching the model.
        Corresponds to the /attribution-patching/scores endpoint.
        """
        return self._post("/attribution-patching/scores", {
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "target_module_paths": target_module_paths or []
        })
