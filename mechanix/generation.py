
from typing import Optional
from .base import _BaseModule

class GenerationModule(_BaseModule):
    def generate_steered(self, prompt: str, strength: float = 1.0, max_tokens: int = 128) -> str:
        """
        Runs a steered generation.
        Corresponds to the /steering/run endpoint.
        """
        response = self._post("/steering/run", {
            "prompt": prompt,
            "multiplier": strength,
            "max_tokens": max_tokens
        })
        return response.get("output", "")
    
    def generate(self, prompt: str, max_tokens: int = 128, sampling_method: Optional[str] = "top-k") -> str:
        """
        Runs a standard generation
        """
        response = self._post("/sampling/generate", {
            "prompt": prompt,
            "method": sampling_method,
            "max_new_tokens": max_tokens
        })
        return response.get("output", "")
