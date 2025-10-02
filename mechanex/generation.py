
from typing import Optional
from .base import _BaseModule

class GenerationModule(_BaseModule):
    def generate(self, prompt: str, max_tokens: int = 128, sampling_method: Optional[str] = "top-k", steering_strength=0) -> str:
        """
        Runs a standard generation
        """
        response = self._post("/generate", {
            "prompt": prompt,
            "sampling_method": sampling_method,
            "max_tokens": max_tokens,
            "use_steering": steering_strength
        })
        return response.get("output", "")
