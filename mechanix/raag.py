from typing import List, Dict
from .base import _BaseModule

class RAAGModule(_BaseModule):
    """Module for Retrieval-Augmented Answer Generation APIs."""
    def generate(self, qa_entries: List[dict], docs: List[dict]) -> dict:
        """
        Performs Retrieval-Augmented Answer Generation.
        Corresponds to the /raag/generate endpoint.
        """
        return self._post("/raag/generate", {"qa_entries": qa_entries, "docs": docs})
