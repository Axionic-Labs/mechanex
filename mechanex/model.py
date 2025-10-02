from typing import List, Dict, Any
from .base import _BaseModule

class ModelModule(_BaseModule):
    """Module for inspecting the model structure."""
    def get_graph(self) -> List[Dict[str, Any]]:
        """
        Retrieves the model's computation graph.
        Corresponds to the /graph endpoint.
        """
        response = self._get("/graph")
        return response.get("graph", [])

    def get_paths(self) -> List[str]:
        """
        Retrieves all available layer paths in the model.
        Corresponds to the /paths endpoint.
        """
        response = self._get("/paths")
        return response.get("paths", [])
