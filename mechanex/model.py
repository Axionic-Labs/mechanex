import warnings
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

    def graph(self) -> List[Dict[str, Any]]:
        """Alias for get_graph()."""
        return self.get_graph()

    def get_paths(self) -> List[str]:
        """
        Retrieves all available layer paths in the model.
        Derives paths from the /graph endpoint since /paths is not available.
        """
        warnings.warn(
            "get_paths() is deprecated. Use get_graph() to inspect model structure.",
            DeprecationWarning,
            stacklevel=2,
        )
        graph = self.get_graph()
        return [node.get("name", "") for node in graph if isinstance(node, dict) and "name" in node]
