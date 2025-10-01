import requests
from .errors import MechanixError

class _BaseModule:
    """A base class for API modules to handle requests and errors."""
    def __init__(self, client):
        self._client = client

    def _post(self, endpoint: str, data: dict) -> dict:
        """Performs a POST request and handles errors."""
        self._client.require_model_loaded()
        try:
            response = requests.post(f"{self._client.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"API request to {endpoint} failed: {e}"
            if e.response is not None:
                error_message += f" | Server response: {e.response.text}"
            raise MechanixError(error_message) from e

    def _get(self, endpoint: str) -> dict:
        """Performs a GET request and handles errors."""
        self._client.require_model_loaded()
        try:
            response = requests.get(f"{self._client.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"API request to {endpoint} failed: {e}"
            if e.response is not None:
                error_message += f" | Server response: {e.response.text}"
            raise MechanixError(error_message) from e
