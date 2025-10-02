import requests
from .errors import MechanexError

class _BaseModule:
    """A base class for API modules to handle requests, errors, and authentication."""

    def __init__(self, client):
        """
        Initialize the module.

        :param client: The main client instance.
        :param api_key: Optional API key for Authorization.
        """
        self._client = client

    def _post(self, endpoint: str, data: dict) -> dict:
        """Performs a POST request with Authorization and handles errors."""
        self._client.require_model_loaded()
        try:
            response = requests.post(
                f"{self._client.base_url}{endpoint}",
                json=data,
                headers=self._client._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"API request to {endpoint} failed: {e}"
            if e.response is not None:
                error_message += f" | Server response: {e.response.text}"
            raise MechanexError(error_message) from e

    def _get(self, endpoint: str) -> dict:
        """Performs a GET request with Authorization and handles errors."""
        self._client.require_model_loaded()
        try:
            response = requests.get(
                f"{self._client.base_url}{endpoint}",
                headers=self._client._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"API request to {endpoint} failed: {e}"
            if e.response is not None:
                error_message += f" | Server response: {e.response.text}"
            raise MechanexError(error_message) from e
