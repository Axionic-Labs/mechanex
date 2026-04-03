from typing import Optional

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

    def _handle_error(self, e: requests.exceptions.RequestException):
        """Internal helper to parse requests errors and raise appropriate MechanexError."""
        from .errors import APIError, AuthenticationError, NotFoundError, ValidationError

        message: str = "Request failed"
        status_code: Optional[int] = None
        details: Optional[dict] = None

        if e.response is not None:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                # Handle FastAPI detail format
                if isinstance(error_data, dict) and "detail" in error_data:
                    message = error_data["detail"]
                else:
                    message = str(error_data)
                details = error_data
            except Exception:
                message = e.response.text or str(e)

        if status_code == 401:
            raise AuthenticationError(f"Authentication failed: {message}", status_code, details) from e
        elif status_code == 402:
            from .errors import InsufficientCreditsError
            raise InsufficientCreditsError(
                f"Insufficient credits: {message}. Run `mechanex topup` to add credits.",
                status_code, details,
            ) from e
        elif status_code == 404:
            raise NotFoundError(f"Resource not found: {message}", status_code, details) from e
        elif status_code == 422:
            raise ValidationError(f"Validation error: {message}", status_code, details) from e
        else:
            raise APIError(message, status_code, details) from e

    def _post(self, endpoint: str, data: dict) -> dict:
        """Performs a POST request with Authorization and handles errors."""
        return self._client._post(endpoint, data)

    def _post_sse(self, endpoint: str, data: dict) -> dict:
        """Performs a POST request against an SSE endpoint and returns parsed JSON event payload."""
        return self._client._post_sse(endpoint, data)

    def _get(self, endpoint: str) -> dict:
        """Performs a GET request with Authorization and handles errors."""
        return self._client._get(endpoint)
