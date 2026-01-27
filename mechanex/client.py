import requests
from typing import Optional, List

from .errors import MechanexError
from .attribution import AttributionModule
from .steering import SteeringModule
from .raag import RAAGModule
from .generation import GenerationModule
from .model import ModelModule
from .base import _BaseModule
from .sae import SAEModule

class Mechanex:
    """
    A client for interacting with the Axionic API.
    """
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.model_name: Optional[str] = None
        self.num_layers: Optional[int] = None
        self.api_key = None

        # Initialize API modules
        self.attribution = AttributionModule(self)
        self.steering = SteeringModule(self)
        self.raag = RAAGModule(self)
        self.generation = GenerationModule(self)
        self.model = ModelModule(self)
        self.sae = SAEModule(self)

    def _get_headers(self) -> dict:
        """Return headers including Authorization if api_key is set."""
        headers = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            raise MechanexError("Please provide an API key to use Mechanex")
        return headers

    def set_key(self, api_key):
        self.api_key = api_key

    def load_model(self, model_name: str) -> 'Mechanex':
        """
        Loads a model into the service, making it available for other operations.
        Corresponds to the /load endpoint.
        """
        try:
            response = requests.post(f"{self.base_url}/load", json={"model_name": model_name, "device": "cuda"}, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            self.model_name = data.get("model_name")
            self.num_layers = data.get("num_layers")
            return self
        except requests.exceptions.RequestException as e:
            from .errors import APIError, AuthenticationError, NotFoundError, ValidationError
            
            message = f"Failed to load model '{model_name}'"
            status_code = None
            details = None

            if e.response is not None:
                status_code = e.response.status_code
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict) and "detail" in error_data:
                        message = error_data["detail"]
                    else:
                        message = str(error_data)
                    details = error_data
                except Exception:
                    message = e.response.text or message

            if status_code == 401:
                raise AuthenticationError(f"Authentication failed: {message}", status_code, details) from e
            elif status_code == 404:
                raise NotFoundError(f"Model not found: {message}", status_code, details) from e
            elif status_code == 422:
                raise ValidationError(f"Invalid model request: {message}", status_code, details) from e
            else:
                raise APIError(message, status_code, details) from e

    @staticmethod
    def get_huggingface_models(host: str = "127.0.0.1", port: int = 8000) -> List[str]:
        """
        Fetches the list of available public models from Hugging Face.
        This is a static method and does not require a model to be loaded.
        """
        try:
            response = requests.get(f"{host}/models")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            from .errors import APIError
            message = f"Could not fetch Hugging Face models"
            if e.response is not None:
                try:
                    message = e.response.json().get("detail", message)
                except Exception:
                    message = e.response.text or message
            raise APIError(message, getattr(e.response, 'status_code', None)) from e
