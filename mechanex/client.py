import requests
from typing import Optional, List

from .errors import MechanexError
from .attribution import AttributionModule
from .steering import SteeringModule
from .raag import RAAGModule
from .generation import GenerationModule
from .model import ModelModule
from .base import _BaseModule

class Mechanex:
    """
    A client for interacting with the Axionic API.
    """
    def __init__(self, base_url: str = "https://mechanex-waitlist-api-926733027827.us-central1.run.app/api"):
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
            response = requests.post(f"{self.base_url}/load", json={"model_name": model_name}, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            print(data)
            self.model_name = data.get("model_name")
            self.num_layers = data.get("num_layers")
            return self
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to load model '{model_name}': {e}"
            if e.response is not None:
                error_message += f" | Server response: {e.response.text}"
            raise MechanexError(error_message) from e

    def require_model_loaded(self):
        """Raises an error if a model hasn't been loaded."""
        if not self.model_name:
            raise MechanexError("No model loaded. Please call client.load_model('your-model') first.")

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
            raise MechanexError(f"Could not fetch Hugging Face models: {e}") from e
