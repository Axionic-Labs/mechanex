import requests
import torch
import numpy as np
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
    def __init__(self, base_url: str = "http://localhost:3000", local_model=None):
        self.base_url = base_url
        self.local_model = local_model
        self._local_vectors = {}
        self.model_name: Optional[str] = None
        self.num_layers: Optional[int] = None
        self.api_key = None
        self._local_vectors = {}
        self._local_behaviors = {}

        # Try to load API key from config file
        try:
            import json
            from pathlib import Path
            config_path = Path.home() / ".mechanex" / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "api_key" in config:
                        self.api_key = config["api_key"]
        except Exception:
            pass # Fail silently if config cannot be read

        # Initialize API modules
        self.attribution = AttributionModule(self)
        self.steering = SteeringModule(self)
        self.raag = RAAGModule(self)
        self.generation = GenerationModule(self)
        self.model = ModelModule(self)
        self.sae = SAEModule(self)

    def signup(self, email, password):
        """Register a new user."""
        try:
            # Note: signup doesn't require auth headers usually
            response = requests.post(f"{self.base_url}/auth/signup", json={"email": email, "password": password})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise self._handle_request_error(e, "Signup failed")

    def login(self, email, password):
        """Authenticate and set API key."""
        try:
            # Login doesn't require auth headers
            response = requests.post(f"{self.base_url}/auth/login", json={"email": email, "password": password})
            response.raise_for_status()
            data = response.json()
            # Assuming the session contains an access_token which we use as the API key
            if "session" in data and "access_token" in data["session"]:
                self.api_key = data["session"]["access_token"]
            return data
        except requests.exceptions.RequestException as e:
            raise self._handle_request_error(e, "Login failed")

    def list_api_keys(self):
        """List API keys for the current user."""
        return self._get("/auth/api-keys")

    def create_api_key(self, name: str = "Default Key"):
        """Create a new API key for the current user."""
        return self._post("/auth/api-keys", {"name": name})

    def whoami(self):
        """Get current user information."""
        return self._get("/auth/whoami")

    def serve(self, model=None, host="0.0.0.0", port=8000, use_vllm=False):
        """Turn the model into an OpenAI compatible endpoint."""
        from .serving import run_server
        run_server(self, model, host, port, use_vllm=use_vllm)

    def _get_headers(self) -> dict:
        """Return headers including Authorization if api_key is set."""
        headers = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            raise MechanexError("API key not found. Please provide an API key or run 'mechanex login' if using the CLI.")
        return headers

    def _get(self, endpoint: str) -> dict:
        """Internal helper for GET requests with auth."""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise self._handle_request_error(e, f"GET {endpoint} failed")

    def _post(self, endpoint: str, data: dict = None) -> dict:
        """Internal helper for POST requests with auth."""
        try:
            response = requests.post(f"{self.base_url}{endpoint}", json=data, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise self._handle_request_error(e, f"POST {endpoint} failed")

    def set_key(self, api_key):
        self.api_key = api_key

    def set_local_model(self, model):
        """Set a local model (e.g. TransformerLens) for local steering fallback."""
        self.local_model = model
        return self

    def load(self, model_name: str, **kwargs) -> 'Mechanex':
        """
        Alias for load_model.
        """
        return self.load_model(model_name, **kwargs)

    def unload(self) -> 'Mechanex':
        """
        Alias for unload_model.
        """
        return self.unload_model()

    def unload_model(self) -> 'Mechanex':
        """
        Unloads the local model and clears associated metadata.
        """
        if self.local_model is not None:
            model_name = getattr(self, "model_name", "model")
            print(f"Unloading {model_name}...")
            # Explicitly move to CPU and clear cache if possible before deletion
            if hasattr(self.local_model, "to"):
                self.local_model.to("cpu")
            self.local_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.model_name = None
        self.num_layers = None
        self._local_vectors = {}
        self._local_behaviors = {}
        import gc
        gc.collect()
        return self

    def load_model(self, model_name: str, **kwargs) -> 'Mechanex':
        """
        Loads a model locally using TransformerLens and automatically configures SAE settings.
        """
        if not self.api_key:
            raise MechanexError("API key required. Call mx.set_key() first.")
        from transformer_lens import HookedTransformer
        print(f"Loading {model_name} locally...")
        
        device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.local_model = HookedTransformer.from_pretrained(model_name, device=device, **kwargs)
        self.model_name = model_name
        self.num_layers = self.local_model.cfg.n_layers

        # Automatically set SAE release based on model name
        sae_mapping = {
            "gpt2-small": "gpt2-small-res-jb",
            "gpt2-medium": "gpt2-medium-res-jb",
            "pythia-70m-deduped": "pythia-70m-deduped-res-jb",
            "pythia-160m-deduped": "pythia-160m-deduped-res-jb",
            "gemma-2-2b": "gemma-scope-2b-pt-res",
            "google/gemma-2-2b": "gemma-scope-2b-pt-res",
            "google/gemma-2-2b-it": "gemma-scope-2b-it-res",
            "gemma-2-9b": "gemma-scope-9b-pt-res",
            "google/gemma-2-9b": "gemma-scope-9b-pt-res",
            "google/gemma-2-9b-it": "gemma-scope-9b-it-res",
            "google/gemma-3-4b-pt": "gemma-scope-2-4b-pt-res",
            "google/gemma-3-4b-it": "gemma-scope-2-4b-it-res",
            "google/gemma-3-12b-pt": "gemma-scope-2-12b-pt-res",
            "google/gemma-3-12b-it": "gemma-scope-2-12b-it-res",
            "google/gemma-3-27b-pt": "gemma-scope-2-27b-pt-res",
            "google/gemma-3-27b-it": "gemma-scope-2-27b-it-res",
        }
        
        # Default to a best-guess if not in mapping
        release = sae_mapping.get(model_name)
        if not release:
            # Heuristic for Gemma models not explicitly in mapping
            if "gemma-3" in model_name:
                size = "4b" if "4b" in model_name else ("12b" if "12b" in model_name else "27b")
                mode = "it" if "-it" in model_name else "pt"
                release = f"gemma-scope-2-{size}-{mode}-res"
            elif "gemma-2" in model_name:
                size = "2b" if "2b" in model_name else ("9b" if "9b" in model_name else "27b")
                mode = "it" if "-it" in model_name else "pt"
                release = f"gemma-scope-{size}-{mode}-res"
            else:
                release = f"{model_name}-res-jb"
                
        self.sae.sae_release = release
        print(f"SAE release automatically set to: {release}")
        
        return self

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
            message = "Could not fetch Hugging Face models"
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict) and "detail" in error_data:
                        message = error_data["detail"]
                    else:
                        message = str(error_data)
                except Exception:
                    message = e.response.text or message
            raise APIError(message, getattr(e.response, 'status_code', None)) from e

    def _handle_request_error(self, e: requests.exceptions.RequestException, default_message: str):
        """Internal helper to parse requests errors and raise appropriate MechanexError."""
        from .errors import APIError, AuthenticationError, NotFoundError, ValidationError
        
        message = default_message
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
            return AuthenticationError(f"Authentication failed: {message}", status_code, details)
        elif status_code == 404:
            return NotFoundError(f"Resource not found: {message}", status_code, details)
        elif status_code == 422:
            return ValidationError(f"Validation error: {message}", status_code, details)
        else:
            return APIError(f"{default_message}: {message}" if message != default_message else message, status_code, details)
