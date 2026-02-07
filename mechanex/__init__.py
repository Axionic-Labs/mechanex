"""
Mechanex: A Python client for the Axionic API.
"""

from .client import Mechanex
from .errors import MechanexError

# Create the singleton instance
_mx = Mechanex()

# Expose instance methods and modules at the package level for convenience
client = _mx
signup = _mx.signup
login = _mx.login
list_api_keys = _mx.list_api_keys
get_balance = _mx.get_balance
serve = _mx.serve
set_key = _mx.set_key
set_model = _mx.set_model
set_local_model = _mx.set_local_model
load_model = _mx.load_model
load = _mx.load
unload = _mx.unload
get_huggingface_models = _mx.get_huggingface_models

# Expose sub-modules
steering = _mx.steering
generation = _mx.generation
model = _mx.model
sae = _mx.sae
raag = _mx.raag
attribution = _mx.attribution
