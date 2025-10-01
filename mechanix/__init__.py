"""
Mechanix: A Python client for the Axionic API.
"""

from .client import Mechanix
from .errors import MechanixError

# Create the instance
_mx = Mechanix()

import sys
sys.modules[__name__] = _mx
