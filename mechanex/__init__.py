"""
Mechanex: A Python client for the Axionic API.
"""

from .client import Mechanex
from .errors import MechanexError

# Create the instance
_mx = Mechanex()

import sys
sys.modules[__name__] = _mx
