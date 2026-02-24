"""Reward result dataclass."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class RewardResult:
    """Result from reward evaluation."""

    score: float
    breakdown: Dict[str, float]
    passed: bool
    details: Optional[Dict[str, Any]] = None
