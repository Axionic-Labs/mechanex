"""Result models for ARA module using Pydantic."""

from typing import Dict, Optional, Any, List
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class RewardResult(BaseModel):
    """Result from reward function evaluation."""

    score: float = Field(..., ge=0.0, le=1.0, description="Composite score")
    breakdown: Dict[str, float] = Field(..., description="Component scores")
    passed: bool = Field(..., description="Above threshold?")

    # Detailed info
    format_details: Optional[Dict[str, Any]] = None
    grounding_details: Optional[Dict[str, Any]] = None
    type_details: Optional[Dict[str, Any]] = None
    reasoning_details: Optional[Dict[str, Any]] = None

    # Debugging
    raw_response: Optional[str] = None
    extracted_tool_call: Optional[Dict[str, Any]] = None

    @field_validator('breakdown')
    @classmethod
    def validate_breakdown(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that breakdown scores are in valid range."""
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score for '{key}' must be between 0.0 and 1.0, got {score}")
        return v


class ValidationResult(BaseModel):
    """Result from validation operations."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Result from sandbox execution."""

    success: bool
    reward_class: Optional[Any] = None  # type object
    reward_instance: Optional[Any] = None
    test_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = Field(default=0.0, ge=0.0)

    model_config = {'arbitrary_types_allowed': True}


class CompilationMetadata(BaseModel):
    """Metadata about reward compilation."""

    teacher_model: str
    compilation_time: datetime
    prompt_hash: int
    code_hash: Optional[str] = None
    validation_passed: bool = False


class GeneratedReward(BaseModel):
    """Generated reward function before validation."""

    tool_schema: Any = Field(..., description="CanonicalSchema instance", alias="schema")
    code: str = Field(..., min_length=1)
    weights: Dict[str, float]
    metadata: CompilationMetadata

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights are non-negative."""
        for key, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for '{key}' must be non-negative, got {weight}")
        return v

    model_config = {'arbitrary_types_allowed': True, 'populate_by_name': True}
