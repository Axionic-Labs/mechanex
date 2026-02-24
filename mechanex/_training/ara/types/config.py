"""Configuration models for ARA module using Pydantic."""

from typing import Dict, Optional, Set
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .enums import TeacherProvider, TaskType


class ARAConfig(BaseModel):
    """Configuration for ARA module."""

    # Teacher settings
    teacher_provider: TeacherProvider = TeacherProvider.GOOGLE
    teacher_model: str = "gemini-2.0-flash"
    teacher_api_key: Optional[str] = None

    # Compilation settings
    default_task_type: TaskType = TaskType.GENERAL
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Sandbox settings
    max_execution_time: float = Field(default=5.0, gt=0.0)
    max_memory_mb: int = Field(default=256, gt=0)

    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = ".ara_cache"
    max_cache_size: int = Field(default=100, gt=0)

    # Weights
    default_weights: Dict[str, float] = Field(default_factory=lambda: {
        'format': 0.2,
        'grounding': 0.3,
        'type_valid': 0.3,
        'reasoning': 0.2
    })

    @field_validator('default_weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    model_config = {'use_enum_values': False}


class SandboxConfig(BaseModel):
    """Configuration for reward sandbox."""

    max_execution_time: float = Field(default=5.0, gt=0.0)
    max_memory_mb: int = Field(default=256, gt=0)
    max_output_size: int = Field(default=10000, gt=0)
    allowed_imports: Set[str] = Field(default_factory=lambda: {
        're', 'json', 'typing', 'dataclasses', 'pydantic',
        'datetime', 'enum', 'collections', 'math'
    })
    blocked_builtins: Set[str] = Field(default_factory=lambda: {
        'eval', 'exec', 'compile', 'open', 'input',
        'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr',
        'breakpoint', 'exit', 'quit'
    })

    model_config = {'frozen': False}


class CompilerConfig(BaseModel):
    """Configuration for reward compiler."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8000, gt=0)
    include_examples: bool = True
    retry_on_failure: int = Field(default=2, ge=0)


# Task-specific reward weight configurations
REWARD_WEIGHTS: Dict[TaskType, Dict[str, float]] = {
    TaskType.JSON_EXTRACTION: {
        'format': 0.15,
        'grounding': 0.25,
        'type_valid': 0.45,
        'reasoning': 0.15
    },
    TaskType.TOOL_ROUTING: {
        'format': 0.15,
        'grounding': 0.35,
        'type_valid': 0.20,
        'reasoning': 0.30
    },
    TaskType.SQL_GENERATION: {
        'format': 0.10,
        'grounding': 0.40,
        'type_valid': 0.30,
        'reasoning': 0.20
    },
    TaskType.API_CALLING: {
        'format': 0.20,
        'grounding': 0.30,
        'type_valid': 0.35,
        'reasoning': 0.15
    },
    TaskType.GENERAL: {
        'format': 0.20,
        'grounding': 0.30,
        'type_valid': 0.30,
        'reasoning': 0.20
    }
}
