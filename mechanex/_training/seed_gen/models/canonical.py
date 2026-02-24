"""Canonical schema representation for SeedGen.

Ported from ARA module. Provides unified internal representation
for tool schemas from any input format.
"""

from typing import List, Optional, Dict, Any
import hashlib
import json
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import SchemaFormat, ParameterType, ConstraintType


class ParameterSpec(BaseModel):
    """Specification for a single parameter."""

    name: str = Field(..., min_length=1)
    type: ParameterType
    description: str = ""
    required: bool = False
    default: Optional[Any] = None

    # Type-specific constraints
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = Field(default=None, ge=0)
    max_length: Optional[int] = Field(default=None, ge=0)
    pattern: Optional[str] = None
    items: Optional['ParameterSpec'] = None  # For arrays
    properties: Optional[Dict[str, 'ParameterSpec']] = None  # For objects

    @field_validator('pattern')
    @classmethod
    def validate_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate that pattern is a valid regex."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v

    @model_validator(mode='after')
    def validate_length_constraints(self) -> 'ParameterSpec':
        """Validate that min_length <= max_length if both are set."""
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(f"min_length ({self.min_length}) cannot be greater than max_length ({self.max_length})")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'required': self.required,
        }
        if self.default is not None:
            result['default'] = self.default
        if self.enum is not None:
            result['enum'] = self.enum
        if self.minimum is not None:
            result['minimum'] = self.minimum
        if self.maximum is not None:
            result['maximum'] = self.maximum
        if self.min_length is not None:
            result['minLength'] = self.min_length
        if self.max_length is not None:
            result['maxLength'] = self.max_length
        if self.pattern is not None:
            result['pattern'] = self.pattern
        return result

    model_config = {'use_enum_values': False}


class Constraint(BaseModel):
    """Business logic constraint."""

    type: ConstraintType
    description: str
    validation_code: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)


class Example(BaseModel):
    """Example input/output pair."""

    prompt: str = Field(..., min_length=1)
    expected_arguments: Dict[str, Any]
    reasoning: Optional[str] = None
    is_edge_case: bool = False


class CanonicalSchema(BaseModel):
    """Normalized tool schema representation."""

    # Identity
    name: str = Field(..., min_length=1)
    version: str = "1.0.0"

    # Description
    description: str = ""
    long_description: Optional[str] = None

    # Parameters
    parameters: List[ParameterSpec] = Field(default_factory=list)
    required_parameters: List[str] = Field(default_factory=list)

    # Constraints
    constraints: List[Constraint] = Field(default_factory=list)

    # Examples (for few-shot prompting)
    examples: List[Example] = Field(default_factory=list)

    # Metadata
    source_format: SchemaFormat = SchemaFormat.UNKNOWN
    original_schema: Optional[Dict[str, Any]] = None
    
    # SeedGen-specific metadata
    complexity: str = "medium"  # simple, medium, complex
    category: str = "general"

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is in snake_case format."""
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            # Auto-convert to snake_case instead of raising
            v = re.sub(r'[\s\-]+', '_', v)
            v = re.sub(r'[^a-zA-Z0-9_]', '', v)
            v = re.sub(r'([a-z])([A-Z])', r'\1_\2', v)
            v = v.lower()
        return v

    @model_validator(mode='after')
    def validate_required_params(self) -> 'CanonicalSchema':
        """Validate that required_parameters references valid parameters."""
        param_names = {p.name for p in self.parameters}
        for req in self.required_parameters:
            if req not in param_names:
                raise ValueError(f"Required parameter '{req}' not found in parameters")
        return self

    model_config = {'use_enum_values': False}

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        properties = {}
        for param in self.parameters:
            prop = {
                'type': param.type.value,
                'description': param.description,
            }
            if param.enum is not None:
                prop['enum'] = param.enum
            if param.minimum is not None:
                prop['minimum'] = param.minimum
            if param.maximum is not None:
                prop['maximum'] = param.maximum
            if param.min_length is not None:
                prop['minLength'] = param.min_length
            if param.max_length is not None:
                prop['maxLength'] = param.max_length
            if param.pattern is not None:
                prop['pattern'] = param.pattern
            if param.default is not None:
                prop['default'] = param.default
            properties[param.name] = prop

        return {
            'title': self.name,
            'description': self.description,
            'type': 'object',
            'properties': properties,
            'required': self.required_parameters,
        }

    def to_prompt_string(self) -> str:
        """Format for inclusion in prompts."""
        lines = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
            "",
            "Parameters:"
        ]
        for param in self.parameters:
            req_str = " (required)" if param.required else " (optional)"
            lines.append(f"  - {param.name}: {param.type.value}{req_str}")
            if param.description:
                lines.append(f"    Description: {param.description}")
            if param.enum:
                lines.append(f"    Allowed values: {param.enum}")
            if param.pattern:
                lines.append(f"    Pattern: {param.pattern}")

        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for ex in self.examples:
                lines.append(f"  Prompt: {ex.prompt}")
                lines.append(f"  Arguments: {json.dumps(ex.expected_arguments)}")

        return '\n'.join(lines)

    def hash(self) -> str:
        """Compute deterministic hash for caching."""
        content = json.dumps(self.to_json_schema(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
