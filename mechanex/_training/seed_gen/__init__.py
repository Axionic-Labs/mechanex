"""
Mechanex Seed Generator POC

This module implements the seed generation pipeline for the Mechanex RL training system.
It uses a teacher model (Gemini) to generate diverse, realistic tool-calling prompts
for any domain.
"""

__version__ = "0.1.0"

# Export parsing infrastructure for multi-format schema support
from .parsing import (
    parse_schema,
    SchemaParser,
    JSONSchemaParser,
    OpenAPIParser,
    NaturalLanguageParser,
    CanonicalSchema,
    ParameterSpec,
    ValidationResult,
)

# Export enums
from .models.enums import SchemaFormat, ParameterType, ConstraintType

__all__ = [
    # Version
    '__version__',
    # Parsing
    'parse_schema',
    'SchemaParser',
    'JSONSchemaParser',
    'OpenAPIParser',
    'NaturalLanguageParser',
    'CanonicalSchema',
    'ParameterSpec',
    'ValidationResult',
    # Enums
    'SchemaFormat',
    'ParameterType',
    'ConstraintType',
]

