"""Enum definitions for SeedGen schema parsing.

Ported from ARA module for multi-format schema support.
"""

from enum import Enum


class SchemaFormat(Enum):
    """Supported input schema formats."""
    JSON_SCHEMA = "json_schema"
    OPENAPI = "openapi"
    NATURAL_LANGUAGE = "natural_language"
    FUNCTION_SIGNATURE = "function_signature"
    PYDANTIC = "pydantic"
    UNKNOWN = "unknown"


class ParameterType(Enum):
    """JSON Schema parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class ConstraintType(Enum):
    """Types of parameter constraints."""
    VALUE_RANGE = "value_range"
    DEPENDENCY = "dependency"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"
