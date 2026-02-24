"""Enum definitions for ARA module."""

from enum import Enum


class SchemaFormat(Enum):
    """Supported input schema formats."""
    JSON_SCHEMA = "json_schema"
    OPENAPI = "openapi"
    NATURAL_LANGUAGE = "natural_language"
    FUNCTION_SIGNATURE = "function_signature"
    PYDANTIC = "pydantic"
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Task types with different reward weight profiles."""
    JSON_EXTRACTION = "json_extraction"
    TOOL_ROUTING = "tool_routing"
    SQL_GENERATION = "sql_generation"
    API_CALLING = "api_calling"
    GENERAL = "general"


class TeacherProvider(Enum):
    """Supported Teacher model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


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
