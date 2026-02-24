"""Schema parsing module for SeedGen.

Supports multiple input formats:
- JSON Schema (JSONSchemaParser)
- OpenAPI 3.0+ (OpenAPIParser)
- Natural Language (NaturalLanguageParser)

Ported from ARA module for multi-format schema support.
"""

from .canonical import CanonicalSchema, ParameterSpec, Constraint, Example
from .base import SchemaParser, ValidationResult
from .json_schema import JSONSchemaParser
from .openapi import OpenAPIParser
from .natural_language import NaturalLanguageParser

__all__ = [
    'CanonicalSchema',
    'ParameterSpec',
    'Constraint',
    'Example',
    'SchemaParser',
    'ValidationResult',
    'JSONSchemaParser',
    'OpenAPIParser',
    'NaturalLanguageParser',
    'parse_schema',
]


def parse_schema(
    input_schema,
    format_hint=None,
    context=None,
    teacher_client=None
) -> CanonicalSchema:
    """
    Parse a tool schema from any supported format.
    
    This is the main entry point for parsing schemas. It auto-detects
    the format and uses the appropriate parser.
    
    Args:
        input_schema: Schema in any format (dict, str, or type)
        format_hint: Optional SchemaFormat to skip auto-detection
        context: Optional context for NL parsing
        teacher_client: Optional teacher client for LLM-based NL parsing
        
    Returns:
        CanonicalSchema: Normalized schema representation
    """
    from ..models.enums import SchemaFormat
    
    # Use a base parser to detect format
    base = JSONSchemaParser()  # Any parser works for detection
    detected_format = format_hint or base.detect_format(input_schema)
    
    if detected_format == SchemaFormat.JSON_SCHEMA:
        parser = JSONSchemaParser()
    elif detected_format == SchemaFormat.OPENAPI:
        parser = OpenAPIParser()
    elif detected_format == SchemaFormat.NATURAL_LANGUAGE:
        parser = NaturalLanguageParser(teacher_client=teacher_client)
    else:
        # Fall back to JSON Schema parser
        parser = JSONSchemaParser()
    
    return parser.parse(input_schema, format_hint=detected_format, context=context)
