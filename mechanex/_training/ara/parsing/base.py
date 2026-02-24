"""Base schema parser for ARA."""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import json
import re

from .canonical import CanonicalSchema
from ..types.enums import SchemaFormat
from ..types.results import ValidationResult


class SchemaParser(ABC):
    """
    Base class for schema parsers.

    Parses tool definitions from multiple formats into canonical form.
    """

    @abstractmethod
    def parse(
        self,
        input_schema: Union[str, Dict, type],
        format_hint: Optional[SchemaFormat] = None,
        context: Optional[str] = None
    ) -> CanonicalSchema:
        """
        Parse input into canonical schema.

        Args:
            input_schema: Raw schema in any supported format
            format_hint: Optional format specification (auto-detected if None)
            context: Optional enterprise context for NL parsing

        Returns:
            CanonicalSchema: Normalized schema representation
        """
        pass

    def detect_format(self, input_schema: Union[str, Dict, type]) -> SchemaFormat:
        """Auto-detect the input format."""
        # If it's a type (class), assume Pydantic
        if isinstance(input_schema, type):
            return SchemaFormat.PYDANTIC

        # If it's a dict
        if isinstance(input_schema, dict):
            # Check for OpenAPI markers
            if 'openapi' in input_schema or 'paths' in input_schema:
                return SchemaFormat.OPENAPI
            # Check for JSON Schema markers
            if 'type' in input_schema or 'properties' in input_schema or '$schema' in input_schema:
                return SchemaFormat.JSON_SCHEMA
            return SchemaFormat.UNKNOWN

        # If it's a string
        if isinstance(input_schema, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(input_schema)
                return self.detect_format(parsed)
            except json.JSONDecodeError:
                pass

            # Check for function signature pattern
            if re.match(r'^def\s+\w+\s*\(', input_schema.strip()):
                return SchemaFormat.FUNCTION_SIGNATURE

            # Assume natural language
            return SchemaFormat.NATURAL_LANGUAGE

        return SchemaFormat.UNKNOWN

    def validate(self, schema: CanonicalSchema) -> ValidationResult:
        """Validate canonical schema for completeness."""
        errors = []
        warnings = []

        # Check required fields
        if not schema.name:
            errors.append("Schema name is required")
        elif not re.match(r'^[a-z][a-z0-9_]*$', schema.name):
            warnings.append(f"Schema name '{schema.name}' should be snake_case")

        if not schema.description:
            warnings.append("Schema description is empty")

        if not schema.parameters:
            warnings.append("Schema has no parameters defined")

        # Check parameters
        param_names = set()
        for param in schema.parameters:
            if param.name in param_names:
                errors.append(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)

            if param.required and param.name not in schema.required_parameters:
                warnings.append(f"Parameter '{param.name}' marked required but not in required_parameters list")

        # Check required_parameters references
        for req_param in schema.required_parameters:
            if req_param not in param_names:
                errors.append(f"Required parameter '{req_param}' not found in parameters")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a name to snake_case."""
        # Replace spaces and hyphens with underscores
        name = re.sub(r'[\s\-]+', '_', name)
        # Remove non-alphanumeric characters except underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        # Convert camelCase to snake_case
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        # Convert to lowercase
        return name.lower()
