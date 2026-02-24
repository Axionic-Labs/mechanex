"""JSON Schema parser for ARA."""

from typing import Union, Optional, Dict, Any, List
import json

from .base import SchemaParser
from .canonical import CanonicalSchema, ParameterSpec, Example
from ..types.enums import SchemaFormat, ParameterType


class JSONSchemaParser(SchemaParser):
    """Parser for JSON Schema format."""

    TYPE_MAPPING = {
        'string': ParameterType.STRING,
        'integer': ParameterType.INTEGER,
        'number': ParameterType.NUMBER,
        'boolean': ParameterType.BOOLEAN,
        'array': ParameterType.ARRAY,
        'object': ParameterType.OBJECT,
        'null': ParameterType.NULL,
    }

    def parse(
        self,
        input_schema: Union[str, Dict],
        format_hint: Optional[SchemaFormat] = None,
        context: Optional[str] = None
    ) -> CanonicalSchema:
        """Parse JSON Schema into canonical form."""
        # Parse string if needed
        if isinstance(input_schema, str):
            schema = json.loads(input_schema)
        else:
            schema = input_schema

        # Extract tool name
        name = schema.get('title') or schema.get('name') or schema.get('$id', 'unknown_tool')
        name = self.normalize_name(name)

        # Parse parameters
        parameters = []
        required_params = []

        # Handle nested parameters structure (common in tool schemas)
        if 'parameters' in schema and isinstance(schema['parameters'], dict):
            param_schema = schema['parameters']
            properties = param_schema.get('properties', {})
            required_params = param_schema.get('required', [])
        else:
            properties = schema.get('properties', {})
            required_params = schema.get('required', [])

        for prop_name, prop_spec in properties.items():
            param = self._parse_parameter(prop_name, prop_spec, prop_name in required_params)
            parameters.append(param)

        # Parse examples if present
        examples = []
        if 'examples' in schema:
            for ex in schema['examples']:
                if isinstance(ex, dict):
                    examples.append(Example(
                        prompt=ex.get('prompt', ''),
                        expected_arguments=ex.get('arguments', ex.get('expected_arguments', {})),
                        reasoning=ex.get('reasoning')
                    ))

        return CanonicalSchema(
            name=name,
            description=schema.get('description', ''),
            parameters=parameters,
            required_parameters=required_params,
            examples=examples,
            source_format=SchemaFormat.JSON_SCHEMA,
            original_schema=schema
        )

    def _parse_parameter(
        self,
        name: str,
        spec: Dict[str, Any],
        required: bool
    ) -> ParameterSpec:
        """Parse a single parameter specification."""
        param_type = self._map_type(spec.get('type', 'string'))

        return ParameterSpec(
            name=name,
            type=param_type,
            description=spec.get('description', ''),
            required=required,
            default=spec.get('default'),
            enum=spec.get('enum'),
            minimum=spec.get('minimum'),
            maximum=spec.get('maximum'),
            min_length=spec.get('minLength'),
            max_length=spec.get('maxLength'),
            pattern=spec.get('pattern')
        )

    def _map_type(self, json_type: Union[str, List[str]]) -> ParameterType:
        """Map JSON Schema type to ParameterType."""
        if isinstance(json_type, list):
            # Handle union types - take first non-null type
            for t in json_type:
                if t != 'null':
                    return self.TYPE_MAPPING.get(t, ParameterType.STRING)
            return ParameterType.NULL

        return self.TYPE_MAPPING.get(json_type, ParameterType.STRING)
