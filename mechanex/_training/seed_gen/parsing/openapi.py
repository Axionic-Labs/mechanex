"""OpenAPI schema parser for SeedGen.

Ported from ARA module.
"""

from typing import Union, Optional, Dict, Any, List, Tuple
import json
import re

from .base import SchemaParser
from .canonical import CanonicalSchema, ParameterSpec, Example
from ..models.enums import SchemaFormat, ParameterType


class OpenAPIParser(SchemaParser):
    """Parser for OpenAPI 3.0+ specifications."""

    TYPE_MAPPING = {
        'string': ParameterType.STRING,
        'integer': ParameterType.INTEGER,
        'number': ParameterType.NUMBER,
        'boolean': ParameterType.BOOLEAN,
        'array': ParameterType.ARRAY,
        'object': ParameterType.OBJECT,
    }

    def parse(
        self,
        input_schema: Union[str, Dict],
        format_hint: Optional[SchemaFormat] = None,
        context: Optional[str] = None
    ) -> CanonicalSchema:
        """
        Parse OpenAPI specification into canonical form.

        Supports:
        - Full OpenAPI 3.0+ documents with paths
        - Individual operation objects
        - Component schemas
        """
        # Parse string if needed
        if isinstance(input_schema, str):
            schema = json.loads(input_schema)
        else:
            schema = input_schema

        # Determine what kind of OpenAPI input we have
        if 'openapi' in schema and 'paths' in schema:
            # Full OpenAPI document - extract first operation
            return self._parse_full_document(schema)
        elif 'operationId' in schema or 'requestBody' in schema:
            # Individual operation object
            return self._parse_operation(schema, schema.get('operationId', 'unknown_operation'))
        elif 'type' in schema and schema.get('type') == 'object':
            # Component schema (treat as request body)
            return self._parse_component_schema(schema)
        else:
            raise ValueError("Unrecognized OpenAPI structure")

    def _parse_full_document(self, spec: Dict) -> CanonicalSchema:
        """Parse a full OpenAPI document, extracting the first operation."""
        paths = spec.get('paths', {})
        components = spec.get('components', {}).get('schemas', {})

        # Find the first operation
        for path, path_item in paths.items():
            for method in ['post', 'put', 'patch', 'get', 'delete']:
                if method in path_item:
                    operation = path_item[method]
                    operation_id = operation.get('operationId', self._path_to_name(path, method))
                    return self._parse_operation(operation, operation_id, components)

        raise ValueError("No operations found in OpenAPI document")

    def _parse_operation(
        self,
        operation: Dict,
        operation_id: str,
        components: Optional[Dict] = None
    ) -> CanonicalSchema:
        """Parse an individual operation into canonical form."""
        components = components or {}
        parameters = []
        required_params = []

        # Parse path/query/header parameters
        for param in operation.get('parameters', []):
            param_spec = self._parse_parameter(param, components)
            parameters.append(param_spec)
            if param.get('required', False):
                required_params.append(param_spec.name)

        # Parse request body
        request_body = operation.get('requestBody', {})
        if request_body:
            body_params, body_required = self._parse_request_body(request_body, components)
            parameters.extend(body_params)
            required_params.extend(body_required)

        # Parse examples from request body or responses
        examples = self._extract_examples(operation, components)

        return CanonicalSchema(
            name=self.normalize_name(operation_id),
            description=operation.get('summary', operation.get('description', '')),
            long_description=operation.get('description') if 'summary' in operation else None,
            parameters=parameters,
            required_parameters=required_params,
            examples=examples,
            source_format=SchemaFormat.OPENAPI,
            original_schema=operation
        )

    def _parse_component_schema(self, schema: Dict) -> CanonicalSchema:
        """Parse a component schema as if it were a tool definition."""
        name = schema.get('title', schema.get('x-tool-name', 'unknown_tool'))
        parameters = []
        required_params = schema.get('required', [])

        for prop_name, prop_spec in schema.get('properties', {}).items():
            param = self._property_to_parameter(prop_name, prop_spec, prop_name in required_params)
            parameters.append(param)

        return CanonicalSchema(
            name=self.normalize_name(name),
            description=schema.get('description', ''),
            parameters=parameters,
            required_parameters=required_params,
            source_format=SchemaFormat.OPENAPI,
            original_schema=schema
        )

    def _parse_parameter(self, param: Dict, components: Dict) -> ParameterSpec:
        """Parse a parameter object (path, query, header)."""
        schema = param.get('schema', {})
        schema = self._resolve_ref(schema, components)

        return ParameterSpec(
            name=param.get('name', 'unknown'),
            type=self._map_type(schema.get('type', 'string')),
            description=param.get('description', schema.get('description', '')),
            required=param.get('required', False),
            default=schema.get('default'),
            enum=schema.get('enum'),
            minimum=schema.get('minimum'),
            maximum=schema.get('maximum'),
            min_length=schema.get('minLength'),
            max_length=schema.get('maxLength'),
            pattern=schema.get('pattern')
        )

    def _parse_request_body(
        self,
        request_body: Dict,
        components: Dict
    ) -> Tuple[List[ParameterSpec], List[str]]:
        """Parse request body into parameters."""
        parameters = []
        required_params = []

        content = request_body.get('content', {})
        # Prefer JSON content type
        media_type = (
            content.get('application/json') or
            content.get('application/x-www-form-urlencoded') or
            next(iter(content.values()), {})
        )

        schema = media_type.get('schema', {})
        schema = self._resolve_ref(schema, components)

        if schema.get('type') == 'object':
            required_list = schema.get('required', [])
            for prop_name, prop_spec in schema.get('properties', {}).items():
                prop_spec = self._resolve_ref(prop_spec, components)
                is_required = prop_name in required_list
                param = self._property_to_parameter(prop_name, prop_spec, is_required)
                parameters.append(param)
                if is_required:
                    required_params.append(prop_name)

        return parameters, required_params

    def _property_to_parameter(
        self,
        name: str,
        prop_spec: Dict,
        required: bool
    ) -> ParameterSpec:
        """Convert a property specification to a parameter."""
        return ParameterSpec(
            name=name,
            type=self._map_type(prop_spec.get('type', 'string')),
            description=prop_spec.get('description', ''),
            required=required,
            default=prop_spec.get('default'),
            enum=prop_spec.get('enum'),
            minimum=prop_spec.get('minimum'),
            maximum=prop_spec.get('maximum'),
            min_length=prop_spec.get('minLength'),
            max_length=prop_spec.get('maxLength'),
            pattern=prop_spec.get('pattern')
        )

    def _resolve_ref(self, schema: Dict, components: Dict) -> Dict:
        """Resolve a $ref pointer to its schema."""
        if '$ref' not in schema:
            return schema

        ref = schema['$ref']
        # Handle #/components/schemas/Name format
        if ref.startswith('#/components/schemas/'):
            schema_name = ref.split('/')[-1]
            resolved = components.get(schema_name, {})
            # Recursively resolve nested refs
            return self._resolve_ref(resolved, components)

        return schema

    def _extract_examples(self, operation: Dict, components: Dict) -> List[Example]:
        """Extract examples from operation."""
        examples = []

        # Try request body examples
        request_body = operation.get('requestBody', {})
        content = request_body.get('content', {}).get('application/json', {})

        if 'example' in content:
            examples.append(Example(
                prompt=f"Call {operation.get('operationId', 'this operation')}",
                expected_arguments=content['example']
            ))

        if 'examples' in content:
            for ex_name, ex_value in content['examples'].items():
                if isinstance(ex_value, dict) and 'value' in ex_value:
                    examples.append(Example(
                        prompt=ex_value.get('summary', ex_name),
                        expected_arguments=ex_value['value']
                    ))

        return examples

    def _map_type(self, openapi_type: Union[str, List[str]]) -> ParameterType:
        """Map OpenAPI type to ParameterType."""
        if isinstance(openapi_type, list):
            # Handle nullable types
            for t in openapi_type:
                if t != 'null':
                    return self.TYPE_MAPPING.get(t, ParameterType.STRING)
            return ParameterType.NULL

        return self.TYPE_MAPPING.get(openapi_type, ParameterType.STRING)

    def _path_to_name(self, path: str, method: str) -> str:
        """Generate an operation name from path and method."""
        # Remove path parameters and clean up
        clean_path = re.sub(r'\{[^}]+\}', '', path)
        clean_path = re.sub(r'[^a-zA-Z0-9]+', '_', clean_path)
        clean_path = clean_path.strip('_')
        return f"{method}_{clean_path}" if clean_path else method
