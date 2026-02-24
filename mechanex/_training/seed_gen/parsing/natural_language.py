"""Natural Language schema parser for SeedGen.

This parser uses an LLM to convert natural language tool descriptions
into structured canonical schemas.

Ported from ARA module.
"""

from typing import Union, Optional, Dict, Any, List, Tuple
import json
import re

from .base import SchemaParser
from .canonical import CanonicalSchema, ParameterSpec, Example
from ..models.enums import SchemaFormat, ParameterType


# Prompt template for LLM-based parsing
NL_PARSING_PROMPT = """You are a tool schema extraction expert. Given a natural language description of a tool or function, extract its structured specification.

<description>
{description}
</description>

{context_section}

Extract the following information and return ONLY valid JSON (no markdown, no code blocks):

{{
    "name": "snake_case_tool_name",
    "description": "Brief description of what the tool does",
    "parameters": [
        {{
            "name": "param_name",
            "type": "string|integer|number|boolean|array|object",
            "description": "What this parameter is for",
            "required": true,
            "enum": ["optional", "list", "of", "values"],
            "pattern": "optional regex pattern",
            "minimum": null,
            "maximum": null
        }}
    ],
    "examples": [
        {{
            "prompt": "Example user request",
            "arguments": {{"param_name": "example_value"}}
        }}
    ]
}}

Rules:
1. Infer parameter types from context (e.g., "ID" suggests string, "count" suggests integer)
2. Mark parameters as required if they seem essential
3. Add enums for parameters with a clear set of valid values
4. Add patterns for IDs, emails, URLs, etc.
5. Generate 1-2 realistic examples
6. Use snake_case for all names

Return ONLY the JSON object, nothing else."""


class NaturalLanguageParser(SchemaParser):
    """Parser for natural language tool descriptions.

    Uses an LLM to convert free-form text descriptions into structured schemas.
    Can operate in two modes:
    - With LLM: Uses TeacherClient for intelligent extraction
    - Without LLM: Uses heuristic-based extraction (fallback)
    """

    TYPE_KEYWORDS = {
        ParameterType.STRING: ['string', 'text', 'name', 'id', 'email', 'url', 'path', 'description'],
        ParameterType.INTEGER: ['integer', 'int', 'count', 'number of', 'quantity', 'index', 'age'],
        ParameterType.NUMBER: ['number', 'float', 'decimal', 'amount', 'price', 'rate', 'percentage'],
        ParameterType.BOOLEAN: ['boolean', 'bool', 'flag', 'is_', 'has_', 'should', 'enable', 'active'],
        ParameterType.ARRAY: ['array', 'list', 'items', 'collection', 'multiple'],
        ParameterType.OBJECT: ['object', 'dict', 'map', 'record', 'structure'],
    }

    def __init__(self, teacher_client=None):
        """
        Initialize the parser.

        Args:
            teacher_client: Optional TeacherClient for LLM-based parsing.
                           If None, uses heuristic-based extraction.
        """
        self.teacher = teacher_client

    def parse(
        self,
        input_schema: Union[str, Dict],
        format_hint: Optional[SchemaFormat] = None,
        context: Optional[str] = None
    ) -> CanonicalSchema:
        """
        Parse natural language description into canonical form.

        Args:
            input_schema: Natural language description of the tool
            format_hint: Optional format hint (ignored for NL parser)
            context: Optional enterprise context for better extraction

        Returns:
            CanonicalSchema: Extracted tool schema
        """
        if not isinstance(input_schema, str):
            raise ValueError("NaturalLanguageParser expects a string description")

        description = input_schema.strip()

        if self.teacher:
            return self._parse_with_llm(description, context)
        else:
            return self._parse_heuristic(description, context)

    def _parse_with_llm(self, description: str, context: Optional[str]) -> CanonicalSchema:
        """Parse using LLM for intelligent extraction."""
        context_section = ""
        if context:
            context_section = f"<context>\n{context}\n</context>\n"

        prompt = NL_PARSING_PROMPT.format(
            description=description,
            context_section=context_section
        )

        try:
            response = self.teacher.generate(
                prompt=prompt,
                system="You are a precise JSON extractor. Return only valid JSON.",
                temperature=0.0,
                max_tokens=2000
            )

            # Extract JSON from response
            parsed = self._extract_json(response)
            return self._dict_to_canonical(parsed, description)

        except Exception as e:
            # Fallback to heuristic if LLM fails
            import logging
            logging.warning(f"LLM parsing failed, using heuristic: {e}")
            return self._parse_heuristic(description, context)

    def _parse_heuristic(self, description: str, context: Optional[str]) -> CanonicalSchema:
        """Parse using heuristic-based extraction (no LLM required)."""
        # Extract tool name
        name = self._extract_name(description)

        # Extract parameters
        parameters, required_params = self._extract_parameters(description)

        # Generate a simple example if we have parameters
        examples = []
        if parameters:
            example_args = {}
            for param in parameters:
                example_args[param.name] = self._generate_example_value(param)
            examples.append(Example(
                prompt=f"Use {name}",
                expected_arguments=example_args
            ))

        return CanonicalSchema(
            name=name,
            description=self._extract_description(description),
            parameters=parameters,
            required_parameters=required_params,
            examples=examples,
            source_format=SchemaFormat.NATURAL_LANGUAGE,
            original_schema={'raw_description': description}
        )

    def _extract_name(self, description: str) -> str:
        """Extract tool name from description."""
        # Look for explicit naming patterns
        patterns = [
            # "tool called X" or "function named X"
            r'(?:tool|function|api|endpoint)\s+(?:called|named)\s+["\']?(\w+)',
            # "the X function" or "the X tool"
            r'(?:the\s+)?(\w+)\s+(?:function|tool|api|endpoint)',
            # "'X' tool" or "X tool"
            r'["\']?(\w+)["\']?\s+(?:tool|function|api)',
            # "X: takes..." pattern (name at start with colon)
            r'^(\w+)\s*:\s*(?:takes|accepts|requires)',
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Skip common stop words
                if name.lower() not in ['a', 'an', 'the', 'this', 'that']:
                    return self.normalize_name(name)

        # Fallback: look for snake_case or camelCase words (likely function names)
        snake_case = re.search(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', description)
        if snake_case:
            return self.normalize_name(snake_case.group(1))

        camel_case = re.search(r'\b([a-z]+[A-Z][a-zA-Z0-9]*)\b', description)
        if camel_case:
            return self.normalize_name(camel_case.group(1))

        # Last fallback: use first significant word
        words = re.findall(r'\b[a-zA-Z]\w+\b', description)
        for word in words:
            if word.lower() not in ['a', 'an', 'the', 'this', 'that', 'tool', 'function', 'api', 'endpoint']:
                return self.normalize_name(word)

        return 'unknown_tool'

    def _extract_description(self, description: str) -> str:
        """Extract a clean description."""
        # Take first sentence or first 200 chars
        sentences = re.split(r'[.!?]', description)
        if sentences:
            first = sentences[0].strip()
            if len(first) > 10:
                return first
        return description[:200].strip()

    def _extract_parameters(self, description: str) -> Tuple[List[ParameterSpec], List[str]]:
        """Extract parameters from description."""
        parameters = []
        required = []

        # Look for parameter patterns
        param_patterns = [
            # 1. "a required X parameter" or "required X parameter"
            r'(?:a\s+)?required\s+["\']?(\w+)["\']?\s+parameter',
            # 2. "an optional X parameter" or "optional X parameter"
            r'(?:an?\s+)?optional\s+["\']?(\w+)["\']?\s+parameter',
            # 3. "takes X (type)" or "requires X (type)" - More flexible version
            r'(?:takes?|requires?|accepts?)\s+["\']?(\w+)["\']?\s*\(([^)]+)\)',
            # 4. "takes a parameter called X"
            r'(?:takes?|accepts?|requires?)\s+(?:a\s+)?(?:parameter|argument|input)\s+(?:called\s+)?["\']?(\w+)["\']?(?:\s*\(([^)]+)\))?',
            # 5. Standalone "X (type): description" or just "X (type)"
            r'\b(\w+)\s*\(([^)]+)\)(?:\s*[:-]\s*([^,.\n]+))?',
            # 6. "X: type - description"
            r'\b(\w+)\s*:\s*(\w+)\s*[-–]\s*([^,.\n]+)',
            # 7. "with X and Y parameters"
            r'with\s+(?:a\s+)?(\w+)(?:\s+and\s+(\w+))?\s+parameter',
        ]

        found_params = set()

        for pattern in param_patterns:
            for match in re.finditer(pattern, description, re.IGNORECASE):
                groups = match.groups()
                param_name = groups[0]

                if param_name.lower() in found_params:
                    continue
                found_params.add(param_name.lower())

                # Try to extract type from match
                param_type = ParameterType.STRING
                type_hint = None
                if len(groups) > 1 and groups[1]:
                    type_hint = groups[1]
                    param_type = self._infer_type(type_hint)
                else:
                    param_type = self._infer_type(param_name)

                # Check if required
                is_required = self._is_required(param_name, description, type_hint=type_hint)

                param = ParameterSpec(
                    name=self.normalize_name(param_name),
                    type=param_type,
                    description=groups[2] if len(groups) > 2 and groups[2] else '',
                    required=is_required
                )
                parameters.append(param)
                if is_required:
                    required.append(param.name)

        # Look for common parameter names in description
        common_params = [
            ('id', ParameterType.STRING, True),
            ('name', ParameterType.STRING, True),
            ('email', ParameterType.STRING, True),
            ('status', ParameterType.STRING, False),
            ('count', ParameterType.INTEGER, False),
            ('amount', ParameterType.NUMBER, False),
            ('enabled', ParameterType.BOOLEAN, False),
            ('active', ParameterType.BOOLEAN, False),
        ]

        for param_name, param_type, default_required in common_params:
            if param_name.lower() in found_params:
                continue

            # Check if this parameter is mentioned
            if re.search(rf'\b{param_name}\b', description, re.IGNORECASE):
                found_params.add(param_name.lower())
                is_required = default_required and self._is_required(param_name, description)
                param = ParameterSpec(
                    name=param_name,
                    type=param_type,
                    required=is_required
                )
                parameters.append(param)
                if is_required:
                    required.append(param_name)

        return parameters, required

    def _infer_type(self, text: str) -> ParameterType:
        """Infer parameter type from text."""
        text_lower = text.lower()

        for param_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return param_type

        return ParameterType.STRING

    def _is_required(self, param_name: str, description: str, type_hint: Optional[str] = None) -> bool:
        """Check if a parameter is required based on context."""
        # Check type hint first (e.g., "required string")
        if type_hint:
            if 'required' in type_hint.lower():
                return True
            if 'optional' in type_hint.lower():
                return False

        # Look for explicit required/optional markers in the full description
        required_patterns = [
            rf'{param_name}\s+(?:is\s+)?required',
            rf'required\s+{param_name}',
            rf'must\s+(?:provide|specify|include)\s+{param_name}',
        ]
        optional_patterns = [
            rf'{param_name}\s+(?:is\s+)?optional',
            rf'optional\s+{param_name}',
            rf'can\s+(?:optionally\s+)?(?:provide|specify)\s+{param_name}',
        ]

        for pattern in required_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return True

        for pattern in optional_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return False

        # Default: ID-like parameters are usually required
        if any(x in param_name.lower() for x in ['id', '_id', 'key']):
            return True

        return False

    def _generate_example_value(self, param: ParameterSpec) -> Any:
        """Generate an example value for a parameter."""
        if param.enum:
            return param.enum[0]

        if param.default is not None:
            return param.default

        type_examples = {
            ParameterType.STRING: f"example_{param.name}",
            ParameterType.INTEGER: 1,
            ParameterType.NUMBER: 1.0,
            ParameterType.BOOLEAN: True,
            ParameterType.ARRAY: [],
            ParameterType.OBJECT: {},
            ParameterType.NULL: None,
        }

        return type_examples.get(param.type, "example")

    def _extract_json(self, response: str) -> Dict:
        """Extract JSON from LLM response."""
        # Try to parse directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract from code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue

        raise ValueError("Could not extract valid JSON from LLM response")

    def _dict_to_canonical(self, data: Dict, original: str) -> CanonicalSchema:
        """Convert extracted dict to CanonicalSchema."""
        parameters = []
        required_params = []

        for param in data.get('parameters', []):
            param_type = ParameterType.STRING
            type_str = param.get('type', 'string').lower()
            for pt in ParameterType:
                if pt.value == type_str:
                    param_type = pt
                    break

            spec = ParameterSpec(
                name=param.get('name', 'unknown'),
                type=param_type,
                description=param.get('description', ''),
                required=param.get('required', False),
                default=param.get('default'),
                enum=param.get('enum'),
                minimum=param.get('minimum'),
                maximum=param.get('maximum'),
                min_length=param.get('minLength'),
                max_length=param.get('maxLength'),
                pattern=param.get('pattern')
            )
            parameters.append(spec)
            if spec.required:
                required_params.append(spec.name)

        examples = []
        for ex in data.get('examples', []):
            examples.append(Example(
                prompt=ex.get('prompt', ''),
                expected_arguments=ex.get('arguments', {})
            ))

        return CanonicalSchema(
            name=self.normalize_name(data.get('name', 'unknown_tool')),
            description=data.get('description', ''),
            parameters=parameters,
            required_parameters=required_params,
            examples=examples,
            source_format=SchemaFormat.NATURAL_LANGUAGE,
            original_schema={'raw_description': original, 'extracted': data}
        )
