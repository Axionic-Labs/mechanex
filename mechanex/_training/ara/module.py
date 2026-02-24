"""Main ARA Module facade."""

import logging
from typing import Union, Dict, Optional, Any

from .types.enums import TeacherProvider, TaskType, SchemaFormat
from .types.config import ARAConfig, REWARD_WEIGHTS
from .types.results import ExecutionResult
from .parsing.canonical import CanonicalSchema
from .parsing.base import SchemaParser
from .parsing.json_schema import JSONSchemaParser
from .parsing.openapi import OpenAPIParser
from .parsing.natural_language import NaturalLanguageParser
from .clients.teacher import TeacherClient
from .compilation.compiler import RewardCompiler
from .validation.sandbox import RewardSandbox, TestCase

logger = logging.getLogger(__name__)


class ARAModule:
    """
    Automated Reward Architect - Main Interface.

    Compiles tool schemas into executable reward functions for RL training.

    Supports multiple input formats:
    - JSON Schema
    - OpenAPI 3.0+ specifications
    - Natural language descriptions

    Example:
        ```python
        ara = ARAModule()

        # Compile from JSON schema
        schema = {
            "name": "update_lead_status",
            "description": "Update CRM lead status",
            "parameters": {...}
        }
        reward_fn = ara.compile(schema)

        # Compile from natural language
        reward_fn = ara.compile(
            "A tool called 'search_users' that takes a required 'query' parameter",
            format_hint=SchemaFormat.NATURAL_LANGUAGE
        )

        # Evaluate response
        result = reward_fn(response, prompt)
        print(result['score'])  # 0.85
        ```
    """

    def __init__(self, config: Optional[ARAConfig] = None):
        """
        Initialize ARA Module.

        Args:
            config: Configuration for ARA module. Uses defaults if not provided.
        """
        self.config = config or ARAConfig()

        # Initialize components
        self.teacher = TeacherClient(
            provider=self.config.teacher_provider,
            model=self.config.teacher_model,
            api_key=self.config.teacher_api_key
        )

        # Initialize all parsers
        self._parsers: Dict[SchemaFormat, SchemaParser] = {
            SchemaFormat.JSON_SCHEMA: JSONSchemaParser(),
            SchemaFormat.OPENAPI: OpenAPIParser(),
            SchemaFormat.NATURAL_LANGUAGE: NaturalLanguageParser(teacher_client=self.teacher),
        }
        self._default_parser = JSONSchemaParser()

        self.compiler = RewardCompiler(self.teacher)
        self.sandbox = RewardSandbox()

        # Cache for compiled rewards
        self._cache: Dict[str, Any] = {}

    def _get_parser(self, input_schema: Union[str, Dict], format_hint: Optional[SchemaFormat] = None) -> SchemaParser:
        """Get the appropriate parser for the input."""
        if format_hint and format_hint in self._parsers:
            return self._parsers[format_hint]

        # Auto-detect format
        detected = self._default_parser.detect_format(input_schema)
        if detected in self._parsers:
            return self._parsers[detected]

        return self._default_parser

    def compile(
        self,
        schema: Union[str, Dict],
        task_type: Optional[TaskType] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        use_cache: bool = True,
        format_hint: Optional[SchemaFormat] = None
    ) -> Any:
        """
        Compile a tool schema into an executable reward function.

        Args:
            schema: Tool schema (JSON Schema, OpenAPI, or natural language)
            task_type: Task type for weight configuration
            custom_weights: Override default weights
            use_cache: Whether to use cached rewards
            format_hint: Optional hint for input format (auto-detected if not provided)

        Returns:
            Callable reward function: fn(response, prompt) -> dict

        Raises:
            CompilationError: If compilation fails
            ValidationError: If generated code fails security validation
        """
        # Get appropriate parser and parse schema
        parser = self._get_parser(schema, format_hint)
        canonical = parser.parse(schema)

        # Check cache
        cache_key = canonical.hash()
        if use_cache and cache_key in self._cache:
            logger.info(f"Using cached reward for {canonical.name}")
            return self._cache[cache_key]

        # Compile reward function
        task_type = task_type or self.config.default_task_type
        weights = custom_weights or REWARD_WEIGHTS.get(task_type, self.config.default_weights)

        generated = self.compiler.compile(
            schema=canonical,
            task_type=task_type,
            custom_weights=weights,
            pass_threshold=self.config.pass_threshold
        )

        # Validate and execute in sandbox
        result = self.sandbox.execute(generated.code)

        if not result.success:
            raise ValueError(f"Sandbox execution failed: {result.error}")

        # Cache the reward instance
        if use_cache:
            self._cache[cache_key] = result.reward_instance

        return result.reward_instance

    def compile_with_code(
        self,
        schema: Union[str, Dict],
        task_type: Optional[TaskType] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        format_hint: Optional[SchemaFormat] = None
    ) -> tuple:
        """
        Compile schema and return both reward function and generated code.

        Args:
            schema: Tool schema (JSON Schema, OpenAPI, or natural language)
            task_type: Task type for weight configuration
            custom_weights: Override default weights
            format_hint: Optional hint for input format (auto-detected if not provided)

        Returns:
            Tuple of (reward_instance, generated_code)
        """
        parser = self._get_parser(schema, format_hint)
        canonical = parser.parse(schema)
        task_type = task_type or self.config.default_task_type
        weights = custom_weights or REWARD_WEIGHTS.get(task_type, self.config.default_weights)

        generated = self.compiler.compile(
            schema=canonical,
            task_type=task_type,
            custom_weights=weights,
            pass_threshold=self.config.pass_threshold
        )

        result = self.sandbox.execute(generated.code)

        if not result.success:
            raise ValueError(f"Sandbox execution failed: {result.error}")

        return result.reward_instance, generated.code

    def validate_code(self, code: str) -> ExecutionResult:
        """
        Validate generated reward code.

        Args:
            code: Python code to validate

        Returns:
            ExecutionResult with validation status
        """
        return self.sandbox.execute(code)

    def clear_cache(self):
        """Clear the reward function cache."""
        self._cache.clear()

    def get_weights(self, task_type: TaskType) -> Dict[str, float]:
        """Get reward weights for a task type."""
        return REWARD_WEIGHTS.get(task_type, self.config.default_weights)
