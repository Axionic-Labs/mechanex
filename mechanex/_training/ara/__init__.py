"""
Automated Reward Architect (ARA) - POC Implementation.

ARA automatically generates reward functions for RL training from tool schemas.
It treats tool definitions as specifications and uses frontier models as compilers
to generate executable Python reward functions.

Example:
    ```python
    from ara import ARAModule, ARAConfig, TeacherProvider

    # Initialize with Google's Gemini
    ara = ARAModule(config=ARAConfig(
        teacher_provider=TeacherProvider.GOOGLE,
        teacher_model="gemini-2.0-flash"
    ))

    # Define a tool schema
    schema = {
        "name": "update_lead_status",
        "description": "Update the status of a lead in the CRM",
        "parameters": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string", "pattern": "^LD-\\\\d+$"},
                "status": {"type": "string", "enum": ["New", "Contacted", "Qualified"]}
            },
            "required": ["lead_id", "status"]
        }
    }

    # Compile into a reward function
    reward_fn = ara.compile(schema)

    # Evaluate a response
    response = '''
    <think>
    I need to update lead LD-1234 to Qualified status.
    </think>
    <tool_call>
    {"name": "update_lead_status", "arguments": {"lead_id": "LD-1234", "status": "Qualified"}}
    </tool_call>
    '''

    result = reward_fn(response, "Update lead LD-1234 to Qualified")
    print(f"Score: {result['score']}")  # Score: 0.92
    print(f"Passed: {result['passed']}")  # Passed: True
    ```
"""

from .module import ARAModule
from .types.enums import SchemaFormat, TaskType, TeacherProvider, ParameterType
from .types.config import ARAConfig, SandboxConfig, CompilerConfig, REWARD_WEIGHTS
from .types.results import RewardResult, ValidationResult, ExecutionResult
from .parsing.canonical import CanonicalSchema, ParameterSpec, Example
from .parsing.json_schema import JSONSchemaParser
from .parsing.openapi import OpenAPIParser
from .parsing.natural_language import NaturalLanguageParser
from .validation.sandbox import TestCase

__version__ = "0.1.0"

__all__ = [
    # Main module
    'ARAModule',
    # Enums
    'SchemaFormat',
    'TaskType',
    'TeacherProvider',
    'ParameterType',
    # Config
    'ARAConfig',
    'SandboxConfig',
    'CompilerConfig',
    'REWARD_WEIGHTS',
    # Results
    'RewardResult',
    'ValidationResult',
    'ExecutionResult',
    # Schema
    'CanonicalSchema',
    'ParameterSpec',
    'Example',
    # Parsers
    'JSONSchemaParser',
    'OpenAPIParser',
    'NaturalLanguageParser',
    # Testing
    'TestCase',
]
