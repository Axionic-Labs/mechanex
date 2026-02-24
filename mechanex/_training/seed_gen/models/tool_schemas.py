"""
Tool Schema Definitions

This module defines the tool schemas used by the Seed Generator POC.
These schemas represent the tools that the student model will learn to call.

Schemas are loaded from JSON files in configs/tool_schemas/ for easy modification.
Supports any domain - not limited to CRM or specific use cases.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dotenv import load_dotenv

# Ensure environment is loaded even if imported directly
load_dotenv()
from pydantic import BaseModel, Field
from enum import Enum


class ParameterType(str, Enum):
    """Supported parameter types for tool arguments."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a single tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter data type")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(default=True, description="Whether the parameter is required")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values for enums")
    minimum: Optional[float] = Field(default=None, description="Minimum value for numbers")
    maximum: Optional[float] = Field(default=None, description="Maximum value for numbers")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")
    items_type: Optional[ParameterType] = Field(default=None, description="Type of array items")


class ToolSchema(BaseModel):
    """Complete tool schema definition."""
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable tool description")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")

    # Metadata for training
    complexity: Literal["simple", "medium", "complex"] = Field(
        default="medium",
        description="Complexity level for curriculum learning"
    )
    category: str = Field(default="general", description="Tool category for grouping")

    def get_required_params(self) -> List[ToolParameter]:
        """Return only required parameters."""
        return [p for p in self.parameters if p.required]

    def get_optional_params(self) -> List[ToolParameter]:
        """Return only optional parameters."""
        return [p for p in self.parameters if not p.required]

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for API compatibility."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type.value, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            if param.default is not None:
                prop["default"] = param.default
            if param.items_type:
                prop["items"] = {"type": param.items_type.value}

            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_canonical(self) -> "CanonicalSchema":
        """Convert to CanonicalSchema for compatibility with parsing infrastructure.
        
        Returns:
            CanonicalSchema: Normalized canonical representation
        """
        from .canonical import CanonicalSchema, ParameterSpec
        from .enums import ParameterType as CanonicalParameterType, SchemaFormat
        
        # Map ToolParameter type to canonical ParameterType
        type_mapping = {
            ParameterType.STRING: CanonicalParameterType.STRING,
            ParameterType.INTEGER: CanonicalParameterType.INTEGER,
            ParameterType.NUMBER: CanonicalParameterType.NUMBER,
            ParameterType.BOOLEAN: CanonicalParameterType.BOOLEAN,
            ParameterType.ARRAY: CanonicalParameterType.ARRAY,
            ParameterType.OBJECT: CanonicalParameterType.OBJECT,
        }
        
        parameters = []
        required_params = []
        
        for param in self.parameters:
            spec = ParameterSpec(
                name=param.name,
                type=type_mapping.get(param.type, CanonicalParameterType.STRING),
                description=param.description,
                required=param.required,
                default=param.default,
                enum=param.enum,
                minimum=param.minimum,
                maximum=param.maximum,
            )
            parameters.append(spec)
            if param.required:
                required_params.append(param.name)
        
        return CanonicalSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            required_parameters=required_params,
            source_format=SchemaFormat.JSON_SCHEMA,
            complexity=self.complexity,
            category=self.category
        )

    @classmethod
    def from_canonical(cls, canonical: "CanonicalSchema") -> "ToolSchema":
        """Create a ToolSchema from a CanonicalSchema.
        
        Args:
            canonical: CanonicalSchema to convert
            
        Returns:
            ToolSchema: Equivalent ToolSchema representation
        """
        from .canonical import CanonicalSchema as CS
        from .enums import ParameterType as CanonicalParameterType
        
        # Map canonical ParameterType to ToolParameter type
        type_mapping = {
            CanonicalParameterType.STRING: ParameterType.STRING,
            CanonicalParameterType.INTEGER: ParameterType.INTEGER,
            CanonicalParameterType.NUMBER: ParameterType.NUMBER,
            CanonicalParameterType.BOOLEAN: ParameterType.BOOLEAN,
            CanonicalParameterType.ARRAY: ParameterType.ARRAY,
            CanonicalParameterType.OBJECT: ParameterType.OBJECT,
            CanonicalParameterType.NULL: ParameterType.STRING,  # Fallback
        }
        
        parameters = []
        for param in canonical.parameters:
            tool_param = ToolParameter(
                name=param.name,
                type=type_mapping.get(param.type, ParameterType.STRING),
                description=param.description,
                required=param.required,
                default=param.default,
                enum=param.enum,
                minimum=param.minimum,
                maximum=param.maximum,
            )
            parameters.append(tool_param)
        
        return cls(
            name=canonical.name,
            description=canonical.description,
            parameters=parameters,
            complexity=canonical.complexity,
            category=canonical.category
        )


# =============================================================================
# SCHEMA LOADING FROM JSON FILES
# =============================================================================

def get_schemas_directory() -> Path:
    """Get the path to the tool schemas config directory."""
    # Priority 1: FILE_STORAGE_PATH env var
    storage_base = os.getenv("FILE_STORAGE_PATH")
    if storage_base:
        return Path(storage_base) / "mechanex_train" / "configs" / "tool_schemas"
        
    # Priority 2: Navigate from src/models/ to configs/tool_schemas/
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root / "configs" / "tool_schemas"


def load_tool_schema_from_file(filepath: Path) -> ToolSchema:
    """Load a single tool schema from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert parameter dicts to ToolParameter objects
    parameters = []
    for param_data in data.get("parameters", []):
        # Convert string type to ParameterType enum
        param_data["type"] = ParameterType(param_data["type"])
        parameters.append(ToolParameter(**param_data))

    return ToolSchema(
        name=data["name"],
        description=data["description"],
        category=data.get("category", "general"),
        complexity=data.get("complexity", "medium"),
        parameters=parameters
    )


def load_all_tool_schemas() -> List[ToolSchema]:
    """Load all tool schemas from the configs/tool_schemas directory."""
    schemas_dir = get_schemas_directory()
    schemas = []

    if not schemas_dir.exists():
        # Silence for default registry; API calls provide specific paths
        return []

    for json_file in sorted(schemas_dir.glob("*.json")):
        try:
            schema = load_tool_schema_from_file(json_file)
            schemas.append(schema)
        except Exception as e:
            print(f"Warning: Failed to load schema from {json_file}: {e}")

    return schemas


# Track registered tools in memory (in addition to those on disk)
REGISTERED_TOOLS: List[ToolSchema] = []

def reload_schemas():
    """Reload schemas from disk and include registered tools."""
    global TOOLS, TOOLS_BY_NAME, CRM_TOOLS, CRM_TOOLS_BY_NAME
    on_disk = load_all_tool_schemas()
    TOOLS = on_disk + REGISTERED_TOOLS
    TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
    # Update aliases
    CRM_TOOLS = TOOLS
    CRM_TOOLS_BY_NAME = TOOLS_BY_NAME

def register_tools(new_tools: List[ToolSchema]):
    """Register new tools in memory and reload the registry."""
    global REGISTERED_TOOLS
    REGISTERED_TOOLS.extend(new_tools)
    reload_schemas()


# =============================================================================
# LOAD CRM TOOLS ON MODULE IMPORT
# =============================================================================

# Registry of all tools (loaded from JSON files)
TOOLS: List[ToolSchema] = load_all_tool_schemas()

# Tool lookup by name
TOOLS_BY_NAME: Dict[str, ToolSchema] = {tool.name: tool for tool in TOOLS}

# Backward compatibility aliases
CRM_TOOLS = TOOLS
CRM_TOOLS_BY_NAME = TOOLS_BY_NAME


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_tool_by_name(name: str) -> Optional[ToolSchema]:
    """Retrieve a tool schema by its name."""
    return TOOLS_BY_NAME.get(name)


def get_tools_by_complexity(complexity: str) -> List[ToolSchema]:
    """Get all tools matching a complexity level."""
    return [t for t in TOOLS if t.complexity == complexity]


def get_tools_by_category(category: str) -> List[ToolSchema]:
    """Get all tools matching a category."""
    return [t for t in TOOLS if t.category == category]


def list_available_tools() -> List[str]:
    """List names of all available tools."""
    return [t.name for t in TOOLS]


def add_tool_schema(filepath: Path) -> ToolSchema:
    """
    Add a new tool schema from a JSON file and reload the registry.

    Args:
        filepath: Path to the new JSON schema file

    Returns:
        The newly loaded ToolSchema
    """
    schema = load_tool_schema_from_file(filepath)
    reload_schemas()
    return schema
