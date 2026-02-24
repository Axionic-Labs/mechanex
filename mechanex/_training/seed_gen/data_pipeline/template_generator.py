"""
Automated Seed Prompt Template Generator

This module uses LLMs to automatically generate seed prompt templates
from tool schemas. Instead of manually maintaining hundreds of templates,
the system reads tool schemas and generates diverse, realistic prompt templates.

Now with async batch processing for faster generation!
"""

import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from google import genai
from google.genai import types
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .prompt_templates import ComplexityLevel, PromptStyle, EdgeCaseType


class TemplateGenerator:
    """
    Generates seed prompt templates from tool schemas using an LLM.

    This replaces manual template creation by automatically generating
    diverse, realistic user prompts for each tool at different complexity levels.
    """

    TEMPLATE_GENERATION_PROMPT = """You are an expert at creating realistic user prompts for tool-calling systems.

Given a tool schema, generate diverse user prompt templates that someone would naturally say when they want to use this tool. The tool can be from ANY domain (CRM, productivity, data analysis, automation, API calls, etc.).

TOOL SCHEMA:
{tool_schema}

Generate {num_templates} diverse prompt templates for this tool with the following characteristics:

COMPLEXITY: {complexity}
STYLE: {style}
IS_EDGE_CASE: {is_edge_case}

Requirements:
1. Templates should feel NATURAL - like real user requests for THIS SPECIFIC TOOL
2. Use placeholders in curly braces {{}} for variable entities based on the tool's parameters (e.g., {{parameter_name}})
3. Vary the phrasing and structure significantly
4. Include different levels of context and detail appropriate to the tool's domain
5. Match the specified complexity and style
6. Adapt language and terminology to match the tool's domain (business, technical, casual, etc.)

**Complexity Guidelines:**
- SIMPLE: Direct, minimal context, straightforward - just the essential parameters
- MEDIUM: Some context, moderate detail - includes reasoning or additional information
- COMPLEX: Rich context, multiple details, narrative - full scenario with background and multiple parameters

**Style Guidelines:**
- DIRECT: Imperative, to-the-point (e.g., "Update lead X to status Y")
- CONVERSATIONAL: Casual, friendly (e.g., "Hey, can you update...")
- FORMAL: Professional, structured (e.g., "Please proceed to update...")
- URGENT: Time-sensitive, high priority (e.g., "URGENT: Update lead...")
- CONTEXTUAL: Includes background/reasoning (e.g., "After the meeting, we need to...")

**Edge Cases** (if is_edge_case=True):
- MISSING_INFO: Omit some required parameters
- AMBIGUOUS: Could be interpreted multiple ways
- INFORMAL: Very casual/slangy language
- TYPO: Include realistic typos

Output ONLY a JSON array of template objects, no other text:
[
  {{
    "template": "prompt template with {{placeholders}}",
    "required_entities": ["list", "of", "required", "placeholders"],
    "optional_entities": ["list", "of", "optional", "placeholders"]
  }},
  ...
]"""

    def __init__(self, model_type: str = "gemini", model_name: str = "gemini-2.0-flash", max_concurrent: int = 5):
        """
        Initialize the template generator.

        Args:
            model_type: Type of model to use (gemini, openai, etc.)
            model_name: Specific model name
            max_concurrent: Max concurrent LLM calls for async processing
        """
        self.model_type = model_type
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._semaphore = None  # Created per async call
        self._init_model()

    def _init_model(self):
        """Initialize the LLM backend."""
        if self.model_type == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = None
                print("⚠️ Warning: No GOOGLE_API_KEY or GEMINI_API_KEY found. Template generation will be disabled.")
        else:
            raise NotImplementedError(f"Model type {self.model_type} not yet implemented")

    def generate_templates_for_tool(
        self,
        tool_schema: Dict[str, Any],
        complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
        style: PromptStyle = PromptStyle.DIRECT,
        is_edge_case: bool = False,
        num_templates: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate seed prompt templates for a specific tool.

        Args:
            tool_schema: The tool schema dictionary
            complexity: Complexity level for templates
            style: Prompt style
            is_edge_case: Whether to generate edge case templates
            num_templates: Number of templates to generate

        Returns:
            List of template dictionaries
        """
        if self.client is None:
            print(f"⚠️ Model not available, returning empty templates for {tool_schema.get('name', 'unknown')}")
            return []

        try:
            prompt = self.TEMPLATE_GENERATION_PROMPT.format(
                tool_schema=json.dumps(tool_schema, indent=2),
                complexity=complexity.value.upper(),
                style=style.value.upper(),
                is_edge_case=is_edge_case,
                num_templates=num_templates
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            templates = json.loads(response_text)

            # Validate structure
            for template in templates:
                if not isinstance(template, dict):
                    continue
                if "template" not in template:
                    continue
                if "required_entities" not in template:
                    template["required_entities"] = []
                if "optional_entities" not in template:
                    template["optional_entities"] = []

            return templates

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response for {tool_schema.get('name')}: {e}")
            print(f"Response was: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"❌ Error generating templates for {tool_schema.get('name')}: {e}")
            return []

    def generate_all_templates_for_tool(
        self,
        tool_schema: Dict[str, Any],
        templates_per_config: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a comprehensive set of templates for a tool across different configurations.

        Args:
            tool_schema: The tool schema dictionary
            templates_per_config: Number of templates per complexity/style combination

        Returns:
            Dictionary mapping config keys to template lists
        """
        all_templates = {}
        tool_name = tool_schema.get("name", "unknown")

        print(f"🔄 Generating comprehensive templates for: {tool_name}")

        # Generate for each complexity level and major styles
        configs = [
            # Regular prompts - different complexity levels
            (ComplexityLevel.SIMPLE, PromptStyle.DIRECT, False),
            (ComplexityLevel.SIMPLE, PromptStyle.CONVERSATIONAL, False),
            (ComplexityLevel.MEDIUM, PromptStyle.DIRECT, False),
            (ComplexityLevel.MEDIUM, PromptStyle.CONVERSATIONAL, False),
            (ComplexityLevel.MEDIUM, PromptStyle.CONTEXTUAL, False),
            (ComplexityLevel.COMPLEX, PromptStyle.CONTEXTUAL, False),
            (ComplexityLevel.COMPLEX, PromptStyle.FORMAL, False),

            # Edge cases
            (ComplexityLevel.SIMPLE, PromptStyle.CONVERSATIONAL, True),
            (ComplexityLevel.MEDIUM, PromptStyle.DIRECT, True),
        ]

        for complexity, style, is_edge in configs:
            key = f"{tool_name}_{complexity.value}_{style.value}{'_edge' if is_edge else ''}"

            templates = self.generate_templates_for_tool(
                tool_schema=tool_schema,
                complexity=complexity,
                style=style,
                is_edge_case=is_edge,
                num_templates=templates_per_config
            )

            if templates:
                all_templates[key] = templates
                print(f"  ✅ Generated {len(templates)} templates for {complexity.value}/{style.value}{'  (edge)' if is_edge else ''}")
            else:
                print(f"  ⚠️  Failed to generate templates for {complexity.value}/{style.value}")

        return all_templates

    async def _generate_templates_async(
        self,
        tool_schema: Dict[str, Any],
        complexity: ComplexityLevel,
        style: PromptStyle,
        is_edge_case: bool,
        num_templates: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Async wrapper around generate_templates_for_tool."""
        key = f"{tool_schema.get('name', 'unknown')}_{complexity.value}_{style.value}{'_edge' if is_edge_case else ''}"
        
        # Run sync generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        templates = await loop.run_in_executor(
            self._executor,
            lambda: self.generate_templates_for_tool(
                tool_schema=tool_schema,
                complexity=complexity,
                style=style,
                is_edge_case=is_edge_case,
                num_templates=num_templates
            )
        )
        return (key, templates)

    async def generate_all_templates_for_tool_async(
        self,
        tool_schema: Dict[str, Any],
        templates_per_config: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Async version: Generate templates for a tool across different configurations concurrently.

        Args:
            tool_schema: The tool schema dictionary
            templates_per_config: Number of templates per complexity/style combination

        Returns:
            Dictionary mapping config keys to template lists
        """
        tool_name = tool_schema.get("name", "unknown")
        print(f"🔄 Generating templates for: {tool_name} (async batch)")

        # Define all configs to generate
        configs = [
            (ComplexityLevel.SIMPLE, PromptStyle.DIRECT, False),
            (ComplexityLevel.SIMPLE, PromptStyle.CONVERSATIONAL, False),
            (ComplexityLevel.MEDIUM, PromptStyle.DIRECT, False),
            (ComplexityLevel.MEDIUM, PromptStyle.CONVERSATIONAL, False),
            (ComplexityLevel.MEDIUM, PromptStyle.CONTEXTUAL, False),
            (ComplexityLevel.COMPLEX, PromptStyle.CONTEXTUAL, False),
            (ComplexityLevel.COMPLEX, PromptStyle.FORMAL, False),
            (ComplexityLevel.SIMPLE, PromptStyle.CONVERSATIONAL, True),
            (ComplexityLevel.MEDIUM, PromptStyle.DIRECT, True),
        ]

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def rate_limited_generate(complexity, style, is_edge):
            async with semaphore:
                return await self._generate_templates_async(
                    tool_schema, complexity, style, is_edge, templates_per_config
                )

        # Run all configs concurrently
        tasks = [
            rate_limited_generate(complexity, style, is_edge)
            for complexity, style, is_edge in configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_templates = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"  ⚠️ Error in async generation: {result}")
                continue
            key, templates = result
            if templates:
                all_templates[key] = templates
                print(f"  ✅ Generated {len(templates)} templates for {key}")

        return all_templates

    async def generate_templates_from_schemas_async(
        self,
        schemas_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        templates_per_config: int = 1,
        schemas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Async version: Generate templates for all tool schemas in parallel.

        Args:
            schemas_dir: Path to directory containing tool schema JSON files
            output_file: Optional path to save generated templates as JSON
            templates_per_config: Number of templates per complexity/style combo
            schemas: Optional direct list of schema dictionaries

        Returns:
            Dictionary mapping tool names to their template configurations
        """
        if schemas_dir is None:
            storage_base = os.getenv("FILE_STORAGE_PATH")
            if storage_base:
                schemas_dir = os.path.join(storage_base, "mechanex_train", "configs", "tool_schemas")

        # Load schemas from directory if provided
        schemas = schemas or []
        if schemas_dir:
            schemas_path = Path(schemas_dir)
            if not schemas_path.exists():
                # Only raise if it's not a default path we're fallbacking to
                if os.getenv("FILE_STORAGE_PATH") or schemas_dir.endswith("tool_schemas"):
                     pass # Be quiet about default missing paths
                else:
                    raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")

            for schema_file in sorted(schemas_path.glob("*.json")):
                try:
                    with open(schema_file, 'r') as f:
                        tool_schema = json.load(f)
                        schemas.append(tool_schema)
                except Exception as e:
                    print(f"❌ Error reading {schema_file}: {e}")

        if not schemas:
            print("⚠️ No schemas provided or found.")
            return {}

        print(f"🚀 Processing {len(schemas)} tool schemas in parallel...")

        # Generate templates for all tools concurrently
        tasks = [
            self.generate_all_templates_for_tool_async(schema, templates_per_config)
            for schema in schemas
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_tool_templates = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Error processing schema: {result}")
                continue
            tool_name = schemas[i].get("name", f"tool_{i}")
            all_tool_templates[tool_name] = result

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_tool_templates, f, indent=2)
            print(f"✅ All templates saved to {output_file}")

        return all_tool_templates

    def generate_templates_from_schemas(
        self,
        schemas_dir: str,
        output_file: Optional[str] = None,
        templates_per_config: int = 1
    ) -> Dict[str, Any]:
        """
        Generate templates for all tool schemas in a directory.

        Args:
            schemas_dir: Path to directory containing tool schema JSON files
            output_file: Optional path to save generated templates as JSON
            templates_per_config: Number of templates per complexity/style combo

        Returns:
            Dictionary mapping tool names to their template configurations
        """
        schemas_path = Path(schemas_dir)

        if not schemas_path.exists():
            raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")

        all_tool_templates = {}

        # Read all JSON schema files
        for schema_file in sorted(schemas_path.glob("*.json")):
            try:
                with open(schema_file, 'r') as f:
                    tool_schema = json.load(f)

                tool_name = tool_schema.get("name", schema_file.stem)

                tool_templates = self.generate_all_templates_for_tool(
                    tool_schema=tool_schema,
                    templates_per_config=templates_per_config
                )

                all_tool_templates[tool_name] = tool_templates

            except Exception as e:
                print(f"❌ Error processing {schema_file}: {e}")
                continue

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(all_tool_templates, f, indent=2)

            print(f"✅ All templates saved to {output_file}")

        return all_tool_templates


def load_templates_from_schemas(
    schemas_dir: str,
    cache_file: Optional[str] = None,
    force_regenerate: bool = False,
    templates_per_config: int = 1
) -> Dict[str, Any]:
    """
    Load or generate templates from tool schemas (synchronous version).

    Args:
        schemas_dir: Path to tool schemas directory
        cache_file: Optional path to cache file for templates
        force_regenerate: If True, regenerate even if cache exists
        templates_per_config: Number of templates per configuration

    Returns:
        Dictionary of tool templates
    """
    # Check if cache exists and should be used
    if cache_file and not force_regenerate and Path(cache_file).exists():
        print(f"📂 Loading templates from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Generate templates (sync version)
    print("🔄 Generating templates from tool schemas...")
    generator = TemplateGenerator()
    templates = generator.generate_templates_from_schemas(
        schemas_dir=schemas_dir,
        output_file=cache_file,
        templates_per_config=templates_per_config
    )

    return templates


async def load_templates_from_schemas_async(
    schemas_dir: Optional[str] = None,
    cache_file: Optional[str] = None,
    force_regenerate: bool = False,
    templates_per_config: int = 1,
    max_concurrent: int = 5,
    schemas: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Load or generate templates from tool schemas (async batch version - FASTER!).

    Args:
        schemas_dir: Path to tool schemas directory
        cache_file: Optional path to cache file for templates
        force_regenerate: If True, regenerate even if cache exists
        templates_per_config: Number of templates per configuration
        max_concurrent: Maximum concurrent LLM calls
        schemas: Optional direct list of schema dictionaries

    Returns:
        Dictionary of tool templates
    """
    # Check if cache exists and should be used
    if cache_file and not force_regenerate and Path(cache_file).exists():
        print(f"📂 Loading templates from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Generate templates using async batch processing
    print("🚀 Generating templates from tool schemas (async batch mode)...")
    generator = TemplateGenerator(max_concurrent=max_concurrent)
    templates = await generator.generate_templates_from_schemas_async(
        schemas_dir=schemas_dir,
        output_file=cache_file,
        templates_per_config=templates_per_config,
        schemas=schemas
    )

    return templates

