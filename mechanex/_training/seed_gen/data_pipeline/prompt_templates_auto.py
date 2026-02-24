"""
Automated Prompt Template System for Seed Generation

This module replaces the manual template system with LLM-generated templates
from tool schemas. Templates are automatically loaded and managed.
"""

import json
import os
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

from .prompt_templates import (
    ComplexityLevel,
    PromptStyle,
    EdgeCaseType,
    PromptTemplate,
    EntityGenerator
)
from .template_generator import load_templates_from_schemas


class AutoPromptTemplateSystem:
    """
    Automated prompt template system that loads LLM-generated templates
    from tool schemas instead of using manually defined templates.
    """

    def __init__(
        self,
        schemas_dir: Optional[str] = None,
        cache_file: Optional[str] = None,
        auto_generate: bool = True
    ):
        """
        Initialize the automated template system.

        Args:
            schemas_dir: Path to tool schemas directory
            cache_file: Path to template cache file
            auto_generate: If True, generate templates from schemas
        """
        self.entity_generator = EntityGenerator()
        self.templates: Dict[str, List[PromptTemplate]] = {}

        # Default paths
        storage_base = os.getenv("FILE_STORAGE_PATH")
        
        if schemas_dir is None:
            if storage_base:
                schemas_dir = Path(storage_base) / "mechanex_train" / "configs" / "tool_schemas"
            else:
                current_dir = Path(__file__).parent.parent.parent
                schemas_dir = current_dir / "configs" / "tool_schemas"

        if cache_file is None:
            if storage_base:
                cache_file = Path(storage_base) / "mechanex_train" / "configs" / "seed_templates_cache.json"
            else:
                cache_file = Path(__file__).parent.parent.parent / "configs" / "seed_templates_cache.json"

        self.schemas_dir = str(schemas_dir)
        self.cache_file = str(cache_file)

        # Load or generate templates
        if auto_generate:
            self._load_templates()
        else:
            print("⚠️ Auto-generation disabled. No templates loaded.")

    def _load_templates(self):
        """Load templates from cache or generate from schemas."""
        try:
            # Load templates from cache or generate
            raw_templates = load_templates_from_schemas(
                schemas_dir=self.schemas_dir,
                cache_file=self.cache_file,
                force_regenerate=False,
                templates_per_config=1
            )

            # Convert raw templates to PromptTemplate objects
            self._convert_to_prompt_templates(raw_templates)

            total_templates = sum(len(templates) for templates in self.templates.values())
            print(f"✅ Loaded {total_templates} templates for {len(self.templates)} tool configurations")

        except Exception as e:
            print(f"⚠️ Could not load/generate templates: {e}")
            print("⚠️ Template system will have empty templates")
            self.templates = {}

    def _convert_to_prompt_templates(self, raw_templates: Dict[str, Any]):
        """Convert raw template data to PromptTemplate objects."""
        for tool_name, tool_configs in raw_templates.items():
            for config_key, template_list in tool_configs.items():
                # Parse config key: toolname_complexity_style[_edge]
                parts = config_key.split('_')

                # Extract complexity and style
                try:
                    # Find complexity
                    complexity = None
                    for part in parts:
                        try:
                            complexity = ComplexityLevel(part)
                            break
                        except ValueError:
                            continue

                    # Find style
                    style = None
                    for part in parts:
                        try:
                            style = PromptStyle(part)
                            break
                        except ValueError:
                            continue

                    # Check if edge case
                    is_edge = 'edge' in parts

                    if not complexity:
                        complexity = ComplexityLevel.MEDIUM
                    if not style:
                        style = PromptStyle.DIRECT

                except Exception as e:
                    print(f"⚠️ Could not parse config key '{config_key}': {e}")
                    continue

                # Convert each template
                for idx, raw_template in enumerate(template_list):
                    template_id = f"{config_key}_{idx}"

                    prompt_template = PromptTemplate(
                        id=template_id,
                        tool_name=tool_name,
                        template=raw_template.get("template", ""),
                        complexity=complexity,
                        style=style,
                        is_edge_case=is_edge,
                        edge_case_type=EdgeCaseType.AMBIGUOUS if is_edge else None,
                        required_entities=raw_template.get("required_entities", []),
                        optional_entities=raw_template.get("optional_entities", [])
                    )

                    # Add to registry
                    if tool_name not in self.templates:
                        self.templates[tool_name] = []

                    self.templates[tool_name].append(prompt_template)

    def get_templates(self, tool_name: str) -> List[PromptTemplate]:
        """Get all templates for a specific tool."""
        return self.templates.get(tool_name, [])

    def get_templates_by_complexity(
        self,
        tool_name: str,
        complexity: ComplexityLevel
    ) -> List[PromptTemplate]:
        """Get templates filtered by complexity."""
        return [
            t for t in self.get_templates(tool_name)
            if t.complexity == complexity
        ]

    def get_templates_by_style(
        self,
        tool_name: str,
        style: PromptStyle
    ) -> List[PromptTemplate]:
        """Get templates filtered by style."""
        return [
            t for t in self.get_templates(tool_name)
            if t.style == style
        ]

    def get_edge_case_templates(self, tool_name: str) -> List[PromptTemplate]:
        """Get edge case templates for a tool."""
        return [t for t in self.get_templates(tool_name) if t.is_edge_case]

    def sample_template(
        self,
        tool_name: str,
        complexity: Optional[ComplexityLevel] = None,
        style: Optional[PromptStyle] = None,
        edge_case: bool = False
    ) -> Optional[PromptTemplate]:
        """Sample a random template with optional filters."""
        templates = self.get_templates(tool_name)

        if not templates:
            return None

        # Apply filters
        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        if style:
            templates = [t for t in templates if t.style == style]
        if edge_case:
            templates = [t for t in templates if t.is_edge_case]
        else:
            templates = [t for t in templates if not t.is_edge_case]

        return random.choice(templates) if templates else None

    def fill_template(
        self,
        template: PromptTemplate,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Fill a template with entity values.

        Returns:
            Tuple of (filled_prompt, entities_used)
        """
        if context is None:
            context = EntityGenerator.generate_context()

        # Add sales rep info
        if "sales_rep" in str(template.template):
            rep_id, rep_name = context.get("sales_rep", EntityGenerator.generate_sales_rep())
            context["sales_rep_id"] = rep_id
            context["sales_rep_name"] = rep_name

        # Add derived values
        if "min_value" not in context:
            context["min_value"] = random.randint(10000, 50000)
        if "max_value" not in context:
            context["max_value"] = context.get("min_value", 50000) + random.randint(50000, 200000)
        if "limit" not in context:
            context["limit"] = random.choice([5, 10, 20, 25, 50])
        if "duration" not in context:
            context["duration"] = random.choice([15, 30, 45, 60, 90])
        if "reminder" not in context:
            context["reminder"] = random.choice([5, 10, 15, 30])
        if "notes" not in context:
            notes_options = [
                "Great conversation, very interested in our solution.",
                "Need to follow up with more technical details.",
                "Decision maker, fast-moving deal.",
                "Budget approved for Q1.",
                "Interested but evaluating competitors.",
                "Hot lead, move fast!",
                "Requested pricing proposal."
            ]
            context["notes"] = random.choice(notes_options)
        if "reason" not in context:
            reason_options = [
                "better geographic coverage",
                "they have experience with this industry",
                "previous relationship with the company",
                "expertise in enterprise deals",
                "availability and capacity",
                "language skills match"
            ]
            context["reason"] = random.choice(reason_options)
        if "description" not in context:
            desc_options = [
                "Discussed their needs and timeline.",
                "They're interested in our enterprise plan.",
                "Went over the proposal details.",
                "Answered their technical questions.",
                "They want a demo next week.",
                "Positive conversation, moving forward.",
                "Need to address some concerns about implementation."
            ]
            context["description"] = random.choice(desc_options)
        if "outcome" not in context:
            context["outcome"] = random.choice(["Positive", "Neutral", "Negative", "No Response"])
        if "priority" not in context:
            context["priority"] = random.choice(["Low", "Medium", "High", "Critical"])
        if "time" not in context:
            context["time"] = EntityGenerator.generate_time()

        # Fill the template
        prompt = template.template
        entities_used: Dict[str, Any] = {}

        # Replace {{placeholder}} format
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            placeholder_single = "{" + key + "}"

            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
                entities_used[key] = value
            elif placeholder_single in prompt:
                prompt = prompt.replace(placeholder_single, str(value))
                entities_used[key] = value

        return prompt, entities_used

    def generate_prompt(
        self,
        tool_name: str,
        complexity: Optional[ComplexityLevel] = None,
        style: Optional[PromptStyle] = None,
        edge_case: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, Dict[str, Any], PromptTemplate]]:
        """
        Generate a complete prompt from templates.

        Returns:
            Tuple of (prompt_text, entities_used, template_used) or None
        """
        template = self.sample_template(tool_name, complexity, style, edge_case)
        if not template:
            return None

        prompt, entities = self.fill_template(template, context)
        return prompt, entities, template

    def get_all_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self.templates.keys())

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded templates."""
        stats = {}
        for tool_name, templates in self.templates.items():
            stats[tool_name] = {
                "total": len(templates),
                "by_complexity": {
                    "simple": len([t for t in templates if t.complexity == ComplexityLevel.SIMPLE]),
                    "medium": len([t for t in templates if t.complexity == ComplexityLevel.MEDIUM]),
                    "complex": len([t for t in templates if t.complexity == ComplexityLevel.COMPLEX]),
                },
                "edge_cases": len([t for t in templates if t.is_edge_case]),
                "by_style": {}
            }
            for style in PromptStyle:
                count = len([t for t in templates if t.style == style])
                if count > 0:
                    stats[tool_name]["by_style"][style.value] = count
        return stats

    def reload_templates(self, force_regenerate: bool = False):
        """
        Reload templates from cache or regenerate.

        Args:
            force_regenerate: If True, regenerate from schemas even if cache exists
        """
        if force_regenerate:
            # Clear cache
            cache_path = Path(self.cache_file)
            if cache_path.exists():
                cache_path.unlink()
                print("🗑️ Cleared template cache")

        self.templates = {}
        self._load_templates()
        print(f"✅ Reloaded templates for {len(self.templates)} tools")
