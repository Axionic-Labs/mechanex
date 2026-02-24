"""
Seed Generator Module

This module implements the seed generation pipeline that uses the teacher model
to generate diverse, realistic tool-calling prompts for any domain.

Implements TICKET-2.1 requirements:
- Prompt template system with domain-specific variations
- Generation across all configured tools
- Diversity via temperature sampling and template rotation
- Structured output with metadata
- Minimum 1000 seed prompts for POC
- Grammar and semantic validation
"""
import asyncio
import json
import random
import os
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from ..api.gemini_client import GeminiClient, GeminiConfig
from ..models.tool_schemas import CRM_TOOLS, ToolSchema
from ..models.seed_prompt import (
    SeedPrompt,
    SeedBatch,
    PromptMetadata,
    ExpectedEntities,
    GenerationStats
)
from .prompt_templates import (
    PromptTemplateSystem,
    ComplexityLevel,
    PromptStyle,
    EntityGenerator
)

# Try to import automated template system, fall back to manual if not available
try:
    from .prompt_templates_auto import AutoPromptTemplateSystem
    USE_AUTO_TEMPLATES = True
except ImportError:
    USE_AUTO_TEMPLATES = False
    print("AutoPromptTemplateSystem not available, using manual templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SeedGeneratorConfig:
    """Configuration for the Seed Generator."""

    # Target counts
    num_prompts: int = 1000
    prompts_per_tool: Optional[int] = None  # If None, distribute evenly

    # Complexity distribution
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.30,
        "medium": 0.50,
        "complex": 0.20
    })

    # Edge case percentage
    edge_case_percentage: float = 0.15

    # Style distribution (optional - for diversity)
    style_weights: Dict[str, float] = field(default_factory=lambda: {
        "direct": 0.25,
        "conversational": 0.25,
        "formal": 0.15,
        "urgent": 0.10,
        "contextual": 0.25
    })

    # Generation settings
    teacher_temperature: float = 0.9
    batch_size: int = 25  # Prompts per API call
    max_retries: int = 3
    concurrent_batches: int = 3

    # Validation settings
    validate_prompts: bool = True
    validation_threshold: float = 0.7  # Minimum quality score to accept
    validation_sample_rate: float = 0.2  # Sample rate for validation (to save API calls)

    # Output settings
    output_dir: str = "output"
    output_format: Literal["json", "jsonl", "both"] = "both"

    # Diversity settings
    use_template_rotation: bool = True
    use_teacher_generation: bool = True  # Also use teacher to generate prompts
    teacher_generation_ratio: float = 0.4  # 40% from teacher, 60% from templates

    # Template system settings
    use_auto_templates: bool = True  # Use LLM-generated templates from schemas
    auto_template_cache: Optional[str] = None  # Path to template cache


class SeedGenerator:
    """
    Seed Generator for creating diverse tool-calling prompts.

    Combines template-based generation with teacher model generation
    to create a diverse dataset of training prompts for any domain.
    Works with any tool schemas provided in configs/tool_schemas/.
    """

    def __init__(
        self,
        config: Optional[SeedGeneratorConfig] = None,
        gemini_config: Optional[GeminiConfig] = None,
        tools: Optional[List[ToolSchema]] = None,
        schemas_dir: Optional[str] = None
    ):
        self.config = config or SeedGeneratorConfig()
        self.gemini_client = GeminiClient(gemini_config)
        self.tools = tools or CRM_TOOLS
        self.schemas_dir = schemas_dir

        # Initialize template system (auto or manual)
        if self.config.use_auto_templates and USE_AUTO_TEMPLATES:
            logger.info("Using automated template system (LLM-generated from schemas)")
            self.template_system = AutoPromptTemplateSystem(
                schemas_dir=self.schemas_dir,
                cache_file=self.config.auto_template_cache
            )
        else:
            logger.info("Using manual template system")
            self.template_system = PromptTemplateSystem()

        self.entity_generator = EntityGenerator()
        self.stats = GenerationStats()

        # Track used templates for rotation
        self._template_usage: Dict[str, int] = {}

    @staticmethod
    def load_schemas_from_mixed(
        sources: List[Any],
        teacher_client: Optional[GeminiClient] = None
    ) -> List[ToolSchema]:
        """
        Load schemas from a mixed list of sources (files, strings, dicts).
        
        Args:
            sources: List of sources (Path objects, JSON strings, YAML strings, dicts, etc.)
            teacher_client: Optional client for NL parsing
            
        Returns:
            List[ToolSchema]: Normalized list of tool schemas
        """
        from ..parsing import parse_schema
        tools = []
        
        for source in sources:
            try:
                # If it's a file path
                if isinstance(source, (str, Path)) and os.path.exists(source):
                    with open(source, 'r') as f:
                        if str(source).endswith('.json'):
                            data = json.load(f)
                        else:
                            # Handle YAML or other (requires PyYAML if truly mixed)
                            data = f.read()
                        source = data

                canonical = parse_schema(source, teacher_client=teacher_client)
                tools.append(ToolSchema.from_canonical(canonical))
            except Exception as e:
                logger.error(f"Failed to load schema from {source}: {e}")
                
        return tools

    async def generate(self) -> SeedBatch:
        """
        Main generation method. Creates the full seed dataset.

        Returns:
            SeedBatch containing all generated prompts
        """
        start_time = datetime.now()
        logger.info(f"Starting seed generation: {self.config.num_prompts} prompts")

        all_prompts: List[SeedPrompt] = []

        # Calculate prompts per tool
        tools = self.tools
        if not tools:
            logger.error("No tools provided to SeedGenerator. Cannot generate prompts.")
            return SeedBatch(prompts=[], stats=self.stats)

        prompts_per_tool = self.config.prompts_per_tool or (self.config.num_prompts // len(tools))
        remaining = self.config.num_prompts - (prompts_per_tool * len(tools))

        logger.info(f"Generating {prompts_per_tool} prompts per tool ({len(tools)} tools)")

        for i, tool in enumerate(tools):
            # Add remaining prompts to the last tool
            target_count = prompts_per_tool + (remaining if i == len(tools) - 1 else 0)

            logger.info(f"Generating prompts for tool: {tool.name} (target: {target_count})")

            tool_prompts = await self._generate_for_tool(tool, target_count)
            all_prompts.extend(tool_prompts)

            logger.info(f"Generated {len(tool_prompts)} prompts for {tool.name}")

        # Create batch
        batch = SeedBatch(
            prompts=all_prompts,
            generation_config={
                "num_prompts": self.config.num_prompts,
                "complexity_distribution": self.config.complexity_distribution,
                "edge_case_percentage": self.config.edge_case_percentage,
                "teacher_temperature": self.config.teacher_temperature,
                "template_rotation": self.config.use_template_rotation,
                "teacher_generation_ratio": self.config.teacher_generation_ratio
            }
        )

        # Update stats
        self.stats.update_from_batch(batch)
        self.stats.generation_duration_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(f"Generation complete. Total: {batch.total_count}, Valid: {batch.valid_count}")
        logger.info(f"Duration: {self.stats.generation_duration_seconds:.2f}s")

        return batch

    async def _generate_for_tool(
        self,
        tool: ToolSchema,
        target_count: int
    ) -> List[SeedPrompt]:
        """Generate prompts for a specific tool."""
        prompts: List[SeedPrompt] = []

        # Split between template and teacher generation
        if self.config.use_teacher_generation:
            teacher_count = int(target_count * self.config.teacher_generation_ratio)
            template_count = target_count - teacher_count
        else:
            template_count = target_count
            teacher_count = 0

        # Generate from templates
        if template_count > 0:
            template_prompts = await self._generate_from_templates(tool, template_count)
            prompts.extend(template_prompts)
            logger.info(f"  Template generation: {len(template_prompts)} prompts")

        # Generate from teacher
        if teacher_count > 0:
            teacher_prompts = await self._generate_from_teacher(tool, teacher_count)
            prompts.extend(teacher_prompts)
            logger.info(f"  Teacher generation: {len(teacher_prompts)} prompts")

        # Validate if enabled
        if self.config.validate_prompts:
            prompts = await self._validate_prompts(prompts, tool)

        return prompts

    async def _generate_from_templates(
        self,
        tool: ToolSchema,
        count: int
    ) -> List[SeedPrompt]:
        """Generate prompts using the template system."""
        prompts: List[SeedPrompt] = []

        # Calculate distribution
        num_simple = int(count * self.config.complexity_distribution.get("simple", 0.3))
        num_medium = int(count * self.config.complexity_distribution.get("medium", 0.5))
        num_complex = count - num_simple - num_medium
        num_edge_cases = int(count * self.config.edge_case_percentage)

        complexity_counts = {
            ComplexityLevel.SIMPLE: num_simple,
            ComplexityLevel.MEDIUM: num_medium,
            ComplexityLevel.COMPLEX: num_complex
        }

        # Generate regular prompts
        for complexity, target in complexity_counts.items():
            for _ in range(target):
                # Optionally rotate styles
                style = self._sample_style() if self.config.use_template_rotation else None

                result = self.template_system.generate_prompt(
                    tool_name=tool.name,
                    complexity=complexity,
                    style=style,
                    edge_case=False
                )

                # Fallback: if no template found with specific complexity/style, try without constraints
                if not result:
                    result = self.template_system.generate_prompt(
                        tool_name=tool.name,
                        complexity=complexity,
                        edge_case=False
                    )
                if not result:
                    result = self.template_system.generate_prompt(
                        tool_name=tool.name,
                        edge_case=False
                    )

                if result:
                    prompt_text, entities, template = result
                    seed_prompt = self._create_seed_prompt(
                        prompt_text=prompt_text,
                        entities=entities,
                        tool_name=tool.name,
                        complexity=complexity.value,
                        is_edge_case=False,
                        template_id=template.id
                    )
                    prompts.append(seed_prompt)
                    self._track_template_usage(template.id)

        # Generate edge cases
        for _ in range(num_edge_cases):
            result = self.template_system.generate_prompt(
                tool_name=tool.name,
                edge_case=True
            )

            if result:
                prompt_text, entities, template = result
                seed_prompt = self._create_seed_prompt(
                    prompt_text=prompt_text,
                    entities=entities,
                    tool_name=tool.name,
                    complexity=template.complexity.value,
                    is_edge_case=True,
                    edge_case_type=template.edge_case_type.value if template.edge_case_type else None,
                    template_id=template.id
                )
                prompts.append(seed_prompt)
                self._track_template_usage(template.id)

        return prompts

    async def _generate_from_teacher(
        self,
        tool: ToolSchema,
        count: int
    ) -> List[SeedPrompt]:
        """Generate prompts using the teacher model."""
        prompts: List[SeedPrompt] = []
        tool_schema = tool.to_json_schema()

        # Generate in batches
        remaining = count
        while remaining > 0:
            batch_size = min(self.config.batch_size, remaining)

            try:
                generated = await self.gemini_client.generate_seed_prompts(
                    tool_schema=tool_schema,
                    num_prompts=batch_size,
                    complexity_distribution=self.config.complexity_distribution,
                    edge_case_percentage=self.config.edge_case_percentage
                )

                for item in generated:
                    seed_prompt = self._create_seed_prompt(
                        prompt_text=item.get("prompt", ""),
                        entities=item.get("expected_entities", {}),
                        tool_name=tool.name,
                        complexity=item.get("complexity", "medium"),
                        is_edge_case=item.get("is_edge_case", False),
                        edge_case_type=item.get("edge_case_type"),
                        template_id=None  # Teacher-generated
                    )
                    prompts.append(seed_prompt)

                remaining -= len(generated)
                logger.debug(f"  Generated batch of {len(generated)} prompts, {remaining} remaining")

            except Exception as e:
                logger.error(f"Error generating from teacher: {e}")
                # Fall back to template generation
                fallback = await self._generate_from_templates(tool, batch_size)
                prompts.extend(fallback)
                remaining -= len(fallback)

        return prompts

    async def _validate_prompts(
        self,
        prompts: List[SeedPrompt],
        tool: ToolSchema
    ) -> List[SeedPrompt]:
        """Validate prompts for quality."""
        if not prompts:
            return prompts

        tool_schema = tool.to_json_schema()

        # Sample prompts for validation
        sample_size = max(1, int(len(prompts) * self.config.validation_sample_rate))
        samples = random.sample(prompts, min(sample_size, len(prompts)))

        validation_results = {}

        for prompt in samples:
            try:
                result = await self.gemini_client.validate_prompt(
                    prompt=prompt.prompt,
                    tool_schema=tool_schema
                )
                validation_results[prompt.id] = result

            except Exception as e:
                print(f"Validation failed for {prompt.id}: {e}")
                validation_results[prompt.id] = {"overall_score": 0.5, "is_valid": True}

        # Update prompts with validation results
        for prompt in prompts:
            if prompt.id in validation_results:
                result = validation_results[prompt.id]
                prompt.quality_score = result.get("overall_score", 0.5)
                prompt.is_valid = result.get("is_valid", True)

                if prompt.quality_score < self.config.validation_threshold:
                    prompt.is_valid = False
                    prompt.validation_errors = result.get("issues", [])
            else:
                # Non-sampled prompts get default score
                prompt.quality_score = 0.8
                prompt.is_valid = True

        return prompts

    def _create_seed_prompt(
        self,
        prompt_text: str,
        entities: Dict[str, Any],
        tool_name: str,
        complexity: str,
        is_edge_case: bool,
        edge_case_type: Optional[str] = None,
        template_id: Optional[str] = None
    ) -> SeedPrompt:
        """Create a SeedPrompt object from raw data."""
        metadata = PromptMetadata(
            tool_name=tool_name,
            tool_coverage=list(entities.keys()),
            complexity=complexity,
            is_edge_case=is_edge_case,
            edge_case_type=edge_case_type,
            template_id=template_id,
            temperature=self.config.teacher_temperature
        )

        return SeedPrompt(
            prompt=prompt_text,
            expected_entities=ExpectedEntities(entities=entities),
            metadata=metadata
        )

    def _sample_style(self) -> PromptStyle:
        """Sample a prompt style based on configured weights."""
        styles = list(self.config.style_weights.keys())
        weights = list(self.config.style_weights.values())

        style_name = random.choices(styles, weights=weights, k=1)[0]
        return PromptStyle(style_name)

    def _track_template_usage(self, template_id: str):
        """Track template usage for rotation."""
        self._template_usage[template_id] = self._template_usage.get(template_id, 0) + 1

    def save_batch(
        self,
        batch: SeedBatch,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save the generated batch to files.

        Returns:
            Dictionary of format -> filepath
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        if self.config.output_format in ["json", "both"]:
            json_path = output_dir / f"seed_prompts_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(batch.model_dump(), f, indent=2, default=str)
            saved_files["json"] = str(json_path)
            logger.info(f"Saved JSON to: {json_path}")

        if self.config.output_format in ["jsonl", "both"]:
            jsonl_path = output_dir / f"seed_prompts_{timestamp}.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(batch.to_jsonl())
            saved_files["jsonl"] = str(jsonl_path)
            logger.info(f"Saved JSONL to: {jsonl_path}")

        # Save stats
        stats_path = output_dir / f"generation_stats_{timestamp}.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats.model_dump(), f, indent=2)
        saved_files["stats"] = str(stats_path)

        return saved_files

    def get_stats(self) -> GenerationStats:
        """Get generation statistics."""
        return self.stats

    def print_stats(self):
        """Print generation statistics to console."""
        print("\n" + "=" * 50)
        print("SEED GENERATION STATISTICS")
        print("=" * 50)
        print(f"Total Generated: {self.stats.total_generated}")
        print(f"Valid: {self.stats.total_valid} ({self.stats.valid_rate:.1%})")
        print(f"Invalid: {self.stats.total_invalid}")
        print(f"\nBy Tool:")
        for tool, count in self.stats.by_tool.items():
            print(f"  {tool}: {count}")
        print(f"\nBy Complexity:")
        for complexity, count in self.stats.by_complexity.items():
            print(f"  {complexity}: {count}")
        print(f"\nEdge Cases: {self.stats.edge_case_count}")
        if self.stats.average_quality_score:
            print(f"Average Quality Score: {self.stats.average_quality_score:.2f}")
        if self.stats.generation_duration_seconds:
            print(f"Duration: {self.stats.generation_duration_seconds:.2f}s")
        print("=" * 50 + "\n")


async def generate_seeds(
    num_prompts: int = 1000,
    output_dir: str = "output",
    validate: bool = True
) -> SeedBatch:
    """
    Convenience function to generate seed prompts.

    Args:
        num_prompts: Number of prompts to generate
        output_dir: Directory to save output
        validate: Whether to validate prompts

    Returns:
        SeedBatch with generated prompts
    """
    config = SeedGeneratorConfig(
        num_prompts=num_prompts,
        output_dir=output_dir,
        validate_prompts=validate
    )

    generator = SeedGenerator(config)
    batch = await generator.generate()
    generator.save_batch(batch)
    generator.print_stats()

    return batch
