"""Reward function compiler for ARA."""

import logging
from datetime import datetime
from typing import Dict, Optional

from ..parsing.canonical import CanonicalSchema
from ..clients.teacher import TeacherClient
from ..types.enums import TaskType
from ..types.config import CompilerConfig, REWARD_WEIGHTS
from ..types.results import GeneratedReward, CompilationMetadata
from .prompts import PromptBuilder
from .extractor import CodeExtractor, CodeExtractionError

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    """Error during reward compilation."""
    pass


class RewardCompiler:
    """
    Compiles tool schemas into executable reward functions.

    Uses Teacher models to generate Python code for reward evaluation.
    """

    def __init__(
        self,
        teacher_client: TeacherClient,
        config: Optional[CompilerConfig] = None
    ):
        self.teacher = teacher_client
        self.config = config or CompilerConfig()
        self.prompt_builder = PromptBuilder()
        self.code_extractor = CodeExtractor()

    def compile(
        self,
        schema: CanonicalSchema,
        task_type: Optional[TaskType] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        pass_threshold: float = 0.7
    ) -> GeneratedReward:
        """
        Compile schema into executable reward function.

        Args:
            schema: Canonical tool schema
            task_type: Hint for weight configuration
            custom_weights: Override default component weights
            pass_threshold: Threshold for passing score

        Returns:
            GeneratedReward: Generated reward code with metadata
        """
        task_type = task_type or TaskType.GENERAL
        weights = custom_weights or REWARD_WEIGHTS.get(task_type, REWARD_WEIGHTS[TaskType.GENERAL])

        # Build compilation prompt
        prompt = self.prompt_builder.build(
            schema=schema,
            task_type=task_type,
            weights=weights,
            pass_threshold=pass_threshold
        )

        # Try compilation with retries
        last_error = None
        for attempt in range(self.config.retry_on_failure + 1):
            try:
                # Generate reward code via Teacher
                raw_response = self.teacher.generate(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                # Extract Python code from response
                reward_code = self.code_extractor.extract(raw_response)

                return GeneratedReward(
                    schema=schema,
                    code=reward_code,
                    weights=weights,
                    metadata=CompilationMetadata(
                        teacher_model=self.teacher.model_name,
                        compilation_time=datetime.now(),
                        prompt_hash=hash(prompt)
                    )
                )

            except CodeExtractionError as e:
                last_error = e
                logger.warning(f"Compilation attempt {attempt + 1} failed: {e}")
                logger.debug(f"Raw response from teacher (first 500 chars): {raw_response[:500]}")
                continue

        # Log the final failed response for debugging
        logger.error(f"All compilation attempts failed. Last error: {last_error}")
        raise CompilationError(f"Failed to compile reward after {self.config.retry_on_failure + 1} attempts: {last_error}")
