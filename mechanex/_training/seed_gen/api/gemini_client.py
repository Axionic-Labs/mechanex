"""
Gemini API Client for Teacher Model

This module provides the interface to Google's Gemini API for seed prompt generation
and validation. Uses the new google-genai package.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
import time


@dataclass
class GeminiConfig:
    """Configuration for the LLM API client."""
    # API key - uses GOOGLE_API_KEY for Google, or respective keys for other providers
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    provider: str = "google"  # google, openai, anthropic
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192

    # Rate limiting
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_min_wait: int = 1
    retry_max_wait: int = 60


class GenerationResult(BaseModel):
    """Result from a generation request."""
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class GeminiClient:
    """
    Client for interacting with Google's Gemini API.

    Provides methods for:
    - Generating diverse seed prompts
    - Validating prompt quality
    - Batch generation with rate limiting
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()

        # Initialize the client with API key (optional - allows template-only mode)
        if self.config.api_key:
            self.client = genai.Client(api_key=self.config.api_key)
        else:
            self.client = None
            # Don't raise - allow template-only mode without API key

        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 60.0 / self.config.requests_per_minute

    def _rate_limit_sync(self):
        """Enforce rate limiting between requests (sync version)."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    async def _rate_limit(self):
        """Enforce rate limiting between requests (async version)."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request

        Returns:
            GenerationResult with the generated text
        """
        await self._rate_limit()

        try:
            # Create generation config
            gen_config = types.GenerateContentConfig(
                temperature=temperature or self.config.temperature,
                max_output_tokens=max_tokens or self.config.max_output_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            # Use asyncio to run the sync API in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=gen_config
                )
            )

            # Extract text from response
            text = response.text if response.text else ""

            return GenerationResult(
                text=text,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
                success=True
            )

        except Exception as e:
            return GenerationResult(
                text="",
                success=False,
                error=str(e)
            )

    def generate_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Synchronous version of generate."""
        self._rate_limit_sync()

        try:
            # Create generation config
            gen_config = types.GenerateContentConfig(
                temperature=temperature or self.config.temperature,
                max_output_tokens=max_tokens or self.config.max_output_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=gen_config
            )

            # Extract text from response
            text = response.text if response.text else ""

            return GenerationResult(
                text=text,
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
                success=True
            )

        except Exception as e:
            return GenerationResult(
                text="",
                success=False,
                error=str(e)
            )

    async def generate_seed_prompts(
        self,
        tool_schema: Dict[str, Any],
        num_prompts: int = 10,
        complexity_distribution: Optional[Dict[str, float]] = None,
        edge_case_percentage: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse seed prompts for a tool using the teacher model.

        Args:
            tool_schema: The tool schema to generate prompts for
            num_prompts: Number of prompts to generate
            complexity_distribution: Distribution of complexity levels
            edge_case_percentage: Percentage of edge case prompts

        Returns:
            List of generated prompts with metadata
        """
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.3,
                "medium": 0.5,
                "complex": 0.2
            }

        prompt = self._build_seed_generation_prompt(
            tool_schema,
            num_prompts,
            complexity_distribution,
            edge_case_percentage
        )

        result = await self.generate(prompt, temperature=0.9)

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")

        # Parse the JSON response
        return self._parse_seed_prompts(result.text, tool_schema["name"])

    def _build_seed_generation_prompt(
        self,
        tool_schema: Dict[str, Any],
        num_prompts: int,
        complexity_distribution: Dict[str, float],
        edge_case_percentage: float
    ) -> str:
        """Build the prompt for seed generation."""

        num_simple = int(num_prompts * complexity_distribution.get("simple", 0.3))
        num_medium = int(num_prompts * complexity_distribution.get("medium", 0.5))
        num_complex = num_prompts - num_simple - num_medium
        num_edge_cases = int(num_prompts * edge_case_percentage)

        # Extract parameter info for examples
        params = tool_schema.get("parameters", {}).get("properties", {})
        required = tool_schema.get("parameters", {}).get("required", [])
        tool_name = tool_schema.get("name", "unknown")
        tool_desc = tool_schema.get("description", "")
        
        # Build example entities from schema
        example_entities = {}
        for param_name in list(params.keys())[:3]:
            param_type = params[param_name].get("type", "string")
            if param_type == "string":
                example_entities[param_name] = f"example_{param_name}"
            elif param_type == "integer":
                example_entities[param_name] = 123
            elif param_type == "number":
                example_entities[param_name] = 99.99
            elif param_type == "boolean":
                example_entities[param_name] = True

        return f"""You are generating diverse, realistic user prompts for training a tool-calling AI model.

TOOL DEFINITION:
```json
{json.dumps(tool_schema, indent=2)}
```

TASK:
Generate exactly {num_prompts} unique user prompts that would require using this tool.

REQUIREMENTS:
1. Each prompt must be a realistic scenario appropriate for this tool's domain and purpose
2. Include specific, grounded entities that map to the tool's parameters
3. Adapt language, terminology, and examples to match this tool's domain
4. Vary the complexity:
   - {num_simple} simple prompts (direct requests, 1-2 parameters)
   - {num_medium} medium prompts (contextual, 3-4 parameters)
   - {num_complex} complex prompts (multi-part scenarios, 5+ parameters or nuanced context)
5. Include {num_edge_cases} edge cases (ambiguous wording, informal language, boundary values)
6. Vary the style: direct, conversational, formal, urgent, contextual

DIVERSITY GUIDELINES:
- Use realistic values appropriate for each parameter type
- Follow any enum constraints or range limits from the schema
- Use diverse names, identifiers, and values that make sense for this domain
- Include varied scenarios that would naturally use this tool

OUTPUT FORMAT:
Return a valid JSON array. Each object must have:
- "prompt": The user request text
- "expected_entities": Object mapping parameter names to expected values
- "complexity": "simple" | "medium" | "complex"
- "is_edge_case": boolean
- "edge_case_type": null | "ambiguous" | "informal" | "boundary" | "missing_info"

Example output structure for this tool:
```json
[
  {{
    "prompt": "A realistic user request for {tool_name}...",
    "expected_entities": {json.dumps(example_entities)},
    "complexity": "medium",
    "is_edge_case": false,
    "edge_case_type": null
  }}
]
```

Generate the {num_prompts} prompts now. Output ONLY the JSON array, no other text."""

    def _parse_seed_prompts(
        self,
        response_text: str,
        tool_name: str
    ) -> List[Dict[str, Any]]:
        """Parse the generated seed prompts from the response."""
        # Clean up the response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            prompts = json.loads(text)

            # Add tool name to each prompt
            for prompt in prompts:
                prompt["tool_name"] = tool_name

            return prompts

        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                try:
                    prompts = json.loads(json_match.group())
                    for prompt in prompts:
                        prompt["tool_name"] = tool_name
                    return prompts
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Failed to parse seed prompts: {e}\nResponse: {text[:500]}...")

    async def validate_prompt(
        self,
        prompt: str,
        tool_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a generated prompt for quality using the teacher model.

        Args:
            prompt: The prompt to validate
            tool_schema: The tool schema for context

        Returns:
            Validation result with scores and feedback
        """
        validation_prompt = f"""Evaluate this tool-calling prompt for quality.

TOOL SCHEMA:
```json
{json.dumps(tool_schema, indent=2)}
```

PROMPT TO EVALUATE:
"{prompt}"

EVALUATION CRITERIA:
1. Grammar (0-1): Is the prompt grammatically correct?
2. Clarity (0-1): Is the intent clear and unambiguous?
3. Grounding (0-1): Are the entity values realistic and specific for this tool's domain?
4. Relevance (0-1): Does the prompt appropriately match the tool's purpose?
5. Naturalness (0-1): Does it sound like a real user request?

OUTPUT FORMAT (JSON only):
```json
{{
  "grammar_score": 0.0-1.0,
  "clarity_score": 0.0-1.0,
  "grounding_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "naturalness_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "is_valid": true/false,
  "issues": ["list of any issues found"],
  "feedback": "brief feedback"
}}
```

Output ONLY the JSON, no other text."""

        result = await self.generate(validation_prompt, temperature=0.1)

        if not result.success:
            return {
                "overall_score": 0.0,
                "is_valid": False,
                "error": result.error
            }

        # Parse validation result
        try:
            text = result.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())

        except json.JSONDecodeError:
            return {
                "overall_score": 0.5,
                "is_valid": True,
                "parse_error": "Could not parse validation response"
            }

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        concurrency: int = 5
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts with controlled concurrency.

        Args:
            prompts: List of prompts to generate from
            temperature: Temperature for all generations
            concurrency: Maximum concurrent requests

        Returns:
            List of GenerationResults
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(prompt: str) -> GenerationResult:
            async with semaphore:
                return await self.generate(prompt, temperature)

        tasks = [generate_with_semaphore(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def test_connection(self) -> bool:
        """Test the API connection (synchronous)."""
        try:
            result = self.generate_sync("Say 'hello' in one word.", max_tokens=10)
            return result.success
        except Exception:
            return False
