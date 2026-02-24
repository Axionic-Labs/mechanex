"""Base class for generated reward functions."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple
import re
import json

from .result import RewardResult


class MXRewardBase(ABC):
    """
    Base class for all generated reward functions.

    Subclasses must implement:
    - _check_format(response) -> float
    - _check_grounding(response, prompt) -> float
    - _check_types(response) -> float
    - _check_reasoning(response) -> float
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        pass_threshold: float = 0.7
    ):
        self.schema = schema
        self.weights = weights or {
            'format': 0.2,
            'grounding': 0.3,
            'type_valid': 0.3,
            'reasoning': 0.2
        }
        self.pass_threshold = pass_threshold

    def __call__(self, response: str, prompt: str) -> RewardResult:
        """
        Evaluate a model response.

        Args:
            response: Model's complete response (with <think> and <tool_call>)
            prompt: Original user prompt

        Returns:
            RewardResult with score and breakdown
        """
        scores = {}
        details = {}

        # Run all checks
        scores['format'], details['format'] = self._check_format_with_details(response)
        scores['grounding'], details['grounding'] = self._check_grounding_with_details(response, prompt)
        scores['type_valid'], details['type_valid'] = self._check_types_with_details(response)
        scores['reasoning'], details['reasoning'] = self._check_reasoning_with_details(response)

        # Compute weighted composite
        final_score = sum(
            self.weights.get(k, 0) * v
            for k, v in scores.items()
        )

        return RewardResult(
            score=final_score,
            breakdown=scores,
            passed=final_score >= self.pass_threshold,
            details=details
        )

    # === Format Checking ===

    def _check_format_with_details(self, response: str) -> Tuple[float, Dict]:
        """Check format and return details."""
        score = self._check_format(response)
        details = {
            'has_think': '<think>' in response and '</think>' in response,
            'has_tool_call': '<tool_call>' in response and '</tool_call>' in response,
            'correct_order': self._check_tag_order(response)
        }
        return score, details

    @abstractmethod
    def _check_format(self, response: str) -> float:
        """Validate structural format. Override in generated class."""
        pass

    def _check_tag_order(self, response: str) -> bool:
        """Check that <think> comes before <tool_call>."""
        think_pos = response.find('<think>')
        tool_pos = response.find('<tool_call>')
        return think_pos >= 0 and tool_pos >= 0 and think_pos < tool_pos

    # === Grounding Checking ===

    def _check_grounding_with_details(self, response: str, prompt: str) -> Tuple[float, Dict]:
        """Check grounding and return details."""
        score = self._check_grounding(response, prompt)
        tool_call = self._extract_tool_call(response)
        details = {
            'extracted_entities': self._extract_entities(prompt),
            'argument_grounding': self._check_argument_grounding(tool_call, prompt)
        }
        return score, details

    @abstractmethod
    def _check_grounding(self, response: str, prompt: str) -> float:
        """Validate argument grounding. Override in generated class."""
        pass

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        entities = []

        # IDs (alphanumeric with dashes/underscores)
        entities.extend(re.findall(r'\b[A-Z]{2,}-\d+\b', text))
        entities.extend(re.findall(r'\b[a-z]+_[a-z0-9_]+\b', text))

        # Quoted strings
        entities.extend(re.findall(r'"([^"]+)"', text))
        entities.extend(re.findall(r"'([^']+)'", text))

        # Numbers
        entities.extend(re.findall(r'\b\d+(?:\.\d+)?\b', text))

        # Capitalized names
        entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

        return list(set(entities))

    def _check_argument_grounding(
        self,
        tool_call: Optional[Dict],
        prompt: str
    ) -> Dict[str, bool]:
        """Check if each argument is grounded in prompt."""
        if not tool_call:
            return {}

        prompt_lower = prompt.lower()
        grounding = {}

        for arg_name, arg_value in tool_call.get('arguments', {}).items():
            if isinstance(arg_value, str):
                grounding[arg_name] = arg_value.lower() in prompt_lower
            elif isinstance(arg_value, (int, float)):
                grounding[arg_name] = str(arg_value) in prompt
            else:
                grounding[arg_name] = True  # Complex types assumed grounded

        return grounding

    # === Type Checking ===

    def _check_types_with_details(self, response: str) -> Tuple[float, Dict]:
        """Check types and return details."""
        score = self._check_types(response)
        tool_call = self._extract_tool_call(response)
        details = {
            'parsed_successfully': tool_call is not None,
            'tool_call': tool_call
        }
        return score, details

    @abstractmethod
    def _check_types(self, response: str) -> float:
        """Validate JSON types. Override in generated class."""
        pass

    # === Reasoning Checking ===

    def _check_reasoning_with_details(self, response: str) -> Tuple[float, Dict]:
        """Check reasoning quality and return details."""
        score = self._check_reasoning(response)
        think_block = self._extract_think_block(response)
        details = {
            'word_count': len(think_block.split()) if think_block else 0,
            'has_logical_connectors': self._has_logical_connectors(think_block),
            'mentions_parameters': self._mentions_parameters(think_block)
        }
        return score, details

    @abstractmethod
    def _check_reasoning(self, response: str) -> float:
        """Validate reasoning quality. Override in generated class."""
        pass

    def _has_logical_connectors(self, text: str) -> bool:
        """Check for logical reasoning words."""
        if not text:
            return False
        connectors = ['because', 'therefore', 'since', 'so', 'thus',
                      'need to', 'should', 'must', 'in order to']
        text_lower = text.lower()
        return any(c in text_lower for c in connectors)

    def _mentions_parameters(self, text: str) -> List[str]:
        """Check which schema parameters are mentioned."""
        if not text:
            return []
        text_lower = text.lower()
        mentioned = []
        for param in self.schema.get('parameters', []):
            param_name = param.get('name', '') if isinstance(param, dict) else str(param)
            if param_name.lower() in text_lower:
                mentioned.append(param_name)
        return mentioned

    # === Utility Methods ===

    def _extract_tool_call(self, response: str) -> Optional[Dict]:
        """Extract tool call JSON from response."""
        match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None

    def _extract_think_block(self, response: str) -> Optional[str]:
        """Extract think block content."""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        return match.group(1).strip() if match else None
