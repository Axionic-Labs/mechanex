"""
Prompt Validator Module

This module implements validation for generated seed prompts to ensure they are:
- Grammatically correct
- Semantically meaningful
- Properly grounded with realistic entities
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.seed_prompt import SeedPrompt
from ..models.tool_schemas import ToolSchema, CRM_TOOLS_BY_NAME


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"       # All checks must pass
    NORMAL = "normal"       # Most checks must pass
    LENIENT = "lenient"     # Basic checks only


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    overall_score: float
    grammar_score: float
    grounding_score: float
    completeness_score: float
    relevance_score: float
    issues: List[str]
    suggestions: List[str]


class PromptValidator:
    """
    Validates generated prompts for quality and correctness.

    Performs:
    - Grammar and syntax checks
    - Entity grounding validation
    - Completeness checks (required entities present)
    - Relevance checks (prompt matches tool purpose)
    """

    # Common grammar patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]$')
    CAPITALIZED_START = re.compile(r'^[A-Z]')
    REPEATED_WORDS = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
    EXCESSIVE_PUNCTUATION = re.compile(r'[!?]{3,}')

    # Entity patterns
    LEAD_ID_PATTERN = re.compile(r'LD-\d{4}')
    SALES_REP_ID_PATTERN = re.compile(r'SR-\d{3}')
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_PATTERN = re.compile(r'[\d\s\-\(\)\+]{10,}')
    DATE_PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}')
    MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)?', re.IGNORECASE)

    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        self.level = level
        self.tools = CRM_TOOLS_BY_NAME

    def validate(
        self,
        prompt: SeedPrompt,
        tool_schema: Optional[ToolSchema] = None
    ) -> ValidationResult:
        """
        Validate a seed prompt.

        Args:
            prompt: The seed prompt to validate
            tool_schema: Optional tool schema for context

        Returns:
            ValidationResult with scores and feedback
        """
        if tool_schema is None:
            tool_schema = self.tools.get(prompt.metadata.tool_name)

        issues: List[str] = []
        suggestions: List[str] = []

        # Run validation checks
        grammar_score, grammar_issues = self._check_grammar(prompt.prompt)
        issues.extend(grammar_issues)

        grounding_score, grounding_issues = self._check_grounding(
            prompt.prompt,
            prompt.expected_entities.entities
        )
        issues.extend(grounding_issues)

        completeness_score, completeness_issues = self._check_completeness(
            prompt.expected_entities.entities,
            tool_schema
        )
        issues.extend(completeness_issues)

        relevance_score, relevance_issues = self._check_relevance(
            prompt.prompt,
            tool_schema
        )
        issues.extend(relevance_issues)

        # Calculate overall score
        weights = {
            "grammar": 0.20,
            "grounding": 0.30,
            "completeness": 0.25,
            "relevance": 0.25
        }

        overall_score = (
            weights["grammar"] * grammar_score +
            weights["grounding"] * grounding_score +
            weights["completeness"] * completeness_score +
            weights["relevance"] * relevance_score
        )

        # Determine if valid based on level
        if self.level == ValidationLevel.STRICT:
            is_valid = overall_score >= 0.85 and len(issues) == 0
        elif self.level == ValidationLevel.NORMAL:
            is_valid = overall_score >= 0.70
        else:  # LENIENT
            is_valid = overall_score >= 0.50

        # Generate suggestions
        suggestions = self._generate_suggestions(
            grammar_score,
            grounding_score,
            completeness_score,
            relevance_score
        )

        return ValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            grammar_score=grammar_score,
            grounding_score=grounding_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score,
            issues=issues,
            suggestions=suggestions
        )

    def _check_grammar(self, text: str) -> Tuple[float, List[str]]:
        """Check grammar and syntax."""
        issues = []
        score = 1.0

        # Check for empty or too short
        if len(text.strip()) < 10:
            issues.append("Prompt is too short")
            score -= 0.5

        # Check for proper capitalization
        if text and not self.CAPITALIZED_START.match(text):
            issues.append("Prompt should start with a capital letter")
            score -= 0.1

        # Check for sentence ending (for formal prompts)
        if text and not self.SENTENCE_ENDINGS.search(text):
            # Only a minor issue for conversational prompts
            score -= 0.05

        # Check for repeated words
        if self.REPEATED_WORDS.search(text):
            issues.append("Contains repeated words")
            score -= 0.15

        # Check for excessive punctuation
        if self.EXCESSIVE_PUNCTUATION.search(text):
            issues.append("Contains excessive punctuation")
            score -= 0.1

        # Check for balanced brackets/quotes
        if text.count('(') != text.count(')'):
            issues.append("Unbalanced parentheses")
            score -= 0.15
        if text.count('"') % 2 != 0:
            issues.append("Unbalanced quotes")
            score -= 0.1

        # Check for very long sentences (potential run-on)
        words = text.split()
        if len(words) > 75:
            issues.append("Prompt may be too long or a run-on sentence")
            score -= 0.1

        return max(0.0, score), issues

    def _check_grounding(
        self,
        text: str,
        entities: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Check that entities are properly grounded in the prompt."""
        issues = []

        if not entities:
            return 0.5, ["No expected entities defined"]

        grounded_count = 0
        total_entities = len(entities)

        for key, value in entities.items():
            if value is None:
                continue

            value_str = str(value)

            # Check if the value appears in the text
            if value_str.lower() in text.lower():
                grounded_count += 1
            elif key == "lead_id" and self.LEAD_ID_PATTERN.search(text):
                # Check for lead ID format
                grounded_count += 1
            elif key == "sales_rep_id" and self.SALES_REP_ID_PATTERN.search(text):
                grounded_count += 1
            elif key == "email" and self.EMAIL_PATTERN.search(text):
                grounded_count += 1
            elif key in ["deal_value", "estimated_value", "min_value", "max_value"]:
                if self.MONEY_PATTERN.search(text) or str(value) in text:
                    grounded_count += 1
            else:
                # Check for partial matches (e.g., first name only)
                if any(part.lower() in text.lower() for part in value_str.split() if len(part) > 2):
                    grounded_count += 0.5

        score = grounded_count / max(total_entities, 1)

        if score < 0.5:
            issues.append(f"Only {grounded_count}/{total_entities} entities are grounded in the prompt")

        return min(1.0, score), issues

    def _check_completeness(
        self,
        entities: Dict[str, Any],
        tool_schema: Optional[ToolSchema]
    ) -> Tuple[float, List[str]]:
        """Check that required parameters are covered."""
        issues = []

        if tool_schema is None:
            return 0.8, []  # Can't validate without schema

        required_params = [p.name for p in tool_schema.get_required_params()]

        if not required_params:
            return 1.0, []

        covered_count = sum(1 for p in required_params if p in entities)
        score = covered_count / len(required_params)

        missing = [p for p in required_params if p not in entities]
        if missing:
            issues.append(f"Missing required parameters: {', '.join(missing)}")

        return score, issues

    def _check_relevance(
        self,
        text: str,
        tool_schema: Optional[ToolSchema]
    ) -> Tuple[float, List[str]]:
        """Check that the prompt is relevant to the tool."""
        issues = []

        if tool_schema is None:
            return 0.8, []

        # Check for tool-related keywords
        tool_keywords = self._get_tool_keywords(tool_schema)
        text_lower = text.lower()

        matched_keywords = sum(1 for kw in tool_keywords if kw.lower() in text_lower)
        keyword_score = min(1.0, matched_keywords / max(len(tool_keywords) * 0.3, 1))

        # Check for action verbs related to the tool
        action_verbs = self._get_action_verbs(tool_schema)
        has_action = any(verb.lower() in text_lower for verb in action_verbs)
        action_score = 1.0 if has_action else 0.7

        score = (keyword_score * 0.6 + action_score * 0.4)

        if score < 0.5:
            issues.append("Prompt may not be relevant to the tool's purpose")

        return score, issues

    def _get_tool_keywords(self, tool_schema: ToolSchema) -> List[str]:
        """Get keywords associated with a tool."""
        keywords = [tool_schema.name.replace("_", " ")]

        # Extract from description
        description_words = tool_schema.description.lower().split()
        keywords.extend([w for w in description_words if len(w) > 4])

        # Add parameter names
        keywords.extend([p.name.replace("_", " ") for p in tool_schema.parameters])

        return keywords

    def _get_action_verbs(self, tool_schema: ToolSchema) -> List[str]:
        """Get action verbs for a tool."""
        verb_map = {
            "update_lead_status": ["update", "change", "modify", "set", "mark"],
            "search_leads": ["search", "find", "look for", "filter", "show", "list", "get"],
            "create_lead": ["create", "add", "new", "register", "enter"],
            "assign_lead": ["assign", "give", "transfer", "reassign", "allocate"],
            "schedule_followup": ["schedule", "set up", "book", "plan", "arrange"],
            "log_activity": ["log", "record", "note", "document", "track"]
        }

        return verb_map.get(tool_schema.name, ["perform", "execute", "do"])

    def _generate_suggestions(
        self,
        grammar_score: float,
        grounding_score: float,
        completeness_score: float,
        relevance_score: float
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []

        if grammar_score < 0.8:
            suggestions.append("Improve grammar and sentence structure")

        if grounding_score < 0.7:
            suggestions.append("Ensure all entity values are clearly mentioned in the prompt")

        if completeness_score < 0.8:
            suggestions.append("Include all required parameters in the prompt context")

        if relevance_score < 0.7:
            suggestions.append("Make the prompt more clearly related to the tool's purpose")

        return suggestions

    def validate_batch(
        self,
        prompts: List[SeedPrompt]
    ) -> Dict[str, Any]:
        """
        Validate a batch of prompts.

        Returns:
            Summary statistics for the batch
        """
        results = []
        valid_count = 0
        total_score = 0.0

        for prompt in prompts:
            result = self.validate(prompt)
            results.append({
                "id": prompt.id,
                "is_valid": result.is_valid,
                "score": result.overall_score
            })

            if result.is_valid:
                valid_count += 1
            total_score += result.overall_score

        return {
            "total": len(prompts),
            "valid": valid_count,
            "invalid": len(prompts) - valid_count,
            "valid_rate": valid_count / len(prompts) if prompts else 0,
            "average_score": total_score / len(prompts) if prompts else 0,
            "results": results
        }


def validate_prompt(
    prompt: SeedPrompt,
    level: ValidationLevel = ValidationLevel.NORMAL
) -> ValidationResult:
    """Convenience function to validate a single prompt."""
    validator = PromptValidator(level)
    return validator.validate(prompt)


def validate_prompts(
    prompts: List[SeedPrompt],
    level: ValidationLevel = ValidationLevel.NORMAL
) -> Dict[str, Any]:
    """Convenience function to validate multiple prompts."""
    validator = PromptValidator(level)
    return validator.validate_batch(prompts)
