"""Prompt templates for reward function compilation."""

import json
from typing import Dict, List, Optional

from ..parsing.canonical import CanonicalSchema, Example
from ..types.enums import TaskType


class PromptBuilder:
    """Builds prompts for reward function compilation."""

    COMPILATION_PROMPT = '''You are a Python compiler that generates reward functions for evaluating LLM tool-calling outputs.

## TOOL SCHEMA
```json
{schema_json}
```

## TASK TYPE
{task_type}

## REWARD WEIGHTS
```json
{weights}
```

## INSTRUCTIONS

Generate a Python class `MXReward_{tool_name}` that evaluates model responses for this tool.

The class MUST implement these four validation methods:

### 1. Format Validation (`_check_format`)
- Verify `<think>` and `</think>` tags are present and in correct order
- Verify `<tool_call>` and `</tool_call>` tags are present
- Check that reasoning block is non-empty (>10 characters)
- Return score 0.0-1.0

### 2. Grounding Validation (`_check_grounding`)
- Extract entities from the prompt (names, IDs, values)
- Verify each argument in the tool call is grounded in the prompt
- Penalize hallucinated or fabricated values
- Return score 0.0-1.0

### 3. Type Validation (`_check_types`)
- Parse the JSON from `<tool_call>` block. The JSON structure is: {{"name": "tool_name", "arguments": {{"param1": value1, ...}}}}
- Check that tool_call["name"] == "{tool_name}"
- Check that required parameters exist in tool_call["arguments"]
- Validate enum values, ranges, patterns for parameters in tool_call["arguments"]
- Return score 0.0-1.0

### 4. Reasoning Quality (`_check_reasoning`)
- Evaluate the `<think>` block quality
- Check for parameter mentions from schema
- Look for logical connectors (because, therefore, since)
- Minimum reasoning length ~50 words
- Return score 0.0-1.0

### Main `__call__` Method
```python
def __call__(self, response: str, prompt: str) -> dict:
    scores = {{}}
    scores['format'] = self._check_format(response)
    scores['grounding'] = self._check_grounding(response, prompt)
    scores['type_valid'] = self._check_types(response)
    scores['reasoning'] = self._check_reasoning(response)

    final_score = sum(self.WEIGHTS[k] * v for k, v in scores.items())

    return {{
        'score': final_score,
        'breakdown': scores,
        'passed': final_score >= {pass_threshold}
    }}
```

{context}

{examples}

## OUTPUT FORMAT

CRITICAL: Return ONLY a Python code block in this EXACT format:

```python
import re
import json

class MXReward_{tool_name}:
    WEIGHTS = {{'format': 0.2, 'grounding': 0.3, 'type_valid': 0.3, 'reasoning': 0.2}}

    def __init__(self):
        pass

    def _check_format(self, response: str) -> float:
        # implementation
        pass

    def _check_grounding(self, response: str, prompt: str) -> float:
        # implementation
        pass

    def _check_types(self, response: str) -> float:
        # implementation
        pass

    def _check_reasoning(self, response: str) -> float:
        # implementation
        pass

    def __call__(self, response: str, prompt: str) -> dict:
        # implementation
        pass
```

Requirements:
1. Start with ```python
2. End with ```
3. Be syntactically valid Python 3.9+ (do NOT use | for type unions, use Optional/Union instead)
4. Only import: `re`, `json`
5. Define a regular Python class `MXReward_{tool_name}` (do NOT use @dataclass decorator)
6. Define WEIGHTS as a class variable dict
7. Include all four _check_* methods and __call__
8. NO explanations, NO additional text, ONLY the code block

Now provide the complete implementation:'''

    def build(
        self,
        schema: CanonicalSchema,
        task_type: TaskType,
        weights: Dict[str, float],
        pass_threshold: float = 0.7
    ) -> str:
        """Build the compilation prompt."""

        context_str = ""
        if schema.long_description:
            context_str = f"\n## ADDITIONAL CONTEXT\n{schema.long_description}\n"

        examples_str = ""
        if schema.examples:
            examples_str = "\n## EXAMPLES FOR GROUNDING\n" + self._format_examples(schema.examples)

        return self.COMPILATION_PROMPT.format(
            schema_json=json.dumps(schema.to_json_schema(), indent=2),
            tool_name=schema.name,
            task_type=task_type.value,
            weights=json.dumps(weights, indent=2),
            pass_threshold=pass_threshold,
            context=context_str,
            examples=examples_str
        )

    def _format_examples(self, examples: List[Example]) -> str:
        """Format examples for the prompt."""
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Prompt: {ex.prompt}")
            lines.append(f"  Expected arguments: {json.dumps(ex.expected_arguments)}")
            if ex.reasoning:
                lines.append(f"  Reasoning: {ex.reasoning}")
            lines.append("")
        return "\n".join(lines)
