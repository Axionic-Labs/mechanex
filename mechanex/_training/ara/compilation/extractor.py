"""Code extraction from Teacher responses."""

import re
import ast
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CodeExtractionError(Exception):
    """Error extracting code from response."""
    pass


class CodeExtractor:
    """Extracts Python code from Teacher responses."""

    CODE_PATTERNS = [
        r'```python\n(.*?)```',  # Python code block
        r'```py\n(.*?)```',      # Alternative python marker
        r'```\n(.*?)```',        # Generic code block
        r'```python(.*?)```',    # No newline after python
        r'```py(.*?)```',        # No newline after py
    ]

    def extract(self, response: str) -> str:
        """
        Extract Python code from response.

        Args:
            response: Raw Teacher response

        Returns:
            Extracted Python code

        Raises:
            CodeExtractionError: If valid code cannot be extracted
        """
        logger.debug(f"Attempting to extract code from response ({len(response)} chars)")

        # Try each regex pattern
        for pattern in self.CODE_PATTERNS:
            matches = re.findall(pattern, response, re.DOTALL)
            logger.debug(f"Pattern {pattern}: found {len(matches)} matches")
            for match in matches:
                code = match.strip()
                if self._validate_syntax(code) and self._has_mxreward_class(code):
                    logger.info(f"Successfully extracted code using pattern: {pattern}")
                    return code

        # Fallback 1: Handle unclosed code blocks (starts with ```python but no closing ```)
        if response.strip().startswith('```python') or response.strip().startswith('```py') or response.strip().startswith('```'):
            logger.debug("Trying fallback extraction for unclosed code block")
            # Remove the opening marker
            for marker in ['```python\n', '```python', '```py\n', '```py', '```\n', '```']:
                if response.strip().startswith(marker):
                    code = response.strip()[len(marker):].strip()
                    # Try to find closing ``` or just use all remaining content
                    closing_pos = code.find('```')
                    if closing_pos > 0:
                        code = code[:closing_pos].strip()

                    if self._validate_syntax(code) and self._has_mxreward_class(code):
                        logger.info("Successfully extracted code from unclosed code block")
                        return code
                    else:
                        logger.debug(f"Unclosed code block extraction failed validation (syntax error or no MXReward class)")
                    break

        # Fallback 2: Find class definition directly
        if "class MXReward" in response:
            logger.debug("Trying fallback extraction (class MXReward found in response)")

            # Find where imports start (usually before class)
            import_start = -1
            for import_marker in ['import re', 'import json', 'from typing']:
                pos = response.find(import_marker)
                if pos >= 0 and (import_start < 0 or pos < import_start):
                    import_start = pos

            # Start from imports if found, otherwise from class
            if import_start >= 0:
                start = import_start
                logger.debug(f"Found imports at position {import_start}")
            else:
                start = response.find("class MXReward")
                logger.debug(f"No imports found, starting from class at position {start}")

            code = response[start:]

            # Find the end of the code
            # Be careful with '\nclass ' - it shouldn't match the class we're extracting!
            # Only look for it AFTER we've seen the full class definition
            end_pos = len(code)

            # First, try to find closing backticks
            backtick_pos = code.find('```', 10)  # Start search after initial content
            if backtick_pos > 0:
                end_pos = backtick_pos
                logger.debug(f"Found closing backticks at position {backtick_pos}")
            else:
                # No closing backticks, look for other markers
                # But be careful not to match '\nclass ' that's part of our class definition
                # Look for a SECOND class definition (another '\nclass ' after ours)
                first_class_pos = code.find('class MXReward')
                if first_class_pos >= 0:
                    # Look for another class AFTER this one
                    second_class_pos = code.find('\nclass ', first_class_pos + 10)
                    if second_class_pos > 0:
                        end_pos = second_class_pos
                        logger.debug(f"Found second class definition at position {second_class_pos}")

                # Also check for section markers
                for marker in ['\n# ===', '\n\n\n\n']:
                    pos = code.find(marker, 10)
                    if pos > 0 and pos < end_pos:
                        end_pos = pos
                        logger.debug(f"Found end marker '{marker}' at position {pos}")

            code = code[:end_pos].strip()

            if self._validate_syntax(code) and self._has_mxreward_class(code):
                logger.info("Successfully extracted code using fallback method")
                return code
            else:
                logger.warning(f"Fallback extraction failed validation (syntax valid: {self._validate_syntax(code)}, has class: {self._has_mxreward_class(code)})")
                if not self._validate_syntax(code):
                    # Log the syntax error for debugging
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        logger.error(f"Syntax error in extracted code: {e}")

        # Log helpful debugging info
        logger.error(f"Code extraction failed. Response preview: {response[:300]}...")
        logger.error(f"Contains 'class MXReward': {'class MXReward' in response}")
        logger.error(f"Contains code blocks: {response.count('```')}")

        raise CodeExtractionError("Could not extract valid Python code from response")

    def _validate_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _has_mxreward_class(self, code: str) -> bool:
        """Check if code contains an MXReward class."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('MXReward'):
                    return True
            return False
        except SyntaxError:
            return False
