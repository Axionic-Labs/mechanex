"""AST-based security validator for generated code."""

import ast
from typing import Set, List

from ..types.results import ValidationResult


class ASTValidator(ast.NodeVisitor):
    """
    Validates Python AST for security violations.

    Checks for:
    - Forbidden imports
    - Blocked builtins
    - Dangerous function calls
    - Unsafe dunder access
    """

    def __init__(
        self,
        allowed_imports: Set[str],
        blocked_builtins: Set[str],
        blocked_calls: Set[str] = None
    ):
        self.allowed_imports = allowed_imports
        self.blocked_builtins = blocked_builtins
        self.blocked_calls = blocked_calls or set()
        self.violations: List[str] = []

    def validate(self, code: str) -> ValidationResult:
        """
        Validate code for security issues.

        Args:
            code: Python source code to validate

        Returns:
            ValidationResult with errors if any violations found
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"Syntax error: {e}"]
            )

        self.violations = []
        self.visit(tree)

        return ValidationResult(
            valid=len(self.violations) == 0,
            errors=self.violations
        )

    def visit_Import(self, node: ast.Import):
        """Check import statements."""
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module not in self.allowed_imports:
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from ... import statements."""
        if node.module:
            module = node.module.split('.')[0]
            if module not in self.allowed_imports:
                self.violations.append(f"Forbidden import: from {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check function calls."""
        call_name = self._get_call_name(node)

        if call_name in self.blocked_builtins:
            self.violations.append(f"Forbidden builtin: {call_name}")

        if call_name in self.blocked_calls:
            self.violations.append(f"Forbidden call: {call_name}")

        # Block __import__ calls (used to bypass import restrictions)
        if call_name == '__import__':
            self.violations.append("Forbidden builtin: __import__")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Check attribute access for dangerous patterns."""
        # Block __dunder__ access except allowed ones
        if node.attr.startswith('__') and node.attr.endswith('__'):
            allowed_dunders = {
                '__init__', '__call__', '__str__', '__repr__',
                '__name__', '__doc__', '__class__', '__dict__'
            }
            if node.attr not in allowed_dunders:
                self.violations.append(f"Forbidden dunder access: {node.attr}")
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ""
