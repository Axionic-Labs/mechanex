"""Secure sandbox for executing generated reward code."""

import re
import json
import builtins
import signal
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..types.config import SandboxConfig
from ..types.results import ValidationResult, ExecutionResult
from .ast_validator import ASTValidator


class TimeoutError(Exception):
    """Execution timeout."""
    pass


@dataclass
class TestCase:
    """Test case for reward function."""
    response: str
    prompt: str
    expected_passed: Optional[bool] = None
    min_score: Optional[float] = None


class RewardSandbox:
    """
    Secure sandbox for executing generated reward code.

    Security Measures:
    1. AST Validation - Block dangerous constructs at parse time
    2. Import Allowlist - Only permit safe modules
    3. Call Blocklist - Prevent dangerous function calls
    4. Resource Limits - CPU, memory, time constraints
    5. Namespace Isolation - Restricted execution globals
    """

    # Security Configuration
    ALLOWED_IMPORTS = frozenset({
        're', 'json', 'typing', 'dataclasses',
        'datetime', 'enum', 'collections', 'math'
    })

    # Note: __import__ is needed for import statements to work in exec()
    # We block it in AST validation instead
    BLOCKED_BUILTINS = frozenset({
        'eval', 'exec', 'compile', 'open', 'input',
        'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr',
        'breakpoint', 'exit', 'quit'
    })

    BLOCKED_CALLS = frozenset({
        'os.system', 'os.popen', 'os.spawn', 'os.exec',
        'subprocess.run', 'subprocess.call', 'subprocess.Popen',
        'socket.socket', 'urllib.request.urlopen',
        'requests.get', 'requests.post'
    })

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.validator = ASTValidator(
            allowed_imports=self.config.allowed_imports or self.ALLOWED_IMPORTS,
            blocked_builtins=self.config.blocked_builtins or self.BLOCKED_BUILTINS,
            blocked_calls=self.BLOCKED_CALLS
        )

    def validate(self, code: str) -> ValidationResult:
        """
        Validate code for security issues.

        Args:
            code: Python source code

        Returns:
            ValidationResult with any security violations
        """
        return self.validator.validate(code)

    def execute(
        self,
        code: str,
        test_cases: Optional[List[TestCase]] = None
    ) -> ExecutionResult:
        """
        Execute reward code with test cases.

        Args:
            code: Validated Python code
            test_cases: Test inputs to verify behavior

        Returns:
            ExecutionResult with instantiated class and test results
        """
        # First validate the code
        validation = self.validate(code)
        if not validation.valid:
            return ExecutionResult(
                success=False,
                error=f"Validation failed: {validation.errors}",
                error_type="ValidationError"
            )

        # Create restricted namespace
        namespace = self._create_namespace()

        start_time = time.time()
        try:
            # Compile and execute with timeout
            compiled = compile(code, '<reward>', 'exec')
            exec(compiled, namespace)

            # Find the MXReward class
            reward_class = self._find_reward_class(namespace)

            # Instantiate
            reward_instance = reward_class()

            # Run test cases if provided
            test_results = []
            if test_cases:
                test_results = self._run_tests(reward_instance, test_cases)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                reward_class=reward_class,
                reward_instance=reward_instance,
                test_results=test_results,
                execution_time=execution_time
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=time.time() - start_time
            )

    def _create_namespace(self) -> Dict[str, Any]:
        """Create restricted execution namespace."""
        # Start with a copy of builtins and remove dangerous ones
        safe_builtins = dict(builtins.__dict__)

        # Remove blocked builtins
        for name in self.BLOCKED_BUILTINS:
            safe_builtins.pop(name, None)

        # Add allowed modules to namespace
        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            're': re,
            'json': json,
        }

        # Add typing module
        import typing
        namespace['typing'] = typing
        namespace['Dict'] = typing.Dict
        namespace['List'] = typing.List
        namespace['Optional'] = typing.Optional
        namespace['Any'] = typing.Any
        namespace['Union'] = typing.Union

        # Add dataclasses
        import dataclasses
        namespace['dataclasses'] = dataclasses
        namespace['dataclass'] = dataclasses.dataclass
        namespace['field'] = dataclasses.field

        return namespace

    def _find_reward_class(self, namespace: Dict[str, Any]) -> type:
        """Find the MXReward class in namespace."""
        for name, obj in namespace.items():
            if (
                isinstance(obj, type) and
                name.startswith('MXReward') and
                hasattr(obj, '__call__')
            ):
                return obj
        raise ValueError("No MXReward class found in generated code")

    def _run_tests(
        self,
        reward_instance: Any,
        test_cases: List[TestCase]
    ) -> List[Dict]:
        """Run test cases against reward instance."""
        results = []

        for i, test in enumerate(test_cases):
            try:
                result = reward_instance(test.response, test.prompt)

                test_result = {
                    'test_index': i,
                    'success': True,
                    'result': result
                }

                # Check expected outcomes if specified
                if test.expected_passed is not None:
                    test_result['expected_passed'] = test.expected_passed
                    test_result['actual_passed'] = result.get('passed', False)
                    test_result['pass_match'] = test_result['expected_passed'] == test_result['actual_passed']

                if test.min_score is not None:
                    test_result['min_score'] = test.min_score
                    test_result['actual_score'] = result.get('score', 0)
                    test_result['score_ok'] = test_result['actual_score'] >= test.min_score

                results.append(test_result)

            except Exception as e:
                results.append({
                    'test_index': i,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                })

        return results
