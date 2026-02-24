from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None

class BaseEnvironment(ABC):
    @abstractmethod
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        pass

class BaseModel(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model."""
        pass

class BaseTeacher(ABC):
    @abstractmethod
    async def generate_trace(self, task: str) -> Dict[str, Any]:
        """Generate a reasoning trace for a given task."""
        pass
