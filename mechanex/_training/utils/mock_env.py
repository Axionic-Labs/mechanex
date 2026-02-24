import logging
from typing import List, Dict, Any
from ..interfaces import BaseEnvironment, ToolCall, ToolResult

logger = logging.getLogger(__name__)

class AutoEnvironment(BaseEnvironment):
    """
    A default environment that simulates tool execution based on provided schemas.
    Useful for users who haven't implemented a custom environment.
    """
    def __init__(self, tool_schemas: List[Dict[str, Any]]):
        self.schemas = {t['name']: t for t in tool_schemas}
        logger.info(f"AutoEnvironment initialized with {len(self.schemas)} tools")

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Simulates execution of a tool.
        Validates the tool name and returns a generic success message.
        """
        if tool_call.name not in self.schemas:
            error_msg = f"Tool '{tool_call.name}' not found in schemas."
            logger.warning(error_msg)
            return ToolResult(success=False, error=error_msg)

        # In a real scenario, we might validate arguments against the schema here.
        # For now, we simulate successful execution.
        logger.info(f"Auto-executing tool: {tool_call.name} with args: {tool_call.arguments}")
        
        # We return a generic success message. 
        # Future improvements could generate mock data based on the schema's return type if present.
        return ToolResult(
            success=True, 
            data=f"Simulated execution of {tool_call.name} successful."
        )
