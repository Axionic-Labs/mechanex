import os
import re
import json
import logging
from typing import Optional, List, Dict, Any
from ..interfaces import BaseTeacher, ToolCall
from ..ara.clients.teacher import TeacherClient
from ..ara.types.enums import TeacherProvider

logger = logging.getLogger(__name__)

class DefaultTeacher(BaseTeacher):
    """
    A default teacher implementation that wraps the ARA TeacherClient.
    Allows initialization via API key and model name strings.
    """
    def __init__(
        self, 
        provider_name: str, 
        schemas: List[Dict[str, Any]],
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None
    ):
        self.schemas = schemas
        # Map string provider to Enum
        try:
            self.provider = TeacherProvider(provider_name.lower())
        except ValueError:
            # Fallback for common aliases or direct mapping
            if "gemini" in provider_name.lower() or "google" in provider_name.lower():
                self.provider = TeacherProvider.GOOGLE
            elif "openai" in provider_name.lower():
                self.provider = TeacherProvider.OPENAI
            elif "anthropic" in provider_name.lower() or "claude" in provider_name.lower():
                self.provider = TeacherProvider.ANTHROPIC
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

        self.client = TeacherClient(
            provider=self.provider,
            model=model_name,
            api_key=api_key
        )

    def _extract_tool_call(self, text: str) -> Optional[ToolCall]:
        """Extract tool call from XML-like tags."""
        pattern = r"<tool_call>(.*?)</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            # Try parsing raw JSON if tags are missing but it looks like JSON
            if text.strip().startswith("{") and text.strip().endswith("}"):
                try:
                    data = json.loads(text.strip())
                    if "name" in data and "arguments" in data:
                        return ToolCall(name=data["name"], arguments=data["arguments"])
                except:
                    pass
            return None
        
        try:
            content = match.group(1).strip()
            # Handle possible markdown backticks
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
            data = json.loads(content)
            return ToolCall(name=data["name"], arguments=data["arguments"])
        except Exception as e:
            logger.warning(f"Failed to parse tool call from teacher: {e}")
            return None

    async def generate_trace(self, task: str) -> Dict[str, Any]:
        """
        Generates a reasoning trace using the underlying TeacherClient.
        """
        tool_definitions = json.dumps(self.schemas, indent=2)
        system_prompt = f"""You are an expert assistant. Your task is to solve the user's request by calling the appropriate tool.
Available tools:
{tool_definitions}

You MUST output your response EXACTLY in this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Do not provide any explanation or reasoning before or after the tags. Just the tool call."""

        try:
            # Use generate with system instruction if supported, or prepend
            response = self.client.generate(prompt=task, system=system_prompt)
            tool_call = self._extract_tool_call(response)
            
            return {
                "success": tool_call is not None,
                "tool_call": tool_call,
                "raw": response,
                "reasoning": "" # We are assuming direct tool calling for now
            }
        except Exception as e:
            logger.error(f"Teacher generation failed: {e}")
            return {
                "success": False,
                "tool_call": None,
                "raw": str(e),
                "reasoning": ""
            }

    async def generate(self, prompt: str, **kwargs) -> str:
        """Direct text generation support."""
        return self.client.generate(prompt, **kwargs)
