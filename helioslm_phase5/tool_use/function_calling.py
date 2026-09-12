"""Function calling with JSON schema validation."""
import json
import inspect
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict  # JSON Schema
    required: List[str]
    returns: Optional[Dict] = None


class ToolRegistry:
    """Registry of available tools with schema validation."""

    def __init__(self):
        self.tools: Dict[str, ToolSchema] = {}
        self.implementations: Dict[str, Callable] = {}

    def register(self, schema: ToolSchema, implementation: Callable):
        """Register a tool with its schema and implementation."""
        self.tools[schema.name] = schema
        self.implementations[schema.name] = implementation

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        return self.tools.get(name)

    def get_implementation(self, name: str) -> Optional[Callable]:
        return self.implementations.get(name)

    def list_tools(self) -> List[Dict]:
        """List all tools in OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                }
            }
            for schema in self.tools.values()
        ]

    def validate_call(self, name: str, arguments: Dict) -> tuple:
        """Validate tool call arguments against schema."""
        schema = self.tools.get(name)
        if not schema:
            return False, f"Unknown tool: {name}"

        # Check required parameters
        for req in schema.required:
            if req not in arguments:
                return False, f"Missing required parameter: {req}"

        # Additional type validation could use jsonschema
        return True, None


class FunctionCaller:
    """
    LLM-powered function caller with schema-guided generation.

    1. Presents available tools to LLM
    2. LLM generates function call JSON
    3. Validates against schema
    4. Executes and returns result
    """

    def __init__(self, llm_engine, tool_registry: ToolRegistry):
        self.llm = llm_engine
        self.registry = tool_registry

    def call(self, user_message: str, conversation_history: Optional[List] = None) -> Dict:
        """
        Process user message and execute any function calls.

        Returns:
            {
                "content": str,  # Text response
                "tool_calls": List[Dict],  # Executed tool calls
                "results": List[Any],  # Tool results
            }
        """
        # Build system prompt with tool schemas
        tools_json = json.dumps(self.registry.list_tools(), indent=2)

        system_prompt = f"""You are a helpful assistant with access to tools.

Available tools:
{tools_json}

When you need to use a tool, respond with a JSON object:
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"param": "value"}}}}]}}

If no tool is needed, respond normally."""

        # In real implementation, call LLM with function calling format
        # For now, simulate

        # Check if user message implies a tool call
        if "search" in user_message.lower():
            tool_call = {"name": "web_search", "arguments": {"query": user_message}}
            valid, error = self.registry.validate_call(tool_call["name"], tool_call["arguments"])

            if valid:
                impl = self.registry.get_implementation(tool_call["name"])
                result = impl(**tool_call["arguments"]) if impl else None

                return {
                    "content": f"I searched for that. Result: {result}",
                    "tool_calls": [tool_call],
                    "results": [result],
                }

        return {
            "content": f"Response to: {user_message}",
            "tool_calls": [],
            "results": [],
        }
