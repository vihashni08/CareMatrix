"""Lightweight, framework-free native tool abstraction for CareMatrix agents.

Designed to prepare CareMatrix agents for genuine LLM-driven function calling / tool use
without introducing external heavyweight agent frameworks (LangChain, CrewAI, AutoGen, etc.).
"""

from __future__ import annotations

import inspect
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("CareMatrix.Tools")


@dataclass
class ToolResult:
    """Standardized output container for tool executions."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool:
    """Lightweight, standardized agent-callable tool.

    Wraps existing Python functions with:
    - Unique name
    - Short description
    - Clearly defined input schema (JSON Schema format)
    - Clearly defined output schema / format
    - Safe error handling (never crashes agent caller on runtime exceptions)
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.tags = tags or []
        self.output_schema = output_schema or {"type": "object"}
        self.parameters = parameters or self._infer_parameters(func)

    def _infer_parameters(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Infer basic JSON Schema from function signature and docstrings."""
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = "string"  # default safe fallback
            if param.annotation != inspect.Parameter.empty:
                # Resolve primitive types if available
                ann = param.annotation
                if isinstance(ann, type) and ann in type_map:
                    param_type = type_map[ann]

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function with safe error handling."""
        try:
            result = self.func(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as exc:
            logger.warning("Tool %s execution failed: %s", self.name, exc)
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {str(exc)}",
                metadata={"traceback": traceback.format_exc()},
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow direct transparent call to preserve existing callable behaviors."""
        return self.func(*args, **kwargs)

    def to_gemini_declaration(self) -> Dict[str, Any]:
        """Export tool declaration compatible with Google Gemini / OpenAI function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry maintaining agent tools with lookup and discovery capabilities."""

    def __init__(self, name: str = "default"):
        self.name = name
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a Tool instance."""
        if tool.name in self._tools:
            logger.debug("Overwriting existing tool '%s' in registry '%s'", tool.name, self.name)
        self._tools[tool.name] = tool
        return tool

    def register_func(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Callable[[Callable[..., Any]], Tool]:
        """Decorator to register a function directly as a tool."""

        def decorator(func: Callable[..., Any]) -> Tool:
            tool = Tool(
                name=name,
                description=description,
                func=func,
                parameters=parameters,
                output_schema=output_schema,
                tags=tags,
            )
            self.register(tool)
            return tool

        return decorator

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Safely execute tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry '{self.name}'. Available: {self.list_names()}",
            )
        return tool.execute(**kwargs)

    def to_gemini_tool_declarations(self) -> List[Dict[str, Any]]:
        """Return function declarations for LLM function calling."""
        return [tool.to_gemini_declaration() for tool in self._tools.values()]

    def __getitem__(self, name: str) -> Tool:
        """Allow dict-style indexing: registry['name']."""
        tool = self.get(name)
        return tool

    def __setitem__(self, name: str, value: Any) -> None:
        """Allow dict-style assignment: registry['name'] = tool_or_callable."""
        if isinstance(value, Tool):
            self.register(value)
        elif callable(value):
            self.register(Tool(name=name, description=f"Custom tool {name}", func=value))
        else:
            raise TypeError(f"Expected Tool or Callable, got {type(value)}")

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return count of registered tools."""
        return len(self._tools)

    def keys(self) -> List[str]:
        return list(self._tools.keys())

    def values(self) -> List[Tool]:
        return list(self._tools.values())

    def items(self) -> List[Tuple[str, Tool]]:
        return list(self._tools.items())

    def __iter__(self):
        return iter(self._tools)

