"""HeliosLM Phase 5 — Advanced Tool Use

Function calling with schema validation, parallel execution, and result integration.
"""
from .function_calling import FunctionCaller, ToolSchema, ToolRegistry
from .parallel_executor import ParallelToolExecutor

__all__ = ["FunctionCaller", "ToolSchema", "ToolRegistry", "ParallelToolExecutor"]
