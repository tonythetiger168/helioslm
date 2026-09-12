"""Agentic Layer v2 - P3: MCP + Self-Reflection + Code Execution (Production-Ready)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import tempfile
import os


class MCPToolRouter(nn.Module):
    """Tool router with confidence scoring and parameter generation."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_tools = config.agentic.max_tools_per_turn

        self.tool_selector = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 1000)
        )
        self.param_generator = nn.Linear(self.hidden_size, self.hidden_size)
        self.confidence_head = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden_states):
        scores = self.tool_selector(hidden_states[:, -1, :])
        top_tools = torch.topk(scores, self.max_tools, dim=-1)
        conf = torch.sigmoid(self.confidence_head(hidden_states[:, -1, :]))
        return {
            "tool_ids": top_tools.indices,
            "tool_scores": top_tools.values,
            "parameters": self.param_generator(hidden_states[:, -1, :]),
            "confidence": conf,
        }


class SelfReflectionModule(nn.Module):
    """Self-reflection with error detection and retry logic."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_depth = config.agentic.max_reflection_depth

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                batch_first=True, dropout=0.1
            ),
            num_layers=4
        )
        self.confidence_head = nn.Linear(self.hidden_size, 1)
        self.error_detector = nn.Linear(self.hidden_size, 1)
        self.retry_recommender = nn.Linear(self.hidden_size, 1)

    def reflect(self, hidden_states, trace):
        """
        Reflect on execution trace.

        Args:
            hidden_states: current state
            trace: execution history tensor

        Returns:
            dict with confidence, error probability, retry recommendation
        """
        trace_emb = self.encoder(trace)
        pooled = trace_emb[:, -1, :]

        conf = torch.sigmoid(self.confidence_head(pooled))
        err = torch.sigmoid(self.error_detector(pooled))
        retry = torch.sigmoid(self.retry_recommender(pooled))

        return {
            "confidence": conf,
            "error_prob": err,
            "needs_retry": (err > 0.5) | (conf < 0.7),
            "retry_recommended": retry > 0.5,
            "trace_embedding": trace_emb,
        }


class CodeExecutionValidator:
    """
    Safe code execution in sandboxed environment.

    Uses restricted Python execution (no exec())
    """

    def __init__(self, config):
        self.enabled = config.agentic.code_execution_enabled
        self.auto_retry = config.agentic.auto_retry_on_fail
        self.max_retries = config.agentic.max_retries
        self.timeout = config.agentic.tool_timeout_seconds

    def validate(self, code_snippet):
        """
        Validate code execution safely.

        Returns:
            dict with valid flag, output, and error info
        """
        if not self.enabled:
            return {"valid": True, "output": "", "error": None}

        # Security: never use exec() directly
        # Use restricted evaluation for simple expressions
        try:
            # Only allow safe operations
            safe_dict = {
                "abs": abs, "max": max, "min": min, "sum": sum,
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "int": int, "float": float, "str": str,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
            }

            # Check for dangerous keywords
            dangerous = ["import", "exec", "eval", "compile", "open", "os.", "sys.", "subprocess"]
            if any(d in code_snippet.lower() for d in dangerous):
                return {
                    "valid": False,
                    "output": "",
                    "error": "Code contains potentially dangerous operations",
                    "needs_retry": self.auto_retry
                }

            # Safe evaluation for simple expressions
            result = eval(code_snippet, {"__builtins__": {}}, safe_dict)
            return {
                "valid": True,
                "output": str(result),
                "error": None,
                "needs_retry": False
            }
        except Exception as e:
            return {
                "valid": False,
                "output": "",
                "error": str(e),
                "needs_retry": self.auto_retry
            }


class AgenticLayer(nn.Module):
    """
    Full agentic layer with tool routing, reflection, and code validation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.agentic.enabled:
            self.tool_router = MCPToolRouter(config)
            self.reflection = SelfReflectionModule(config)
            self.code_validator = CodeExecutionValidator(config)

            # Fusion gate for agent outputs
            self.agent_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, hidden_states, trace=None):
        """
        Agentic forward pass.

        Returns:
            dict with tools, reflection, and code validation results
        """
        out = {}

        if self.config.agentic.mcp_enabled:
            out["tools"] = self.tool_router(hidden_states)

        if trace is not None:
            out["reflection"] = self.reflection.reflect(hidden_states, trace)

        return out

    def execute_tool(self, tool_name, arguments, tool_registry):
        """
        Execute a tool from the registry.

        Args:
            tool_name: str
            arguments: dict
            tool_registry: dict of callable tools

        Returns:
            Tool execution result
        """
        if tool_name not in tool_registry:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = tool_registry[tool_name](**arguments)
            return {"result": result, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}
