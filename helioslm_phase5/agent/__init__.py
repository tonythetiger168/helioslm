"""HeliosLM Phase 5 — ReAct Agent Loop

Reasoning + Acting + Tool use + Memory integration.
Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
           (Yao et al., 2023)

The ReAct loop:
  1. Thought: LLM reasons about what to do next
  2. Action: LLM selects a tool/action to execute
  3. Observation: Execute action and observe result
  4. Repeat until answer is found or max iterations reached
"""
from .react_loop import ReActAgent, ReActStep
from .tool_executor import SafeToolExecutor
from .memory_integration import MemoryAugmentedAgent

__all__ = ["ReActAgent", "ReActStep", "SafeToolExecutor", "MemoryAugmentedAgent"]
