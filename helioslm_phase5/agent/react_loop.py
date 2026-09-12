"""ReAct: Reasoning + Acting loop implementation.

The core insight of ReAct is that interleaving reasoning (thought) with
actions (tool use) leads to better problem-solving than either alone.

Thought → Action → Observation → Thought → Action → ... → Answer
"""
import json
import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


@dataclass
class ReActStep:
    """A single step in the ReAct trajectory."""
    step_type: StepType
    content: str
    metadata: Dict = field(default_factory=dict)


class ReActAgent:
    """
    ReAct agent with integrated reasoning and acting.

    Features:
      - Structured thought generation
      - JSON-formatted action parsing
      - Error recovery and retry
      - Trajectory logging for training
      - Max iteration limits to prevent infinite loops
    """

    def __init__(
        self,
        llm_engine,
        tool_registry: Dict[str, Callable],
        max_iterations: int = 10,
        max_tool_calls: int = 5,
        reflection_enabled: bool = True,
    ):
        self.llm = llm_engine
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.reflection_enabled = reflection_enabled

        self.trajectory: List[ReActStep] = []
        self.tool_call_count = 0

    def run(self, query: str, context: Optional[str] = None) -> Dict:
        """
        Execute ReAct loop for a given query.

        Args:
            query: User query to answer
            context: Additional context

        Returns:
            {
                "answer": str,
                "trajectory": List[ReActStep],
                "iterations": int,
                "tool_calls": int,
                "success": bool,
            }
        """
        self.trajectory = []
        self.tool_call_count = 0

        # Initial thought
        system_prompt = self._build_system_prompt()

        for iteration in range(self.max_iterations):
            # Build prompt from trajectory
            prompt = self._build_prompt(query, context)

            # Generate next step
            response = self.llm.generate(prompt, max_tokens=512, temperature=0.3)

            # Parse response
            parsed = self._parse_response(response)

            if parsed["type"] == "answer":
                self.trajectory.append(ReActStep(StepType.ANSWER, parsed["content"]))
                return {
                    "answer": parsed["content"],
                    "trajectory": self.trajectory,
                    "iterations": iteration + 1,
                    "tool_calls": self.tool_call_count,
                    "success": True,
                }

            elif parsed["type"] == "thought":
                self.trajectory.append(ReActStep(StepType.THOUGHT, parsed["content"]))
                # Continue to next iteration for action

            elif parsed["type"] == "action":
                self.trajectory.append(ReActStep(StepType.ACTION, json.dumps(parsed["action"])))

                # Execute action
                if self.tool_call_count >= self.max_tool_calls:
                    self.trajectory.append(ReActStep(
                        StepType.OBSERVATION,
                        "Error: Maximum tool calls reached.",
                        {"error": "max_tool_calls"}
                    ))
                    break

                observation = self._execute_action(parsed["action"])
                self.trajectory.append(ReActStep(StepType.OBSERVATION, str(observation)))
                self.tool_call_count += 1

            # Reflection after each action-observation pair
            if self.reflection_enabled and len(self.trajectory) >= 3:
                self._reflect_on_trajectory()

        # Max iterations reached without answer
        return {
            "answer": "I could not find a complete answer within the allowed iterations.",
            "trajectory": self.trajectory,
            "iterations": self.max_iterations,
            "tool_calls": self.tool_call_count,
            "success": False,
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])

        return f"""You are a reasoning agent that solves problems by thinking step by step and using tools.

Available tools:
{tools_desc}

Respond in one of these formats:

1. Thought:
THOUGHT: <your reasoning about what to do next>

2. Action:
ACTION: {{"tool": "tool_name", "arguments": {{"param": "value"}}}}

3. Answer:
ANSWER: <your final answer>

Rules:
- Always think before acting
- Use tools only when necessary
- If a tool fails, try a different approach
- Provide a clear ANSWER when you have enough information
"""

    def _build_prompt(self, query: str, context: Optional[str]) -> str:
        """Build prompt from current trajectory."""
        lines = [self._build_system_prompt()]

        if context:
            lines.append(f"Context: {context}")

        lines.append(f"Query: {query}")
        lines.append("")

        # Add trajectory history
        for step in self.trajectory:
            if step.step_type == StepType.THOUGHT:
                lines.append(f"THOUGHT: {step.content}")
            elif step.step_type == StepType.ACTION:
                lines.append(f"ACTION: {step.content}")
            elif step.step_type == StepType.OBSERVATION:
                lines.append(f"OBSERVATION: {step.content}")

        lines.append("THOUGHT:")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format."""
        response = response.strip()

        # Check for answer
        if response.startswith("ANSWER:"):
            return {"type": "answer", "content": response[7:].strip()}

        # Check for action
        action_match = re.search(r'ACTION:\s*(\{.*?\})', response, re.DOTALL)
        if action_match:
            try:
                action = json.loads(action_match.group(1))
                return {"type": "action", "action": action}
            except json.JSONDecodeError:
                pass

        # Default to thought
        thought = response
        if response.startswith("THOUGHT:"):
            thought = response[8:].strip()

        return {"type": "thought", "content": thought}

    def _execute_action(self, action: Dict) -> Any:
        """Execute a tool action."""
        tool_name = action.get("tool", "")
        arguments = action.get("arguments", {})

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            result = self.tools[tool_name](**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _reflect_on_trajectory(self):
        """Reflect on recent steps and potentially backtrack."""
        # Check if last observation indicates failure
        last_obs = None
        for step in reversed(self.trajectory):
            if step.step_type == StepType.OBSERVATION:
                last_obs = step.content
                break

        if last_obs and "Error" in last_obs:
            # Add reflection as a thought
            self.trajectory.append(ReActStep(
                StepType.THOUGHT,
                f"The previous action failed. I need to try a different approach.",
                {"reflection": True, "error_observed": last_obs}
            ))
