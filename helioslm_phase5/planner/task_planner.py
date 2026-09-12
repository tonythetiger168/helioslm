"""Hierarchical task planner for complex agent workflows.

Decomposes high-level goals into executable sub-tasks with dependency tracking.
"""
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskNode:
    """A single task in the plan."""
    id: str
    description: str
    tool: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0


class TaskPlanner:
    """
    Hierarchical task planner.

    Given a high-level goal, decomposes into sub-tasks using LLM,
    then executes with dependency resolution.
    """

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tasks: Dict[str, TaskNode] = {}

    def plan(self, goal: str, context: Optional[str] = None) -> List[TaskNode]:
        """
        Decompose goal into task plan using LLM.

        Args:
            goal: High-level goal description
            context: Additional context

        Returns:
            tasks: List of task nodes with dependencies
        """
        # Prompt LLM to generate plan
        prompt = self._create_planning_prompt(goal, context)

        # In real implementation, call LLM
        # plan_json = self.llm.generate(prompt)

        # Simulated plan for demonstration
        plan = [
            TaskNode(
                id="task_1",
                description="Search for relevant information",
                tool="web_search",
                parameters={"query": goal},
            ),
            TaskNode(
                id="task_2",
                description="Analyze findings",
                tool="code_execution",
                parameters={"code": "# analysis code"},
                dependencies=["task_1"],
            ),
            TaskNode(
                id="task_3",
                description="Generate final response",
                tool="llm_generate",
                parameters={"prompt": goal},
                dependencies=["task_2"],
            ),
        ]

        self.tasks = {t.id: t for t in plan}
        return plan

    def _create_planning_prompt(self, goal: str, context: Optional[str]) -> str:
        """Create planning prompt for LLM."""
        return f"""You are a task planner. Decompose the following goal into sub-tasks.

Goal: {goal}
Context: {context or "None"}

Output a JSON list of tasks with:
- id: unique task ID
- description: what to do
- tool: which tool to use (web_search, code_execution, file_read, llm_generate)
- parameters: tool parameters
- dependencies: list of task IDs that must complete first

Example:
[
  {{"id": "t1", "description": "Search web", "tool": "web_search", "parameters": {{"query": "..."}}, "dependencies": []}},
  {{"id": "t2", "description": "Analyze", "tool": "code_execution", "parameters": {{"code": "..."}}, "dependencies": ["t1"]}}
]
"""

    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            deps_satisfied = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if deps_satisfied:
                ready.append(task)

        return ready

    def update_task_status(self, task_id: str, status: TaskStatus, result=None, error=None):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].result = result
            self.tasks[task_id].error = error


class PlanExecutor:
    """Execute task plans with parallelization where possible."""

    def __init__(self, planner: TaskPlanner, tool_registry: Dict[str, callable]):
        self.planner = planner
        self.tools = tool_registry

    def execute(self, goal: str) -> Dict:
        """Execute a plan for the given goal."""
        # Create plan
        tasks = self.planner.plan(goal)

        results = {}

        # Execute until all tasks complete
        while any(t.status != TaskStatus.COMPLETED for t in self.planner.tasks.values()):
            ready = self.planner.get_ready_tasks()

            if not ready:
                # Check for deadlocks
                pending = [t for t in self.planner.tasks.values() if t.status == TaskStatus.PENDING]
                if pending:
                    # Some tasks are blocked by failed dependencies
                    for t in pending:
                        failed_deps = [
                            d for d in t.dependencies
                            if self.planner.tasks.get(d, TaskNode("", "")).status == TaskStatus.FAILED
                        ]
                        if failed_deps:
                            self.planner.update_task_status(t.id, TaskStatus.BLOCKED)
                break

            # Execute ready tasks (could be parallelized)
            for task in ready:
                self._execute_task(task)

        # Return final results
        final_task = max(self.planner.tasks.values(), key=lambda t: len(t.dependencies))
        return {
            "goal": goal,
            "tasks_completed": sum(1 for t in self.planner.tasks.values() if t.status == TaskStatus.COMPLETED),
            "tasks_failed": sum(1 for t in self.planner.tasks.values() if t.status == TaskStatus.FAILED),
            "result": final_task.result if final_task else None,
        }

    def _execute_task(self, task: TaskNode):
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS

        try:
            tool = self.tools.get(task.tool)
            if tool is None:
                raise ValueError(f"Unknown tool: {task.tool}")

            result = tool(**task.parameters)
            self.planner.update_task_status(task.id, TaskStatus.COMPLETED, result=result)
        except Exception as e:
            task.retry_count += 1
            if task.retry_count >= task.max_retries:
                self.planner.update_task_status(task.id, TaskStatus.FAILED, error=str(e))
            else:
                task.status = TaskStatus.PENDING  # Retry
