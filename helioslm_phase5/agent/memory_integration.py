"""Integrate Phase 5 advanced memory into ReAct agent.

Enables the agent to:
  - Retrieve relevant past experiences during reasoning
  - Store successful trajectories for future reuse
  - Recall procedural knowledge for tool usage
"""
from typing import Dict, List, Optional
from .react_loop import ReActAgent


class MemoryAugmentedAgent:
    """
    ReAct agent enhanced with episodic, semantic, and procedural memory.

    Before each thought, retrieves relevant memories to inform reasoning.
    After each successful trajectory, stores it for future retrieval.
    """

    def __init__(self, react_agent: ReActAgent, memory_system):
        self.agent = react_agent
        self.memory = memory_system

    def run(self, query: str, context: Optional[str] = None) -> Dict:
        """
        Run memory-augmented ReAct loop.

        1. Retrieve relevant memories before starting
        2. Inject memories into context
        3. Run standard ReAct
        4. Store successful trajectory
        """
        # Retrieve relevant memories
        # In real implementation, compute query embedding
        # memories = self.memory.retrieve_relevant(query_embedding)

        # Build augmented context
        augmented_context = self._build_memory_context(context)

        # Run ReAct with augmented context
        result = self.agent.run(query, augmented_context)

        # Store successful trajectory
        if result["success"]:
            self._store_trajectory(query, result)

        return result

    def _build_memory_context(self, original_context: Optional[str]) -> str:
        """Build context string with relevant memories."""
        parts = []

        if original_context:
            parts.append(f"Original context: {original_context}")

        # Add episodic memories (past similar queries)
        # episodes = self.memory.episodic.retrieve(query_embedding, top_k=3)
        # for ep in episodes:
        #     parts.append(f"Past experience: {ep.content}")

        # Add semantic memories (relevant facts)
        # fact = self.memory.semantic.query(query_embedding)
        # if fact:
        #     parts.append(f"Relevant fact: {fact}")

        # Add procedural memories (relevant skills)
        # procedures = self.memory.procedural.get_procedure("problem_solving")
        # if procedures:
        #     parts.append(f"Suggested approach: {' -> '.join(procedures)}")

        return "\n".join(parts) if parts else ""

    def _store_trajectory(self, query: str, result: Dict):
        """Store successful trajectory in memory."""
        # Create summary of trajectory
        trajectory_summary = f"Query: {query}\nAnswer: {result['answer']}\n"
        trajectory_summary += f"Steps: {result['iterations']}, Tools: {result['tool_calls']}"

        # Store as episodic memory
        # self.memory.episodic.add(trajectory_summary, embedding)

        # Extract and store any new facts as semantic memory
        # self.memory.semantic.store_fact("key", "value", embedding)

        # Update procedural memory with successful strategy
        # self.memory.procedural.learn_procedure("strategy_name", steps)

        pass  # Placeholder for actual memory operations
