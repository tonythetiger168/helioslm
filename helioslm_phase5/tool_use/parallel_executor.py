"""Parallel tool execution for independent function calls."""
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor


class ParallelToolExecutor:
    """
    Execute multiple independent tool calls in parallel.

    When LLM generates multiple function calls with no dependencies,
    execute them concurrently for lower latency.
    """

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_parallel(self, tool_calls: List[Dict], registry) -> List[Any]:
        """
        Execute tool calls in parallel.

        Args:
            tool_calls: List of {"name": str, "arguments": dict}
            registry: ToolRegistry with implementations

        Returns:
            results: List of tool results in same order
        """
        def execute_single(call):
            name = call["name"]
            args = call["arguments"]

            valid, error = registry.validate_call(name, args)
            if not valid:
                return {"error": error}

            impl = registry.get_implementation(name)
            if not impl:
                return {"error": f"No implementation for {name}"}

            try:
                return impl(**args)
            except Exception as e:
                return {"error": str(e)}

        # Submit all tasks
        futures = [self.executor.submit(execute_single, call) for call in tool_calls]

        # Collect results
        results = [f.result() for f in futures]
        return results

    async def execute_parallel_async(self, tool_calls: List[Dict], registry) -> List[Any]:
        """Async version for integration with async frameworks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_parallel, tool_calls, registry)
