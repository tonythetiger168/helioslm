"""Learn optimal strategies from past successes and failures."""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Strategy:
    """A learned strategy for solving a type of problem."""
    name: str
    description: str
    steps: List[str]
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(total, 1)


class StrategyLearner:
    """
    Learn and select optimal strategies for different problem types.

    Maintains a library of strategies with success rates,
    selecting the best one for each new problem.
    """

    def __init__(self):
        self.strategies: Dict[str, List[Strategy]] = {}  # problem_type -> strategies

    def add_strategy(self, problem_type: str, strategy: Strategy):
        """Add a new strategy for a problem type."""
        if problem_type not in self.strategies:
            self.strategies[problem_type] = []
        self.strategies[problem_type].append(strategy)

    def select_strategy(self, problem_type: str) -> Optional[Strategy]:
        """Select the best strategy for a problem type."""
        strategies = self.strategies.get(problem_type, [])
        if not strategies:
            return None

        # Select by highest success rate
        return max(strategies, key=lambda s: s.success_rate)

    def update_outcome(self, problem_type: str, strategy_name: str, success: bool):
        """Update strategy outcome."""
        strategies = self.strategies.get(problem_type, [])
        for strategy in strategies:
            if strategy.name == strategy_name:
                if success:
                    strategy.success_count += 1
                else:
                    strategy.failure_count += 1
                break

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        stats = {}
        for problem_type, strategies in self.strategies.items():
            stats[problem_type] = {
                "num_strategies": len(strategies),
                "best_strategy": max(strategies, key=lambda s: s.success_rate).name if strategies else None,
                "avg_success_rate": sum(s.success_rate for s in strategies) / max(len(strategies), 1),
            }
        return stats
