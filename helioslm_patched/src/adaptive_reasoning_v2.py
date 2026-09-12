"""Adaptive Reasoning V2 + MCTS - P6 (Production-Ready)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveReasoningController(nn.Module):
    """
    5-level adaptive reasoning depth controller.

    Improvements:
      - Level embedding for fusion into model
      - Confidence-based budget adjustment
    """

    LEVELS = {
        "minimal": 128,
        "low": 512,
        "medium": 2048,
        "high": 4096,
        "max": 8192,
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.adaptive = config.adaptive_reasoning_v2.enabled

        self.complexity_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                batch_first=True, dropout=0.1
            ),
            num_layers=2
        )

        self.level_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)
        )

        self.depth_estimator = nn.Linear(self.hidden_size, 1)

        # Level embedding for model fusion
        self.level_embed = nn.Embedding(5, self.hidden_size)

    def classify(self, hidden_states):
        """Classify reasoning complexity level."""
        encoded = self.complexity_encoder(hidden_states)
        pooled = encoded.mean(dim=1)
        logits = self.level_classifier(pooled)
        level_idx = torch.argmax(logits, dim=-1)
        level_names = ["minimal", "low", "medium", "high", "max"]
        return level_names[level_idx.item()], logits, level_idx

    def get_budget(self, hidden_states, total_token_budget=4096):
        """
        Get reasoning budget with confidence adjustment.

        Returns:
            budget: int
            level_name: str
            level_idx: int
            confidence: float
        """
        level_name, _, level_idx = self.classify(hidden_states)
        base_budget = self.LEVELS[level_name]
        confidence = torch.sigmoid(self.depth_estimator(hidden_states.mean(dim=1)))
        adjusted = int(base_budget * (0.8 + 0.4 * confidence.item()))
        budget = min(adjusted, int(total_token_budget * 0.5))
        return budget, level_name, level_idx.item(), confidence.item()


class MCTSNode:
    """Monte Carlo Tree Search node."""

    def __init__(self, hidden_state, parent=None, action=None):
        self.hidden_state = hidden_state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c=1.414):
        if not self.children:
            return None
        choices_weights = [
            (c.value / max(c.visits, 1)) + c * math.sqrt((2 * math.log(max(self.visits, 1)) / max(c.visits, 1)))
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, action, hidden_state):
        child = MCTSNode(hidden_state, parent=self, action=action)
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        self.children.append(child)
        return child


class MCTSReasoningSearch(nn.Module):
    """
    MCTS for reasoning path exploration.

    Improvements:
      - Better action generation using model predictions
      - Value network for state evaluation
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_simulations = config.adaptive_reasoning_v2.mcts_simulations
        self.exploration_constant = config.adaptive_reasoning_v2.mcts_c_puct

        self.action_generator = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, self.hidden_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def generate_actions(self, hidden_state, num_actions=4):
        """Generate candidate reasoning actions."""
        base = self.action_generator(hidden_state)
        noise = torch.randn_like(base) * 0.1
        actions = [base + noise * (i + 1) / num_actions for i in range(num_actions)]
        return actions

    def evaluate(self, hidden_state):
        """Evaluate state value."""
        return torch.tanh(self.value_head(hidden_state.mean(dim=1)))

    def search(self, initial_state, max_depth=10):
        """
        Run MCTS search.

        Returns:
            best_state: best child hidden state
            best_value: average value of best path
        """
        root = MCTSNode(initial_state)
        root.untried_actions = self.generate_actions(initial_state)

        for _ in range(self.num_simulations):
            node = root

            # Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.exploration_constant)
                if node is None:
                    break

            if node is None:
                continue

            # Expansion
            if not node.is_fully_expanded() and node.untried_actions:
                action = node.untried_actions[0]
                new_state = action.unsqueeze(0) if action.dim() == 1 else action
                node = node.expand(action, new_state)

            # Evaluation
            value = self.evaluate(node.hidden_state).item()

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        if not root.children:
            return initial_state, 0.0

        best = max(root.children, key=lambda c: c.visits)
        return best.hidden_state, best.value / max(best.visits, 1)
