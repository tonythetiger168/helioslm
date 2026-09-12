"""Adaptive Reasoning V2 + MCTS - P6: 5-Level Depth + Monte Carlo Tree Search"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveReasoningController(nn.Module):
    """P6: 5-level adaptive reasoning depth"""
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
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16, batch_first=True),
            num_layers=2
        )
        self.level_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 5)
        )
        self.depth_estimator = nn.Linear(self.hidden_size, 1)

    def classify(self, hidden_states):
        encoded = self.complexity_encoder(hidden_states)
        pooled = encoded.mean(dim=1)
        logits = self.level_classifier(pooled)
        level_idx = torch.argmax(logits, dim=-1)
        level_names = ["minimal", "low", "medium", "high", "max"]
        return level_names[level_idx.item()], logits

    def get_budget(self, hidden_states, total_token_budget=4096):
        level_name, _ = self.classify(hidden_states)
        base_budget = self.LEVELS[level_name]
        confidence = torch.sigmoid(self.depth_estimator(hidden_states.mean(dim=1)))
        adjusted = int(base_budget * (0.8 + 0.4 * confidence.item()))
        return min(adjusted, int(total_token_budget * 0.5)), level_name


class MCTSNode:
    """P6: Monte Carlo Tree Search node for reasoning path exploration"""
    def __init__(self, hidden_state, parent=None, action=None):
        self.hidden_state = hidden_state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=1.414):
        choices_weights = [
            (c.value / c.visits) + c * math.sqrt((2 * math.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, action, hidden_state):
        child = MCTSNode(hidden_state, parent=self, action=action)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child


class MCTSReasoningSearch(nn.Module):
    """P6: MCTS for exploring optimal reasoning paths"""
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
        base = self.action_generator(hidden_state)
        noise = torch.randn_like(base) * 0.1
        actions = [base + noise * (i + 1) / num_actions for i in range(num_actions)]
        return actions

    def evaluate(self, hidden_state):
        return torch.tanh(self.value_head(hidden_state.mean(dim=1)))

    def search(self, initial_state, max_depth=10):
        root = MCTSNode(initial_state)
        root.untried_actions = self.generate_actions(initial_state)
        for _ in range(self.num_simulations):
            node = root
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.best_child(self.exploration_constant)
            if not node.is_fully_expanded() and node.untried_actions:
                action = node.untried_actions[0]
                new_state = action.unsqueeze(0) if action.dim() == 1 else action
                node = node.expand(action, new_state)
            value = self.evaluate(node.hidden_state).item()
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
        best = max(root.children, key=lambda c: c.visits)
        return best.hidden_state, best.value / best.visits
