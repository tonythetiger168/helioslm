"""Chain-of-Thought Compiler v2 - P2 (Production-Ready)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTCompiler(nn.Module):
    """
    Structured reasoning step compiler with tool selection.

    Improvements:
      - Step type actually controls generation behavior
      - Tool selection with confidence scoring
      - Structured output format
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_reasoning_tokens = config.cot.max_reasoning_tokens
        self.force_steps = config.cot.force_steps
        self.tools = config.cot.tools if config.cot.tool_use_enabled else []

        # Step encoder
        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                batch_first=True, dropout=0.1
            ),
            num_layers=4
        )

        # Step classifier: [continue, use_tool, output_answer]
        self.step_classifier = nn.Linear(self.hidden_size, 3)

        # Tool selector
        self.tool_selector = nn.Linear(self.hidden_size, len(self.tools)) if self.tools else None
        self.tool_confidence = nn.Linear(self.hidden_size, 1)

        # Step budget by difficulty
        self.step_budgets = {
            "simple": 3,
            "medium": 8,
            "hard": 15,
        }

    def compile_reasoning(self, hidden_states, task_difficulty="medium"):
        """
        Compile structured reasoning steps.

        Returns:
            dict with step_type, max_steps, encoded states, tool_info
        """
        encoded = self.step_encoder(hidden_states)

        # Classify next step type
        step_logits = self.step_classifier(encoded[:, -1, :])
        step_type = torch.argmax(step_logits, dim=-1)
        step_probs = F.softmax(step_logits, dim=-1)

        # Determine max steps by difficulty
        max_steps = self.step_budgets.get(task_difficulty, 8)

        # Tool selection if step_type == 1 (use_tool)
        tool_info = None
        if self.tool_selector is not None and step_type.item() == 1:
            tool_scores = self.tool_selector(encoded[:, -1, :])
            tool_idx = torch.argmax(tool_scores, dim=-1)
            tool_conf = torch.sigmoid(self.tool_confidence(encoded[:, -1, :]))
            tool_info = {
                "tool_name": self.tools[tool_idx.item()] if tool_idx.item() < len(self.tools) else None,
                "tool_idx": tool_idx.item(),
                "confidence": tool_conf.item(),
                "all_scores": tool_scores,
            }

        return {
            "step_type": step_type,
            "step_probs": step_probs,
            "max_steps": max_steps,
            "encoded": encoded,
            "tool_info": tool_info,
            "should_continue": step_type.item() == 0,
            "should_answer": step_type.item() == 2,
        }

    def select_tool(self, hidden_states):
        """Auto-select tool with confidence."""
        if self.tool_selector is None:
            return None
        scores = self.tool_selector(hidden_states[:, -1, :])
        tool_idx = torch.argmax(scores, dim=-1)
        conf = torch.sigmoid(self.tool_confidence(hidden_states[:, -1, :]))
        return {
            "tool": self.tools[tool_idx.item()] if tool_idx.item() < len(self.tools) else None,
            "confidence": conf.item(),
            "scores": scores,
        }
