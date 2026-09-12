"""Chain-of-Thought Compiler - P2 Production"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class CoTCompiler(nn.Module):
    """Production CoT compiler with structured reasoning, automatic tool selection, step verification"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_reasoning_tokens = config.cot.max_reasoning_tokens
        self.force_steps = config.cot.force_steps
        self.tools = config.cot.tools if config.cot.tool_use_enabled else []

        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True, dropout=0.1,
            ), num_layers=4,
        )
        self.step_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        self.tool_selector = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU(),
            nn.Linear(256, max(len(self.tools), 1)),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.reasoning_gate = nn.Parameter(torch.zeros(1))

    def compile_reasoning(self, hidden_states: torch.Tensor, task_difficulty: str = "medium",
                          current_step: int = 0) -> Dict:
        encoded = self.step_encoder(hidden_states)
        step_logits = self.step_classifier(encoded[:, -1, :])
        step_type = torch.argmax(step_logits, dim=-1)

        difficulty_steps = {"simple": 3, "medium": 8, "hard": 15, "expert": 25}
        max_steps = difficulty_steps.get(task_difficulty, 8)
        quality = torch.sigmoid(self.quality_head(encoded[:, -1, :]))
        should_stop = (step_type == 2) | (current_step >= max_steps) | (quality > 0.95)

        return {
            "step_type": step_type,
            "max_steps": max_steps,
            "current_step": current_step,
            "encoded": encoded,
            "quality": quality,
            "should_stop": should_stop,
        }

    def select_tool(self, hidden_states: torch.Tensor) -> Optional[str]:
        if not self.tools:
            return None
        scores = self.tool_selector(hidden_states[:, -1, :])
        tool_idx = torch.argmax(scores, dim=-1)
        if tool_idx.item() < len(self.tools):
            return self.tools[tool_idx.item()]
        return None

    def verify_step(self, prev_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(prev_state[:, -1, :], current_state[:, -1, :], dim=-1)
        return similarity

    def generate_reasoning_trace(self, initial_hidden: torch.Tensor, max_steps: int = 15) -> List[Dict]:
        trace = []
        current = initial_hidden
        step = 0
        while step < max_steps:
            result = self.compile_reasoning(current, current_step=step)
            trace.append({"step": step, "type": result["step_type"].item(), "quality": result["quality"].item()})
            if result["should_stop"].all():
                break
            if result["step_type"] == 1:
                tool = self.select_tool(current)
                trace[-1]["tool"] = tool
            step += 1
        return trace
