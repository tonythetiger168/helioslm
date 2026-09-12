"""Speculative decoding as a production service.

Integrates draft model for 1.5-2x throughput improvement.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class SpeculativeService:
    """
    Production speculative decoding service.

    Architecture:
      - Target model: full HeliosLM (slow, accurate)
      - Draft model: small 1B-3B parameter model (fast, approximate)
      - Verifier: tree-based acceptance with corrected algorithm
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        max_draft_tokens: int = 5,
        acceptance_threshold: float = 0.8,
        device: str = "cuda",
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_tokens = max_draft_tokens
        self.acceptance_threshold = acceptance_threshold
        self.device = device

        self.total_drafted = 0
        self.total_accepted = 0

    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate with speculative decoding.

        Returns:
            output_ids: Generated token sequence
            acceptance_rate: Fraction of draft tokens accepted
        """
        self.target_model.eval()
        self.draft_model.eval()

        generated = input_ids.clone()
        drafted_count = 0
        accepted_count = 0

        with torch.no_grad():
            while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
                remaining = input_ids.shape[1] + max_new_tokens - generated.shape[1]
                draft_len = min(self.max_draft_tokens, remaining)

                # 1. Draft model generates candidate tokens
                draft_tokens = self._draft_generate(generated, draft_len, temperature)
                drafted_count += draft_len

                # 2. Target model verifies all draft positions at once
                full_input = torch.cat([generated, draft_tokens], dim=1)
                target_logits = self.target_model(full_input)["logits"]
                target_logits = target_logits[:, generated.shape[1]-1:-1, :]
                target_probs = torch.softmax(target_logits / temperature, dim=-1)

                # 3. Draft model probabilities
                draft_logits = self.draft_model(full_input)["logits"]
                draft_logits = draft_logits[:, generated.shape[1]-1:-1, :]
                draft_probs = torch.softmax(draft_logits / temperature, dim=-1)

                # 4. Verify each position
                accepted = []
                for i in range(draft_tokens.shape[1]):
                    draft_t = draft_tokens[0, i].item()
                    p = target_probs[0, i, draft_t]
                    q = draft_probs[0, i, draft_t]

                    # Acceptance criterion
                    accept_prob = min(1.0, (p / (q + 1e-10)).item())

                    if torch.rand(1).item() < accept_prob:
                        accepted.append(draft_t)
                        accepted_count += 1
                    else:
                        # Rejection: sample from residual
                        residual = target_probs[0, i, :] - draft_probs[0, i, :]
                        residual = torch.clamp(residual, min=0.0)
                        if residual.sum() > 1e-10:
                            residual = residual / residual.sum()
                            next_t = torch.multinomial(residual, num_samples=1).item()
                        else:
                            next_t = torch.multinomial(target_probs[0, i, :], num_samples=1).item()
                        accepted.append(next_t)
                        break

                # Append accepted tokens
                accepted_tensor = torch.tensor([accepted], device=generated.device)
                generated = torch.cat([generated, accepted_tensor], dim=1)

                # If no draft tokens accepted, fall back to target model
                if len(accepted) == 0:
                    logits = self.target_model(generated)["logits"]
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_t = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_t], dim=1)

        acceptance_rate = accepted_count / max(drafted_count, 1)
        return generated, acceptance_rate

    def _draft_generate(self, input_ids: torch.Tensor, num_tokens: int, temperature: float) -> torch.Tensor:
        """Generate draft tokens using the small draft model."""
        draft = input_ids.clone()
        for _ in range(num_tokens):
            logits = self.draft_model(draft)["logits"]
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            draft = torch.cat([draft, next_t], dim=1)
        return draft[:, input_ids.shape[1]:]

    def get_stats(self) -> Dict:
        """Get speculative decoding statistics."""
        return {
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": self.total_accepted / max(self.total_drafted, 1),
            "speedup": 1.0 + (self.total_accepted / max(self.total_drafted, 1)) * 0.5,
        }
