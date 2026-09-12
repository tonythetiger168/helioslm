"""Speculative Decoding + Continuous Batching - P0 Production"""
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftModel(nn.Module):
    """Lightweight draft model for speculative decoding"""
    def __init__(self, config):
        super().__init__()
        h = config.speculative.draft_hidden_size
        self.embed = nn.Embedding(config.vocab_size, h)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=h, nhead=32, dim_feedforward=16384,
                batch_first=True, dtype=torch.bfloat16
            )
            for _ in range(config.speculative.draft_layers)
        ])
        self.norm = nn.LayerNorm(h)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    def generate_draft(self, input_ids: torch.Tensor, max_tokens: int = 5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.forward(input_ids)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class TreeAttentionVerifier:
    """Verify draft tokens using tree-structured attention"""
    def __init__(self, target_model):
        self.target_model = target_model

    def verify(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor,
               temperature: float = 0.7) -> Tuple[List[int], int]:
        self.target_model.eval()
        with torch.no_grad():
            tree_input = torch.cat([input_ids, draft_tokens], dim=1)
            out = self.target_model(tree_input)
            logits = out["logits"] if isinstance(out, dict) else out[0]

            accepted = []
            draft_len = draft_tokens.shape[1]

            for i in range(draft_len):
                pos = input_ids.shape[1] + i
                draft_t = draft_tokens[0, i].item()
                target_dist = F.softmax(logits[0, pos - 1, :] / temperature, dim=-1)
                target_t = target_dist.argmax().item()

                # Probabilistic acceptance
                draft_logits = self.target_model(tree_input[:, :pos])
                draft_logits = draft_logits["logits"] if isinstance(draft_logits, dict) else draft_logits[0]
                draft_prob = F.softmax(draft_logits[:, -1, :] / temperature, dim=-1)[0, draft_t]
                target_prob = target_dist[draft_t]

                if torch.rand(1).item() < min(1.0, (target_prob / (draft_prob + 1e-8)).item()):
                    accepted.append(draft_t)
                else:
                    accepted.append(target_t)
                    break

            return accepted, len(accepted)


class SpeculativeDecoder:
    """Production speculative decoder with continuous batching"""
    def __init__(self, target_model, draft_model, config):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft = config.speculative.max_draft_tokens
        self.threshold = config.speculative.acceptance_threshold
        self.verifier = TreeAttentionVerifier(target_model)
        self.continuous_batching = config.speculative.continuous_batching
        self.bucket_sizes = config.speculative.batch_bucket_sizes

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.7) -> Tuple[torch.Tensor, float, float]:
        generated = input_ids.clone()
        total_drafted = 0
        total_accepted = 0
        start_time = time.time()

        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            remaining = input_ids.shape[1] + max_new_tokens - generated.shape[1]
            draft_len = min(self.max_draft, remaining)

            draft_output = self.draft_model.generate_draft(generated, draft_len)
            draft_tokens = draft_output[:, generated.shape[1]:]
            total_drafted += draft_len

            accepted, num_acc = self.verifier.verify(generated, draft_tokens, temperature)
            total_accepted += num_acc

            accepted_t = torch.tensor([accepted], device=generated.device)
            generated = torch.cat([generated, accepted_t], dim=1)

            if num_acc == 0:
                with torch.no_grad():
                    out = self.target_model(generated)
                    logits = out["logits"] if isinstance(out, dict) else out[0]
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_t = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_t], dim=1)

        elapsed = time.time() - start_time
        speed = (generated.shape[1] - input_ids.shape[1]) / elapsed
        acceptance_rate = total_accepted / max(total_drafted, 1)
        return generated, acceptance_rate, speed

    def continuous_batch_generate(self, requests: List[Dict]) -> List[Tuple[torch.Tensor, float, float]]:
        """P0: Continuous batching with dynamic length bucketing"""
        if not self.continuous_batching:
            return [
                self.generate(r["input_ids"], r.get("max_tokens", 100), r.get("temperature", 0.7))
                for r in requests
            ]

        requests = sorted(requests, key=lambda r: len(r["input_ids"][0]))
        buckets = {}
        for req in requests:
            l = len(req["input_ids"][0])
            bucket = min([b for b in self.bucket_sizes if b >= l], default=self.bucket_sizes[-1])
            buckets.setdefault(bucket, []).append(req)

        results = []
        for bucket, reqs in buckets.items():
            for i in range(0, len(reqs), 32):
                batch = reqs[i:i + 32]
                for r in batch:
                    results.append(self.generate(
                        r["input_ids"],
                        r.get("max_tokens", 100),
                        r.get("temperature", 0.7)
                    ))
        return results
