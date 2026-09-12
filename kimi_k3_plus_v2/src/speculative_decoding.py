"""
Speculative Decoding - Phase 1 速度增强
使用 30B 草稿模型 + 树注意力验证
预期提速: 2.5~3.5x
"""

import torch
import torch.nn as nn


class DraftModel(nn.Module):
    """30B 草稿模型"""
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.speculative.draft_hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.speculative.draft_hidden_size,
                nhead=32,
                dim_feedforward=16384,
                batch_first=True
            )
            for _ in range(config.speculative.draft_layers)
        ])
        self.norm = nn.LayerNorm(config.speculative.draft_hidden_size)
        self.lm_head = nn.Linear(config.speculative.draft_hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def generate_draft(self, input_ids, max_tokens=5):
        """快速生成候选 token"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.forward(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class TreeAttentionVerifier:
    """树注意力验证器 - 并行验证多个草稿路径"""
    def __init__(self, target_model):
        self.target_model = target_model

    def verify(self, input_ids, draft_tokens):
        """
        使用树注意力并行验证草稿 token
        返回: (accepted_tokens, num_accepted)
        """
        self.target_model.eval()
        with torch.no_grad():
            # 构建树结构输入
            tree_input = torch.cat([input_ids, draft_tokens], dim=1)
            logits, _ = self.target_model(tree_input)

            # 验证每个草稿 token
            accepted = []
            draft_len = draft_tokens.shape[1]

            for i in range(draft_len):
                pos = input_ids.shape[1] + i
                draft_token = draft_tokens[0, i]
                target_token = logits[0, pos - 1, :].argmax()

                if draft_token == target_token:
                    accepted.append(draft_token.item())
                else:
                    accepted.append(target_token.item())
                    break

            return accepted, len(accepted)


class SpeculativeDecoder:
    """推测解码主类"""
    def __init__(self, target_model, draft_model, config):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_tokens = config.speculative.max_draft_tokens
        self.acceptance_threshold = config.speculative.acceptance_threshold
        self.verifier = TreeAttentionVerifier(target_model)

    def generate(self, input_ids, max_new_tokens, temperature=0.7):
        """推测解码生成"""
        generated = input_ids.clone()
        total_drafted = 0
        total_accepted = 0

        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            remaining = input_ids.shape[1] + max_new_tokens - generated.shape[1]
            draft_len = min(self.max_draft_tokens, remaining)

            # 1. 草稿模型快速生成
            draft_output = self.draft_model.generate_draft(generated, draft_len)
            draft_tokens = draft_output[:, generated.shape[1]:]
            total_drafted += draft_len

            # 2. 目标模型并行验证
            accepted, num_accepted = self.verifier.verify(generated, draft_tokens)
            total_accepted += num_accepted

            # 3. 追加接受的 token
            accepted_tensor = torch.tensor([accepted], device=generated.device)
            generated = torch.cat([generated, accepted_tensor], dim=1)

            if num_accepted == 0:
                # 回退到目标模型单步生成
                with torch.no_grad():
                    logits, _ = self.target_model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

        acceptance_rate = total_accepted / max(total_drafted, 1)
        return generated, acceptance_rate
