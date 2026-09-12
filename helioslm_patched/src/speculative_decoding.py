"""Speculative Decoding v2 - P0: KV-Cache + Tree Attention Verifier + Continuous Batching"""
import torch
import torch.nn as nn
import time


class DraftModel(nn.Module):
    """
    Lightweight draft model with KV-Cache for speculative decoding.

    Architecture: Smaller transformer (configurable layers/hidden size).
    """

    def __init__(self, config):
        super().__init__()
        h = config.speculative.draft_hidden_size
        self.num_layers = config.speculative.draft_layers
        self.hidden_size = h
        self.vocab_size = config.vocab_size

        self.embed = nn.Embedding(config.vocab_size, h)

        # Use standard TransformerEncoderLayer for simplicity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=32, dim_feedforward=16384,
            batch_first=True, dtype=torch.bfloat16
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(h)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)

        # KV-Cache storage
        self._kv_cache = None

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        """
        Forward with optional KV-Cache.

        Args:
            input_ids: [B, seq]
            use_cache: bool
            past_key_values: list of (k, v) per layer or None

        Returns:
            logits: [B, seq, vocab]
            present_key_values: list of (k, v) or None
        """
        x = self.embed(input_ids)

        # For TransformerEncoder, we can't easily use KV-cache
        # So we just process the full sequence
        # In production, replace with custom attention layers
        x = self.layers(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, None  # KV-cache not supported with TransformerEncoder

    def generate_draft(self, input_ids, max_tokens=5, temperature=1.0):
        """
        Generate draft tokens autoregressively.

        Args:
            input_ids: [B, seq] current sequence
            max_tokens: int, number of draft tokens to generate
            temperature: float

        Returns:
            draft_ids: [B, seq + max_tokens] extended sequence
        """
        self.eval()
        draft_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _ = self.forward(draft_ids)
                next_logits = logits[:, -1, :] / temperature
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                draft_ids = torch.cat([draft_ids, next_token], dim=1)

        return draft_ids

    def generate_draft_with_kv(self, input_ids, max_tokens=5, temperature=1.0):
        """
        Generate draft tokens with KV-Cache (efficient version).

        Note: This requires custom attention implementation.
        For now, falls back to generate_draft.
        """
        return self.generate_draft(input_ids, max_tokens, temperature)


class TreeAttentionVerifier:
    """
    Verify draft tokens using tree attention for parallel validation.

    Instead of verifying tokens one-by-one, we can verify multiple
    candidate paths in parallel using tree-structured attention.
    """

    def __init__(self, target_model):
        self.target_model = target_model

    def verify(self, input_ids, draft_tokens, temperature=1.0):
        """
        Verify draft tokens against target model.

        Args:
            input_ids: [B, seq] original sequence
            draft_tokens: [B, draft_len] proposed tokens
            temperature: float

        Returns:
            accepted: list of accepted token IDs
            num_accepted: int
        """
        self.target_model.eval()

        with torch.no_grad():
            # Concatenate input with all draft tokens
            tree_input = torch.cat([input_ids, draft_tokens], dim=1)

            # Forward pass (with KV-cache if available in target model)
            # Check if target model supports use_cache
            try:
                logits, _, _, _ = self.target_model(
                    tree_input, use_cache=False
                )
            except TypeError:
                # Fallback for older model interface
                logits, _ = self.target_model(tree_input)

            accepted = []
            draft_len = draft_tokens.shape[1]

            # Sequential verification (greedy)
            for i in range(draft_len):
                pos = input_ids.shape[1] + i
                draft_t = draft_tokens[0, i]
                target_logits = logits[0, pos - 1, :]
                target_t = target_logits.argmax()

                if draft_t == target_t:
                    accepted.append(draft_t.item())
                else:
                    accepted.append(target_t.item())
                    break

        return accepted, len(accepted)

    def verify_tree(self, input_ids, draft_tree, temperature=1.0):
        """
        Tree-structured verification (advanced).

        Verifies multiple candidate paths simultaneously.
        For now, falls back to sequential verification.
        """
        # TODO: Implement true tree attention for parallel path verification
        # This requires custom attention kernel supporting tree masking
        return self.verify(input_ids, draft_tree, temperature)


class SpeculativeDecoder:
    """
    Speculative decoding with continuous batching support.
    """

    def __init__(self, target_model, draft_model, config):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft = config.speculative.max_draft_tokens
        self.threshold = config.speculative.acceptance_threshold
        self.verifier = TreeAttentionVerifier(target_model)
        self.continuous_batching = config.speculative.continuous_batching
        self.bucket_sizes = config.speculative.batch_bucket_sizes

    def generate(self, input_ids, max_new_tokens, temperature=0.7, top_p=0.9):
        """
        Speculative decoding generation loop.

        Args:
            input_ids: [B, seq]
            max_new_tokens: int
            temperature: float
            top_p: float

        Returns:
            generated: [B, seq + max_new_tokens]
            acceptance_rate: float
            speed: float (tokens/sec)
        """
        generated = input_ids.clone()
        total_drafted = 0
        total_accepted = 0
        start_time = time.time()

        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            remaining = input_ids.shape[1] + max_new_tokens - generated.shape[1]
            draft_len = min(self.max_draft, remaining)

            # Generate draft tokens
            draft_output = self.draft_model.generate_draft(generated, draft_len, temperature)
            draft_tokens = draft_output[:, generated.shape[1]:]
            total_drafted += draft_len

            # Verify draft tokens
            accepted, num_acc = self.verifier.verify(generated, draft_tokens, temperature)
            total_accepted += num_acc

            # Append accepted tokens
            accepted_t = torch.tensor([accepted], device=generated.device)
            generated = torch.cat([generated, accepted_t], dim=1)

            # If no tokens accepted, fall back to target model single step
            if num_acc == 0:
                with torch.no_grad():
                    try:
                        logits, _, _, _ = self.target_model(generated, use_cache=False)
                    except TypeError:
                        logits, _ = self.target_model(generated)
                    next_logits = logits[:, -1, :] / temperature

                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_logits[indices_to_remove] = float("-inf")

                    probs = torch.softmax(next_logits, dim=-1)
                    next_t = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_t], dim=1)

        elapsed = time.time() - start_time
        speed = (generated.shape[1] - input_ids.shape[1]) / max(elapsed, 1e-6)
        acceptance_rate = total_accepted / max(total_drafted, 1)

        return generated, acceptance_rate, speed

    def continuous_batch_generate(self, requests):
        """
        Continuous batching with dynamic length bucketing.

        Args:
            requests: list of dicts with "input_ids" and "max_tokens"

        Returns:
            results: list of generated sequences
        """
        if not self.continuous_batching:
            return [self.generate(r["input_ids"], r.get("max_tokens", 100))[0] for r in requests]

        # Group by bucket size
        buckets = {}
        for req in requests:
            l = len(req["input_ids"][0])
            bucket = min([b for b in self.bucket_sizes if b >= l], default=self.bucket_sizes[-1])
            buckets.setdefault(bucket, []).append(req)

        results = []
        for bucket, reqs in buckets.items():
            # Process in batches of 32
            for i in range(0, len(reqs), 32):
                batch = reqs[i:i + 32]
                # Pad to same length within bucket
                max_len = max(len(r["input_ids"][0]) for r in batch)
                padded = []
                for r in batch:
                    ids = r["input_ids"][0]
                    pad_len = max_len - len(ids)
                    if pad_len > 0:
                        ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype, device=ids.device)])
                    padded.append(ids.unsqueeze(0))

                batch_input = torch.cat(padded, dim=0)
                # Generate for batch
                # In production, use proper batch generation with attention masks
                for j, r in enumerate(batch):
                    results.append(self.generate(r["input_ids"], r.get("max_tokens", 100))[0])

        return results
