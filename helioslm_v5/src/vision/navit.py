"""NaViT - Native Vision Transformer with Arbitrary Resolution

Supports variable aspect ratios and resolutions without resizing distortion.
Batches of mixed image sizes are handled with padding-based batching
(see ``forward_packed``); true nested-tensor sequence packing is NOT
implemented.
"""
import torch
import torch.nn as nn


class NaViTEncoder(nn.Module):
    """
    Native Vision Transformer supporting arbitrary image resolutions.

    Key features:
      - No fixed image size constraint
      - Dynamic patch grid based on input resolution
      - Padding-based batching for mixed-size image lists
        (``forward_packed``); this is zero-padding + key padding mask,
        not true sequence packing.
      - Aspect ratio bucketing

    Position encoding:
      - Factorized 2D learned position embeddings:
        ``row_embed[i] + col_embed[j]`` for patch (i, j). Each table has
        ``max_grid`` entries, read from ``config.multimodal.vision_max_grid``
        when present (fallback default 64, i.e. up to 64x64 patches =
        896x896 px at patch size 14).
      - No position-embedding interpolation is performed. If the patch
        grid of an input exceeds ``max_grid`` in either dimension, a
        ``ValueError`` is raised (increase ``vision_max_grid`` in the
        config to support larger images).
    """

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.multimodal.vision_patch_size
        self.hidden_size = config.multimodal.vision_hidden_size
        self.num_layers = config.multimodal.vision_num_layers
        self.num_heads = config.multimodal.vision_num_heads
        self.max_grid = getattr(config.multimodal, "vision_max_grid", 64)

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        # Factorized 2D position embedding: row table + column table,
        # each of length max_grid (no collisions for any grid <= max_grid).
        self.row_embed = nn.Embedding(self.max_grid, self.hidden_size)
        self.col_embed = nn.Embedding(self.max_grid, self.hidden_size)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.norm = nn.LayerNorm(self.hidden_size)

    def _position_embedding(self, h_p: int, w_p: int, device, dtype) -> torch.Tensor:
        """Return [h_p * w_p, hidden] factorized 2D position embedding."""
        if h_p > self.max_grid or w_p > self.max_grid:
            raise ValueError(
                f"Patch grid ({h_p}x{w_p}) exceeds max_grid={self.max_grid}. "
                f"Increase `config.multimodal.vision_max_grid` (and retrain/finetune "
                f"the position tables) to support larger images; position-embedding "
                f"interpolation is not implemented."
            )
        rows = torch.arange(h_p, device=device)
        cols = torch.arange(w_p, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        pos = self.row_embed(grid_r.reshape(-1)) + self.col_embed(grid_c.reshape(-1))
        return pos.to(dtype)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: [B, 3, H, W]. All images in the batch must have the
                SAME H, W (tensor constraint). For mixed-size images use
                ``forward_packed``.

        Returns:
            features: [B, num_patches, hidden] (after final LayerNorm)
        """
        B, C, H, W = images.shape

        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"image size ({H}x{W}) is smaller than patch_size="
                f"{self.patch_size}: it would produce zero patches"
            )

        # Extract patches
        patches = self.patch_embed(images)  # [B, hidden, H/p, W/p]
        B, hidden, h_p, w_p = patches.shape

        # Flatten patches
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]

        # Factorized 2D position encoding (unique per (row, col))
        pos_emb = self._position_embedding(h_p, w_p, images.device, patches.dtype)
        patches = patches + pos_emb.unsqueeze(0)

        # Encode
        features = self.encoder(patches)
        return self.norm(features)

    def forward_packed(self, images_list: list):
        """
        Process images of different sizes in one batch.

        NOTE: despite the name, this is padding-based batching (pad to the
        longest patch sequence + key padding mask), not true sequence
        packing with nested tensors.

        Args:
            images_list: list of [3, H_i, W_i] tensors (mixed sizes allowed)

        Returns:
            features: [N, max_num_patches, hidden] — padded batch, after the
                same final LayerNorm as ``forward``. Padded positions contain
                norm(pad) outputs and should be ignored via ``mask``.
            mask: [N, max_num_patches] bool, True for real patches.
        """
        if len(images_list) == 0:
            raise ValueError(
                "forward_packed requires a non-empty list of images, got []"
            )
        all_patches = []

        for img in images_list:
            C, H, W = img.shape
            if H < self.patch_size or W < self.patch_size:
                raise ValueError(
                    f"image size ({H}x{W}) is smaller than patch_size="
                    f"{self.patch_size}: it would produce zero patches"
                )
            patches = self.patch_embed(img.unsqueeze(0))  # [1, hidden, h_p, w_p]
            _, hidden, h_p, w_p = patches.shape
            patches = patches.flatten(2).transpose(1, 2).squeeze(0)  # [num_patches, hidden]

            pos_emb = self._position_embedding(h_p, w_p, img.device, patches.dtype)
            patches = patches + pos_emb

            all_patches.append(patches)

        # Pad batch on the input device/dtype
        ref = all_patches[0]
        max_len = max(p.shape[0] for p in all_patches)
        packed = torch.zeros(len(all_patches), max_len, self.hidden_size,
                             device=ref.device, dtype=ref.dtype)
        mask = torch.zeros(len(all_patches), max_len, dtype=torch.bool, device=ref.device)

        for i, patches in enumerate(all_patches):
            packed[i, :patches.shape[0]] = patches
            mask[i, :patches.shape[0]] = True

        # Encode with padding mask; apply the same final norm as forward()
        features = self.encoder(packed, src_key_padding_mask=~mask)
        return self.norm(features), mask
