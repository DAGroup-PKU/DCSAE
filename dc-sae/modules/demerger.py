"""Standalone Spatial DeMerger: low-res token grid -> high-res token grid."""

import math
from typing import Optional

import torch
import torch.nn as nn


class SpatialDeMerger(nn.Module):
    """
    General-purpose spatial expansion module.

    Flow:
      1. LayerNorm + Linear proj -> hidden_dim
      2. Transformer self-attention layers (on low-res grid)  [skipped if num_transformer_layers=0]
      3. Linear proj -> expand_size^2 * output_dim
      4. Reshape [B, N, expand_size^2 * C] -> [B, N * expand_size^2, output_dim]
      5. Add learnable positional encoding on expanded grid

    When num_transformer_layers=0 the module degrades to a two-linear MLP:
        LayerNorm -> Linear(input->hidden) -> Linear(hidden -> expand^2 * output) -> reshape -> pos_embed

    Args:
        input_dim: Input token dimension (= hidden_size after fused_proj).
        hidden_dim: Internal dimension (default = input_dim).
        num_transformer_layers: Self-attention layers (0 = MLP-only, no Transformer).
        expand_size: Spatial expansion factor (expand_size=2 -> 4x tokens).
        output_dim: Per-token output dimension (default = input_dim).
        nhead: Number of attention heads (ignored when num_transformer_layers=0).
        dropout: Dropout rate in Transformer layers.
        num_output_tokens: Token count of the expanded grid, i.e. N * expand_size^2. Sizes the
            positional encoding, which is created here so it is part of the state_dict before a
            checkpoint is loaded.
    """

    def __init__(
        self,
        input_dim: int,
        num_output_tokens: int,
        hidden_dim: Optional[int] = None,
        num_transformer_layers: int = 2,
        expand_size: int = 2,
        output_dim: Optional[int] = None,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.expand_size = expand_size
        self.num_sub_tokens = expand_size ** 2
        self.num_transformer_layers = num_transformer_layers

        # Input projection (always present)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_act = nn.GELU()

        # Transformer self-attention on low-res grid (optional)
        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=self.hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_transformer_layers
            )
        else:
            self.transformer = None  # MLP-only mode

        # Expansion projection
        self.expand_proj = nn.Linear(
            self.hidden_dim, self.num_sub_tokens * self.output_dim
        )

        # Learnable positional encoding on the expanded (high-res) grid. The attribute name is kept
        # as `_pos_embed` so existing checkpoints load unchanged.
        self._pos_embed = nn.Parameter(torch.zeros(1, num_output_tokens, self.output_dim))
        nn.init.normal_(self._pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim] — low-res token grid.

        Returns:
            [B, N * expand_size^2, output_dim] — high-res token grid with positional encoding.
        """
        B, N, C = x.shape
        side = int(math.sqrt(N))
        assert side * side == N, f"Token count {N} is not a perfect square"

        # Project to hidden dim
        h = self.input_act(self.input_proj(self.input_norm(x)))  # [B, N, hidden_dim]

        # Transformer self-attention on low-res grid (skipped when num_transformer_layers=0)
        if self.transformer is not None:
            h = self.transformer(h)  # [B, N, hidden_dim]

        # Expand channels then reshape to spatial high-res grid
        expanded = self.expand_proj(h)  # [B, N, expand^2 * output_dim]
        expanded = expanded.view(B, side, side, self.expand_size, self.expand_size, self.output_dim)
        expanded = expanded.permute(0, 1, 3, 2, 4, 5).contiguous()
        out_side = side * self.expand_size
        expanded = expanded.view(B, out_side * out_side, self.output_dim)

        # Add learnable positional encoding on expanded grid
        if expanded.shape[1] != self._pos_embed.shape[1]:
            raise ValueError(
                f"SpatialDeMerger was built for {self._pos_embed.shape[1]} output tokens, "
                f"got {expanded.shape[1]}; check image_size / decode patch size"
            )
        expanded = expanded + self._pos_embed.to(dtype=expanded.dtype)

        return expanded
