"""ViT-based decoder with configurable transformer layers."""

import math
import sys
import os

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on path for models.rae
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.rae.stage1.decoders.utils import ViTMAEConfig, ACT2FN


# =========================
# Sinusoidal Positional Embeddings
# =========================

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 2D sin/cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# =========================
# ViT Transformer Components
# =========================

class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if head_mask is None and not output_attentions:
            context_layer = nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            return (context_layer,)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTLayer(nn.Module):
    """Transformer block for ViT decoder."""
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs


# =========================
# ViT-XL Decoder
# =========================

class ViTXLDecoder(nn.Module):
    """
    Configurable ViT Decoder for AutoencoderKL.

    Default XL configuration:
    - hidden_size: 1024
    - num_layers: 24
    - num_heads: 16
    - intermediate_size: 4096
    """
    def __init__(
        self,
        encoder_hidden_size: int = 1280,
        decoder_hidden_size: int = 1024,
        decoder_num_layers: int = 24,
        decoder_num_heads: int = 16,
        decoder_intermediate_size: int = 4096,
        image_size: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.gradient_checkpointing = gradient_checkpointing

        self.decoder_embed = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=True)
        print("self.num_patches:", self.num_patches)
        print("self.patch_size:", self.patch_size)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_hidden_size), requires_grad=False
        )

        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

        decoder_config = ViTMAEConfig(
            hidden_size=decoder_hidden_size,
            num_hidden_layers=decoder_num_layers,
            num_attention_heads=decoder_num_heads,
            intermediate_size=decoder_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=out_channels,
            qkv_bias=True,
        )

        self.decoder_layers = nn.ModuleList(
            [ViTLayer(decoder_config) for _ in range(decoder_num_layers)]
        )

        self.decoder_norm = nn.LayerNorm(decoder_hidden_size, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            decoder_hidden_size, patch_size ** 2 * out_channels, bias=True
        )

        self.decoder_config = decoder_config
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.trainable_cls_token, std=0.02)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Interpolate position encodings for different resolutions."""
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        if embeddings_positions == num_positions:
            return self.decoder_pos_embed

        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        dim = self.decoder_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate latent to match expected num_patches."""
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h, w = int(l ** 0.5), int(l ** 0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target_size = (int(self.num_patches ** 0.5), int(self.num_patches ** 0.5))
        x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x

    def unpatchify(self, patchified_pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert patchified representation back to image."""
        patch_size = self.patch_size
        num_channels = self.out_channels
        num_patches_h = self.image_size // patch_size
        num_patches_w = self.image_size // patch_size

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h, num_patches_w, patch_size, patch_size, num_channels
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size, num_channels, num_patches_h * patch_size, num_patches_w * patch_size
        )
        return pixel_values

    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)

    def forward(self, hidden_states: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, H, W] latent from encoder
            interpolate_pos_encoding: whether to interpolate position encodings
        Returns:
            reconstructed image [B, C, image_size, image_size]
        """
        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        x = self.decoder_embed(hidden_states)
        x = self.interpolate_latent(x)

        cls_token = self.trainable_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed

        for layer_module in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module, hidden_states, None, False,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=False)
            hidden_states = layer_outputs[0]

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)

        logits = logits[:, 1:, :]
        pixel_values = self.unpatchify(logits)

        return pixel_values
