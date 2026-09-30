import json
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


def _channel_average(x: torch.Tensor, target_channels: int) -> torch.Tensor:
    if x.shape[1] == target_channels:
        return x

    batch, channels, height, width = x.shape
    if channels % target_channels != 0:
        raise ValueError(
            f"Cannot average channels from {channels} to {target_channels}: not divisible"
        )

    factor = channels // target_channels
    return x.reshape(batch, target_channels, factor, height, width).mean(dim=2)


def _build_act(act_fn: Optional[str]) -> nn.Module:
    if act_fn is None:
        return nn.Identity()

    act_name = str(act_fn).lower()
    if act_name == "silu":
        return nn.SiLU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported act_fn: {act_fn}")


def _pool_to_patch_grid(
    x: torch.Tensor,
    patch_size: int,
    current_stride: int,
    shortcut: Optional[str] = None,
) -> torch.Tensor:
    if patch_size < current_stride:
        raise ValueError(
            f"patch_size ({patch_size}) must be >= current_stride ({current_stride})"
        )

    if patch_size == current_stride:
        return x

    if patch_size % current_stride != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be divisible by current_stride ({current_stride})"
        )

    pool_size = patch_size // current_stride
    pooled = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)
    if shortcut is None:
        return pooled
    if shortcut == "averaging":
        residual = F.pixel_unshuffle(x, pool_size)
        residual = _channel_average(residual, x.shape[1])
        return pooled + residual
    raise ValueError(f"Unsupported patch pooling shortcut: {shortcut}")


class HFResBlock(nn.Module):
    def __init__(self, channels: int, norm_num_groups: int = 8):
        super().__init__()
        groups = _resolve_group_count(channels, norm_num_groups)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class HFEncoder(nn.Module):
    """
    Extract patch-aligned HF tokens from image space.
    Reference: diffusers VAE Encoder with down_blocks + mid_block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        patch_size: int = 16,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
        residual_downsample: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        self.residual_downsample = residual_downsample
        self.out_channels = out_channels
        self._block_out_channels = list(block_out_channels)

        num_downsample = len(down_block_types) - 1
        self.down_factor = 2 ** num_downsample

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for i, down_block in enumerate(self.down_blocks):
            is_final = i == len(self.down_blocks) - 1
            if self.residual_downsample and not is_final:
                out_ch = self._block_out_channels[i]
                residual = F.pixel_unshuffle(h, 2)
                residual = _channel_average(residual, out_ch)
                h = down_block(h) + residual
            else:
                h = down_block(h)

        h = self.mid_block(h)
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)
        return _pool_to_patch_grid(
            h,
            patch_size=self.patch_size,
            current_stride=self.down_factor,
            shortcut="averaging" if self.residual_downsample else None,
        )

    @classmethod
    def from_config(cls, config_path: str, patch_size: int = 14):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "down_block_types" in config:
            config["down_block_types"] = tuple(config["down_block_types"])
        if "block_out_channels" in config:
            config["block_out_channels"] = tuple(config["block_out_channels"])
        config["patch_size"] = patch_size
        return cls(**config)


class DCAEDownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: Optional[str] = "averaging",
    ):
        super().__init__()
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pixel_unshuffle(x, 2)
        h = self.conv(h)

        if self.shortcut is None:
            return h
        if self.shortcut == "averaging":
            return h + _channel_average(F.pixel_unshuffle(x, 2), self.conv.out_channels)
        raise ValueError(f"Unsupported downsample shortcut: {self.shortcut}")


class DCAEProjectOutBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        act_fn: Optional[str] = "silu",
        shortcut: Optional[str] = "averaging",
    ):
        super().__init__()
        self.shortcut = shortcut
        self.norm = nn.GroupNorm(
            num_groups=_resolve_group_count(in_channels, norm_num_groups),
            num_channels=in_channels,
            eps=1e-6,
        )
        self.act = _build_act(act_fn)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.act(self.norm(x)))
        if self.shortcut is None:
            return h
        if self.shortcut == "averaging":
            return h + _channel_average(x, self.conv.out_channels)
        raise ValueError(f"Unsupported project_out shortcut: {self.shortcut}")


class DCAELikeHFEncoder(nn.Module):
    """
    Stage-based HF encoder inspired by the DC-AE encoder structure.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        patch_size: int = 16,
        width_list: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        depth_list: Tuple[int, ...] = (2, 2, 2, 2, 2),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        downsample_match_channel: bool = True,
        downsample_shortcut: Optional[str] = "averaging",
        out_shortcut: Optional[str] = "averaging",
        patch_pool_shortcut: Optional[str] = "averaging",
    ):
        super().__init__()
        if len(width_list) == 0:
            raise ValueError("width_list must not be empty")
        if len(width_list) != len(depth_list):
            raise ValueError(
                f"width_list/depth_list length mismatch: {len(width_list)} vs {len(depth_list)}"
            )

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_pool_shortcut = patch_pool_shortcut
        self.width_list = tuple(width_list)
        self.depth_list = tuple(depth_list)
        self.down_factor = 2 ** max(len(self.width_list) - 1, 0)

        self.project_in = nn.Conv2d(in_channels, self.width_list[0], kernel_size=3, stride=1, padding=1)

        stages = []
        for stage_id, (width, depth) in enumerate(zip(self.width_list, self.depth_list)):
            stage_ops = [HFResBlock(width, norm_num_groups=norm_num_groups) for _ in range(depth)]
            if stage_id < len(self.width_list) - 1:
                next_width = self.width_list[stage_id + 1]
                if not downsample_match_channel and next_width != width:
                    raise ValueError(
                        "downsample_match_channel=False requires adjacent stage widths to match, "
                        f"got {width} -> {next_width}"
                    )
                stage_ops.append(
                    DCAEDownsampleBlock(
                        in_channels=width,
                        out_channels=next_width if downsample_match_channel else width,
                        shortcut=downsample_shortcut,
                    )
                )
            stages.append(nn.Sequential(*stage_ops) if stage_ops else nn.Identity())
        self.stages = nn.ModuleList(stages)

        self.project_out = DCAEProjectOutBlock(
            in_channels=self.width_list[-1],
            out_channels=out_channels,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            shortcut=out_shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project_in(x)
        for stage in self.stages:
            h = stage(h)
        h = self.project_out(h)
        return _pool_to_patch_grid(
            h,
            patch_size=self.patch_size,
            current_stride=self.down_factor,
            shortcut=self.patch_pool_shortcut,
        )

    @classmethod
    def from_config(cls, config_path: str, patch_size: int = 14):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "width_list" in config:
            config["width_list"] = tuple(config["width_list"])
        if "depth_list" in config:
            config["depth_list"] = tuple(config["depth_list"])
        config["patch_size"] = patch_size
        return cls(**config)


class ViTHFEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHFEncoder(nn.Module):
    """ViT-based HF encoder. A lightweight Vision Transformer for high-frequency feature extraction."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        patch_size: int = 32,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))  # will be resized in forward
        self._pos_embed_initialized = False

        self.blocks = nn.ModuleList([
            ViTHFEncoderBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def _get_pos_embed(self, num_patches: int, h: int, w: int) -> torch.Tensor:
        if self.pos_embed.shape[1] == num_patches:
            return self.pos_embed
        if not self._pos_embed_initialized:
            pos = torch.zeros(1, num_patches, self.hidden_size, device=self.pos_embed.device, dtype=self.pos_embed.dtype)
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos_embed = nn.Parameter(pos)
            self._pos_embed_initialized = True
            return self.pos_embed
        # interpolate for different resolutions
        old_len = self.pos_embed.shape[1]
        old_h = old_w = int(math.sqrt(old_len))
        embed = self.pos_embed.reshape(1, old_h, old_w, self.hidden_size).permute(0, 3, 1, 2)
        embed = F.interpolate(embed, size=(h, w), mode="bilinear", align_corners=False)
        return embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h = self.patch_embed(x)  # [B, hidden_size, H/ps, W/ps]
        gh, gw = h.shape[2], h.shape[3]
        h = h.flatten(2).transpose(1, 2)  # [B, N, hidden_size]
        h = h + self._get_pos_embed(gh * gw, gh, gw)
        for block in self.blocks:
            h = block(h)
        h = self.proj(self.norm(h))  # [B, N, out_channels]
        h = h.transpose(1, 2).reshape(B, self.out_channels, gh, gw)
        return h

    @classmethod
    def from_config(cls, config_path: str, patch_size: int = 14):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["patch_size"] = patch_size
        return cls(**config)


def normalize_hf_encoder_type(hf_encoder_type: Optional[str]) -> str:
    value = str(hf_encoder_type or "cnn").strip().lower().replace("_", "-").replace(" ", "-")
    if value in {"cnn", "hf-encoder", "default"}:
        return "cnn"
    if value in {"dc-ae-hf-encoder", "dc-ae", "dcae", "dc-ae-like"}:
        return "dc-ae-hf-encoder"
    if value in {"vit", "vit-hf-encoder", "vit-hf"}:
        return "vit"
    raise ValueError(
        "Unsupported hf_encoder_type. Expected one of {'cnn', 'dc-ae-hf-encoder', 'vit'}, "
        f"got: {hf_encoder_type}"
    )


def build_hf_encoder(
    hf_encoder_type: Optional[str],
    in_channels: int,
    out_channels: int,
    patch_size: int,
    config_path: Optional[str] = None,
) -> nn.Module:
    encoder_type = normalize_hf_encoder_type(hf_encoder_type)
    if encoder_type == "cnn":
        if config_path is not None:
            return HFEncoder.from_config(config_path, patch_size=patch_size)
        return HFEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
        )

    if encoder_type == "dc-ae-hf-encoder":
        if config_path is not None:
            return DCAELikeHFEncoder.from_config(config_path, patch_size=patch_size)
        return DCAELikeHFEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
        )

    # encoder_type == "vit"
    if config_path is not None:
        return ViTHFEncoder.from_config(config_path, patch_size=patch_size)
    return ViTHFEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
    )


__all__ = [
    "HFResBlock",
    "HFEncoder",
    "DCAEDownsampleBlock",
    "DCAEProjectOutBlock",
    "DCAELikeHFEncoder",
    "ViTHFEncoderBlock",
    "ViTHFEncoder",
    "normalize_hf_encoder_type",
    "build_hf_encoder",
]
