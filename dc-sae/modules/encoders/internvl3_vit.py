"""InternVL3-ViT encoder (InternViT from InternVL3 checkpoints)."""

import contextlib
import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class InternVL3ScalingLayer(nn.Module):
    """ImageNet normalization: [-1, 1] → ImageNet-normed."""

    def __init__(self):
        super().__init__()
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225])
        shift = 1 - 2 * imagenet_mean
        scale = 2 * imagenet_std
        self.register_buffer("shift", shift[None, :, None, None])
        self.register_buffer("scale", scale[None, :, None, None])

    def forward(self, x):
        return (x + self.shift) / self.scale


def _pixel_shuffle_downsample(x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """
    InternVL-style pixel shuffle: merge 2x2 spatial blocks into channel dim.

    Input:  [B, H, W, C]  e.g. [B, 16, 16, 1024]
    Output: [B, H/2, W/2, C*4]  e.g. [B, 8, 8, 4096]

    Matches InternVLChatModel.pixel_shuffle (ps_version='v2').
    """
    n, w, h, c = x.size()
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    new_c = int(c / (scale_factor * scale_factor))
    x = x.view(n, w, new_h, int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, new_h, new_w, new_c)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class InternVL3ViTEncoder2D(nn.Module):
    """
    InternViT encoder extracted from InternVL3 checkpoints.

    Loads the full InternVLChatModel via trust_remote_code, keeps only
    vision_model, deletes the LLM and mlp1 to save memory.

    With pixel_shuffle (default):
        256x256 → resize 224 → ViT → [B, 257, 1024] → drop CLS → [B, 256, 1024]
        → spatial [B, 16, 16, 1024] → pixel_shuffle → [B, 8, 8, 4096]
        → flatten → [B, 64, 4096]
        hidden_size=4096, decode_patch_size=32 (256/8), 64 tokens

    Same output shape as Qwen3-ViT (64 tokens x 4096 dim, decode_patch_size=32).
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        resize_target: int = 224,
        lora_rank: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        enable_lora: bool = False,
        freeze_encoder: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.encoder_type = "internvl3"
        self.model_name = model_name

        print(f"Loading InternVL3 from {model_name}...")
        try:
            from transformers import AutoModel
            full_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, AttributeError):
            from transformers import AutoModel
            full_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=False,
            )

        self.vision_model = full_model.vision_model

        if hasattr(full_model, "language_model"):
            del full_model.language_model
        if hasattr(full_model, "mlp1"):
            del full_model.mlp1
        if hasattr(full_model, "lm_head"):
            del full_model.lm_head
        del full_model
        torch.cuda.empty_cache()

        if enable_lora and lora_rank > 0:
            print("Warning: LoRA is not wired for InternVL3-ViT yet; running without LoRA adapters.")

        if freeze_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.normalization_layer = InternVL3ScalingLayer()

        vision_cfg = self.vision_model.config
        self._vit_patch_size = vision_cfg.patch_size  # 14
        self._vit_hidden_size = vision_cfg.hidden_size  # 1024
        self._vit_image_size = vision_cfg.image_size  # 448

        self.downsample_ratio = 0.5
        self.hidden_size = int(self._vit_hidden_size / (self.downsample_ratio ** 2))  # 4096
        self.patch_size = self._vit_patch_size  # 14 (original)
        self.num_register_tokens = 0
        self.has_cls_token = True

        self._resize_target = self._find_resize_target(resize_target)

        print(f"InternVL3-ViT Encoder initialized:")
        print(f"  - vit_hidden_size: {self._vit_hidden_size}")
        print(f"  - output hidden_size (after pixel_shuffle): {self.hidden_size}")
        print(f"  - vit_patch_size: {self._vit_patch_size}")
        print(f"  - resize_target: {self._resize_target}")

    def _find_resize_target(self, resize_target: int) -> int:
        """Validate resize target for a square InternVL3 token grid."""
        if resize_target <= 0:
            raise ValueError(f"resize_target must be positive, got: {resize_target}")
        if resize_target % self._vit_patch_size != 0:
            raise ValueError(
                f"resize_target ({resize_target}) must be divisible by patch_size "
                f"({self._vit_patch_size})"
            )
        vit_side = resize_target // self._vit_patch_size
        if vit_side % 2 != 0:
            raise ValueError(
                f"resize_target ({resize_target}) gives vit_side={vit_side}, "
                "which must be even for 2x pixel_shuffle downsample"
            )
        return resize_target

    def set_lora_enabled(self, enabled: bool):
        pass

    @contextlib.contextmanager
    def lora_disabled(self):
        yield

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] tensor, range [-1, 1]

        Returns:
            SimpleNamespace with last_hidden_state: [B, N, 4096]
            where N = (resize_target / patch_size / 2)^2
        """
        B, C, H, W = x.shape

        if H != self._resize_target or W != self._resize_target:
            x = F.interpolate(x, size=(self._resize_target, self._resize_target),
                              mode="bilinear", align_corners=False)

        x = self.normalization_layer(x)
        x = x.to(dtype=self.vision_model.embeddings.patch_embedding.weight.dtype)

        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.vision_model.parameters())
        ):
            outputs = self.vision_model(
                pixel_values=x,
                output_hidden_states=False,
                return_dict=True,
            )

        vit_embeds = outputs.last_hidden_state  # [B, N+1, 1024] (with CLS)
        vit_embeds = vit_embeds[:, 1:, :]  # drop CLS → [B, N, 1024]

        side = int(math.sqrt(vit_embeds.shape[1]))
        assert side * side == vit_embeds.shape[1], (
            f"Token count {vit_embeds.shape[1]} is not a perfect square"
        )

        spatial = vit_embeds.reshape(B, side, side, self._vit_hidden_size)
        spatial = _pixel_shuffle_downsample(spatial, scale_factor=self.downsample_ratio)
        merged = spatial.reshape(B, -1, self.hidden_size)

        return SimpleNamespace(last_hidden_state=merged)

    def get_backbone(self):
        return self.vision_model
