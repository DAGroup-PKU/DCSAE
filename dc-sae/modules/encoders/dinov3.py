"""DINOv3 encoder with LoRA support."""

import contextlib
import sys
import os

import torch
from torch import nn

# Ensure project root is on path for models.dino_v3
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel
from ..common import LoRALinear, LoRAConv2d, _set_lora_enabled, lora_disabled


# =========================
# Normalization for DINO input
# =========================

class ScalingLayer2D(nn.Module):
    """
    input: [-1,1] (after dataset transform t*2-1)
    output: aligned to ImageNet normalization for DINO (approx)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


# =========================
# LoRA for DINOv3
# =========================

def add_lora_to_dinov3(dino_model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """
    Targets:
      - dino_model.embeddings.patch_embeddings (Conv2d)
      - each block.attention.{q_proj,k_proj,v_proj} (Linear)

    If r <= 0, do NOT add any LoRA layers (return original model unchanged).
    """
    if r <= 0:
        return dino_model

    pe = dino_model.embeddings.patch_embeddings
    dino_model.embeddings.patch_embeddings = LoRAConv2d(pe, r=r, alpha=alpha, dropout=dropout, enabled=True)

    for blk in dino_model.layer:
        attn = blk.attention
        attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)

    return dino_model


# =========================
# Encoder (DINOv3 + LoRA)
# =========================

class Encoder2D(nn.Module):
    """DINOv3 Encoder with LoRA support"""
    def __init__(self, dinov3_model_dir: str, lora_rank=32, lora_alpha=32, lora_dropout=0.0, enable_lora=True):
        super().__init__()
        self.encoder_type = "dinov3"
        self.dino_v3 = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path=dinov3_model_dir,
            use_safetensors=True,
        )
        add_lora_to_dinov3(self.dino_v3, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.dino_v3, enable_lora)

        self.normalization_layer = ScalingLayer2D()

        self.hidden_size = self.dino_v3.config.hidden_size
        self.patch_size = self.dino_v3.config.patch_size
        self.num_register_tokens = getattr(self.dino_v3.config, 'num_register_tokens', 4)

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.dino_v3, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.dino_v3):
            yield

    def forward(self, x: torch.Tensor):
        x = self.normalization_layer(x)
        return self.dino_v3(pixel_values=x)

    def get_backbone(self):
        return self.dino_v3
