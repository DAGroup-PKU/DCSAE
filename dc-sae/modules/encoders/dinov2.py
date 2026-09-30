"""DINOv2 (with registers) encoder with LoRA support."""

import contextlib

import torch
from torch import nn

from ..common import LoRALinear, LoRAConv2d, _set_lora_enabled, lora_disabled

try:
    from transformers import Dinov2WithRegistersModel
    HAS_DINOV2 = True
except ImportError:
    HAS_DINOV2 = False
    print("Warning: transformers Dinov2WithRegistersModel not found. DINOv2 encoder will not be available.")


# =========================
# Normalization for DINOv2 input
# =========================

class DINOv2ScalingLayer(nn.Module):
    """
    DINOv2 input normalization.
    input: [-1,1] (after dataset transform t*2-1)
    output: aligned to ImageNet normalization for DINOv2
    """
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


# =========================
# LoRA for DINOv2
# =========================

def add_lora_to_dinov2(dinov2_model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """
    Targets:
      - embeddings.patch_embeddings.projection (Conv2d)
      - each encoder.layer[i].attention.attention.{query,key,value} (Linear)

    If r <= 0, do NOT add any LoRA layers (return original model unchanged).
    """
    if r <= 0:
        return dinov2_model

    if hasattr(dinov2_model.embeddings, 'patch_embeddings'):
        pe = dinov2_model.embeddings.patch_embeddings.projection
        dinov2_model.embeddings.patch_embeddings.projection = LoRAConv2d(pe, r=r, alpha=alpha, dropout=dropout, enabled=True)

    for layer in dinov2_model.encoder.layer:
        attn = layer.attention.attention
        attn.query = LoRALinear(attn.query, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.key = LoRALinear(attn.key, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.value = LoRALinear(attn.value, r=r, alpha=alpha, dropout=dropout, enabled=True)

    return dinov2_model


# =========================
# Encoder (DINOv2 + LoRA)
# =========================

class DINOv2Encoder2D(nn.Module):
    """
    DINOv2 (with registers) Encoder with LoRA support.

    DINOv2-B: hidden_size=768, patch_size=14
    DINOv2-L: hidden_size=1024, patch_size=14
    DINOv2-G: hidden_size=1536, patch_size=14
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-base",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if not HAS_DINOV2:
            raise ImportError(
                "DINOv2 requires transformers with Dinov2WithRegistersModel. "
                "Please install: pip install transformers>=4.36.0"
            )

        self.encoder_type = "dinov2"
        self.model_name = model_name

        try:
            self.dinov2 = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.dinov2 = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=False)

        if normalize:
            self.dinov2.layernorm.elementwise_affine = False
            self.dinov2.layernorm.weight = None
            self.dinov2.layernorm.bias = None

        add_lora_to_dinov2(self.dinov2, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.dinov2, enable_lora)

        self.normalization_layer = DINOv2ScalingLayer()

        self.hidden_size = self.dinov2.config.hidden_size
        self.patch_size = self.dinov2.config.patch_size
        self.num_register_tokens = getattr(self.dinov2.config, 'num_register_tokens', 4)

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.dinov2, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.dinov2):
            yield

    def forward(self, x: torch.Tensor):
        if x.shape[-1] == 256 and x.shape[-2] == 256:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.normalization_layer(x)
        outputs = self.dinov2(x, output_hidden_states=True)
        return outputs

    def get_backbone(self):
        return self.dinov2


# =========================
# Deep Compression Encoder (DINOv2 + LoRA, 112x112 input)
# =========================

class Dinov2DeepCompressionEncoder2D(nn.Module):
    """
    DINOv2 (with registers) Encoder that interpolates input to 112x112.
    This yields 8x8 spatial tokens (112/14=8) instead of 16x16,
    achieving 32x spatial compression from 256x256 input.

    DINOv2-B: hidden_size=768, patch_size=14
    DINOv2-L: hidden_size=1024, patch_size=14
    DINOv2-G: hidden_size=1536, patch_size=14
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-base",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if not HAS_DINOV2:
            raise ImportError(
                "DINOv2 requires transformers with Dinov2WithRegistersModel. "
                "Please install: pip install transformers>=4.36.0"
            )

        self.encoder_type = "dinov2_deep_compression"
        self.model_name = model_name

        try:
            self.dinov2 = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.dinov2 = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=False)

        if normalize:
            self.dinov2.layernorm.elementwise_affine = False
            self.dinov2.layernorm.weight = None
            self.dinov2.layernorm.bias = None

        add_lora_to_dinov2(self.dinov2, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.dinov2, enable_lora)

        self.normalization_layer = DINOv2ScalingLayer()

        self.hidden_size = self.dinov2.config.hidden_size
        self.patch_size = self.dinov2.config.patch_size
        self.num_register_tokens = getattr(self.dinov2.config, 'num_register_tokens', 4)

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.dinov2, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.dinov2):
            yield

    def forward(self, x: torch.Tensor):
        h = (x.shape[-2] // 32) * 14
        w = (x.shape[-1] // 32) * 14
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.normalization_layer(x)
        outputs = self.dinov2(x, output_hidden_states=True)
        return outputs

    def get_backbone(self):
        return self.dinov2
