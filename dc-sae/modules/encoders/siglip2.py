"""SigLIP2 encoder with LoRA support."""

import contextlib

import torch
from torch import nn

from ..common import LoRALinear, LoRAConv2d, _set_lora_enabled, lora_disabled

try:
    from transformers import SiglipModel
    HAS_SIGLIP = True
except ImportError:
    HAS_SIGLIP = False
    print("Warning: transformers SiglipModel not found. SigLIP2 encoder will not be available.")


# =========================
# Normalization for SigLIP2 input
# =========================

class SigLIP2ScalingLayer(nn.Module):
    """
    SigLIP2 input normalization.
    input: [-1,1] (after dataset transform t*2-1)
    output: aligned to SigLIP2 expected normalization
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([0.0, 0.0, 0.0])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([1.0, 1.0, 1.0])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


# =========================
# LoRA for SigLIP2
# =========================

def add_lora_to_siglip2(siglip_vision_model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """
    Targets:
      - vision_model.embeddings.patch_embedding (Conv2d)
      - each encoder.layers[i].self_attn.{q_proj,k_proj,v_proj} (Linear)

    If r <= 0, do NOT add any LoRA layers (return original model unchanged).
    """
    if r <= 0:
        return siglip_vision_model

    if hasattr(siglip_vision_model.embeddings, 'patch_embedding'):
        pe = siglip_vision_model.embeddings.patch_embedding
        siglip_vision_model.embeddings.patch_embedding = LoRAConv2d(pe, r=r, alpha=alpha, dropout=dropout, enabled=True)

    for layer in siglip_vision_model.encoder.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)

    return siglip_vision_model


# =========================
# Encoder (SigLIP2 + LoRA)
# =========================

class SigLIP2Encoder2D(nn.Module):
    """SigLIP2 Encoder with LoRA support"""
    def __init__(self, model_name: str, lora_rank=32, lora_alpha=32, lora_dropout=0.0, enable_lora=True):
        super().__init__()
        if not HAS_SIGLIP:
            raise ImportError("SigLIP2 requires transformers with SiglipModel. Please install: pip install transformers>=4.36.0")

        self.encoder_type = "siglip2"
        self.model_name = model_name

        full_model = SiglipModel.from_pretrained(model_name)
        self.vision_model = full_model.vision_model
        del full_model

        self.vision_model.post_layernorm.elementwise_affine = False
        self.vision_model.post_layernorm.weight = None
        self.vision_model.post_layernorm.bias = None

        add_lora_to_siglip2(self.vision_model, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.vision_model, enable_lora)

        self.normalization_layer = SigLIP2ScalingLayer()

        self.hidden_size = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size
        self.num_register_tokens = 0

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.vision_model, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.vision_model):
            yield

    def forward(self, x: torch.Tensor):
        x = self.normalization_layer(x)
        outputs = self.vision_model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        return outputs

    def get_backbone(self):
        return self.vision_model
