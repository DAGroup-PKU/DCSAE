"""
Encoder modules for different vision backbone families.

Available encoders:
- Encoder2D (DINOv3)
- DINOv2Encoder2D
- SigLIP2Encoder2D
- Qwen3ViTEncoder2D
"""

from .dinov3 import Encoder2D, ScalingLayer2D, add_lora_to_dinov3
from .dinov2 import DINOv2Encoder2D, Dinov2DeepCompressionEncoder2D, DINOv2ScalingLayer, add_lora_to_dinov2
from .siglip2 import SigLIP2Encoder2D, SigLIP2ScalingLayer, add_lora_to_siglip2
from .qwen3_vit import (
    Qwen3ViTEncoder2D,
    Qwen3ViTScalingLayer,
    Qwen3ViTAvgPoolEncoder2D,
    Qwen3ViTTransformerMergerEncoder2D,
)
from .internvl3_vit import InternVL3ViTEncoder2D, InternVL3ScalingLayer
from .hf import HFResBlock, HFEncoder, DCAELikeHFEncoder, build_hf_encoder, normalize_hf_encoder_type

__all__ = [
    "Encoder2D",
    "ScalingLayer2D",
    "add_lora_to_dinov3",
    "DINOv2Encoder2D",
    "Dinov2DeepCompressionEncoder2D",
    "DINOv2ScalingLayer",
    "add_lora_to_dinov2",
    "SigLIP2Encoder2D",
    "SigLIP2ScalingLayer",
    "add_lora_to_siglip2",
    "Qwen3ViTEncoder2D",
    "Qwen3ViTScalingLayer",
    "Qwen3ViTAvgPoolEncoder2D",
    "Qwen3ViTTransformerMergerEncoder2D",
    "InternVL3ViTEncoder2D",
    "InternVL3ScalingLayer",
    "HFResBlock",
    "HFEncoder",
    "DCAELikeHFEncoder",
    "build_hf_encoder",
    "normalize_hf_encoder_type",
]
