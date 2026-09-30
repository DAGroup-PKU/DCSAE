"""
dc-sae/modules — Modular encoder/decoder components for dc-sae.

Re-exports all public symbols for convenient access:
    from modules import Encoder2D, ViTXLDecoder, AutoencoderKL, ...
"""

from .common import (
    CausalAutoencoderOutput,
    CausalEncoderOutput,
    CausalDecoderOutput,
    LoRALinear,
    LoRAConv2d,
    _set_lora_enabled,
    lora_disabled,
    vf_marginal_cos_loss,
    vf_mdms_loss,
    mask_channels,
    _grad_norm,
)

from .encoders import (
    Encoder2D,
    ScalingLayer2D,
    add_lora_to_dinov3,
    DINOv2Encoder2D,
    Dinov2DeepCompressionEncoder2D,
    DINOv2ScalingLayer,
    add_lora_to_dinov2,
    SigLIP2Encoder2D,
    SigLIP2ScalingLayer,
    add_lora_to_siglip2,
    Qwen3ViTEncoder2D,
    Qwen3ViTScalingLayer,
    Qwen3ViTAvgPoolEncoder2D,
    Qwen3ViTTransformerMergerEncoder2D,
    InternVL3ViTEncoder2D,
    InternVL3ScalingLayer,
    HFResBlock,
    HFEncoder,
    DCAELikeHFEncoder,
    build_hf_encoder,
    normalize_hf_encoder_type,
)

from .demerger import SpatialDeMerger

from .decoders import (
    Decoder2D,
    ViTXLDecoder,
)

from .autoencoder import AutoencoderKL

__all__ = [
    # Output types
    "CausalAutoencoderOutput",
    "CausalEncoderOutput",
    "CausalDecoderOutput",
    # LoRA
    "LoRALinear",
    "LoRAConv2d",
    "_set_lora_enabled",
    "lora_disabled",
    # Losses / utils
    "vf_marginal_cos_loss",
    "vf_mdms_loss",
    "mask_channels",
    "_grad_norm",
    # Encoders
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
    # Decoders
    "Decoder2D",
    "ViTXLDecoder",
    # DeMerger
    "SpatialDeMerger",
    # Autoencoder
    "AutoencoderKL",
]
