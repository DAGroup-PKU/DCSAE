"""
VAE utility functions for loading and using DINOv3/SigLIP2/DINOv2 VAE models.
Supports both CNN decoder and diffusion decoder.

包含:
- Latent 归一化/反归一化
- VAE 模型加载
- 从 latent 重建图像
- Encoder 配置获取
- 从命令行参数构建 VAE

统一了 train_vae, eval_vae, projects/rae 中的重复代码
"""

import os
import argparse
from typing import Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn

# ==========================================
#           SAE Normalization (DINOv3)
# ==========================================

EMA_SHIFT_FACTOR = 0.0019670347683131695
EMA_SCALE_FACTOR = 0.247765451669693


def normalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using SAE statistics (DINOv3)."""
    return (tensor - EMA_SHIFT_FACTOR) / EMA_SCALE_FACTOR


def denormalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using SAE statistics (DINOv3)."""
    return tensor * EMA_SCALE_FACTOR + EMA_SHIFT_FACTOR


# ==========================================
#           SigLIP2 Normalization
# ==========================================

SIGLIP2_SHIFT_FACTOR = 0.0
SIGLIP2_SCALE_FACTOR = 0.6689115762710571


def normalize_siglip2(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using SigLIP2 statistics."""
    return (tensor - SIGLIP2_SHIFT_FACTOR) / SIGLIP2_SCALE_FACTOR


def denormalize_siglip2(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using SigLIP2 statistics."""
    return tensor * SIGLIP2_SCALE_FACTOR + SIGLIP2_SHIFT_FACTOR


# ==========================================
#           DINOv2 Normalization
# ==========================================

# DINOv2-B (base) normalization statistics
# These are placeholder values - you may need to compute proper statistics
# from your training data similar to how DINOv3/SigLIP2 stats were computed.
DINOV2_SHIFT_FACTOR = 0.0
DINOV2_SCALE_FACTOR = 1.0  # Placeholder - adjust based on actual feature distribution


def normalize_dinov2(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using DINOv2 statistics."""
    return (tensor - DINOV2_SHIFT_FACTOR) / DINOV2_SCALE_FACTOR


def denormalize_dinov2(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using DINOv2 statistics."""
    return tensor * DINOV2_SCALE_FACTOR + DINOV2_SHIFT_FACTOR


# ==========================================
#       Generic Normalization Helpers
# ==========================================

def get_normalize_fn(encoder_type: str = "dinov3"):
    """Get normalization function based on encoder type."""
    if encoder_type == "dinov3":
        return normalize_sae
    elif encoder_type == "siglip2":
        return normalize_siglip2
    elif encoder_type == "dinov2":
        return normalize_dinov2
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'siglip2', 'dinov2'")


def get_denormalize_fn(encoder_type: str = "dinov3"):
    """Get denormalization function based on encoder type."""
    if encoder_type == "dinov3":
        return denormalize_sae
    elif encoder_type == "siglip2":
        return denormalize_siglip2
    elif encoder_type == "dinov2":
        return denormalize_dinov2
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'siglip2', 'dinov2'")


# ==========================================
#     Latent Stats Loading & Normalization
# ==========================================

def load_latent_stats(stats_path: str, device: torch.device = None, verbose: bool = True):
    """
    Load pre-computed latent statistics from .npz or .pt file.
    
    Args:
        stats_path: Path to latent_stats.npz / latent_stats.pt file (from compute_latent_stats.py)
        device: Device to load tensors to (default: CPU)
        verbose: Whether to print loading information
    
    Returns:
        dict with keys: 'mean', 'std', 'shift_factor', 'scale_factor'
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats file not found: {stats_path}")
    
    ext = os.path.splitext(stats_path)[1].lower()
    if ext == ".npz":
        data = np.load(stats_path)
        mean = torch.from_numpy(data["mean"]).float()  # [C]
        std = torch.from_numpy(data["std"]).float()    # [C]
    elif ext in [".pt", ".pth"]:
        data = torch.load(stats_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid latent stats format in {stats_path}: expected dict, got {type(data)}")
        if "mean" not in data or "std" not in data:
            raise KeyError(f"Latent stats file {stats_path} must contain keys 'mean' and 'std'")
        mean = torch.as_tensor(data["mean"]).float()
        std = torch.as_tensor(data["std"]).float()
    else:
        raise ValueError(
            f"Unsupported latent stats format: {stats_path}. "
            "Expected file extension: .npz, .pt, or .pth"
        )
    
    # Compute scalar shift/scale factors (mean of means/stds)
    shift_factor = float(mean.mean().item())
    scale_factor = float(std.mean().item())
    
    if device is not None:
        mean = mean.to(device)
        std = std.to(device)
    
    if verbose:
        print(f"[load_latent_stats] Loaded from: {stats_path}")
        print(f"[load_latent_stats] Channels: {len(mean)}, shift_factor: {shift_factor:.6f}, scale_factor: {scale_factor:.6f}")
    
    return {
        'mean': mean,
        'std': std,
        'shift_factor': shift_factor,
        'scale_factor': scale_factor,
    }


def normalize_with_stats(
    tensor: torch.Tensor, 
    stats: dict, 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Normalize tensor using pre-computed latent statistics.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        stats: Dict from load_latent_stats() containing 'mean', 'std', 'shift_factor', 'scale_factor'
        per_channel: If True, use per-channel mean/std. If False, use scalar shift/scale factors.
    
    Returns:
        Normalized tensor
    """
    if per_channel:
        # Per-channel normalization: (x - mean[c]) / std[c]
        mean = stats['mean'].to(tensor.device)  # [C]
        std = stats['std'].to(tensor.device)    # [C]
        # Reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return (tensor - mean) / (std + 1e-8)
    else:
        # Scalar normalization: (x - shift_factor) / scale_factor
        shift = stats['shift_factor']
        scale = stats['scale_factor']
        return (tensor - shift) / scale


def denormalize_with_stats(
    tensor: torch.Tensor, 
    stats: dict, 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Denormalize tensor using pre-computed latent statistics.
    
    Args:
        tensor: Normalized tensor [B, C, H, W]
        stats: Dict from load_latent_stats() containing 'mean', 'std', 'shift_factor', 'scale_factor'
        per_channel: If True, use per-channel mean/std. If False, use scalar shift/scale factors.
    
    Returns:
        Denormalized tensor
    """
    if per_channel:
        # Per-channel denormalization: x * std[c] + mean[c]
        mean = stats['mean'].to(tensor.device)  # [C]
        std = stats['std'].to(tensor.device)    # [C]
        # Reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return tensor * std + mean
    else:
        # Scalar denormalization: x * scale_factor + shift_factor
        shift = stats['shift_factor']
        scale = stats['scale_factor']
        return tensor * scale + shift


class LatentNormalizer:
    """
    A helper class that wraps normalization/denormalization with loaded stats.
    Can be used as a drop-in replacement for normalize_fn/denormalize_fn.
    """
    def __init__(self, stats: dict, per_channel: bool = False):
        """
        Args:
            stats: Dict from load_latent_stats()
            per_channel: Whether to use per-channel normalization
        """
        self.stats = stats
        self.per_channel = per_channel
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return normalize_with_stats(tensor, self.stats, self.per_channel)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return denormalize_with_stats(tensor, self.stats, self.per_channel)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default call is normalize."""
        return self.normalize(tensor)


# ==========================================
#           VAE Loading
# ==========================================

