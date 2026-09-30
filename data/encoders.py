"""Abstract LatentEncoder interface and concrete implementations.

Provides a unified encode API for different VAE/SAE models, used by both
the feature extraction script (extract_features_wds.py) and the online
encoding path in ImageNetWDS.
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class LatentEncoder(ABC):
    """Abstract base class for all VAE/SAE encoders.

    All implementations must produce float32 latent tensors of shape
    ``[B, C, h, w]`` from input images ``[B, 3, H, W]`` in ``[-1, 1]``.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to latent space.

        Args:
            x: ``[B, 3, H, W]`` images normalised to ``[-1, 1]``.

        Returns:
            ``[B, C, h, w]`` float32 latent tensor.
        """

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """Total number of latent channels ``C``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used as the ``feat_*`` subdirectory name."""

    @property
    def spatial_downsample(self) -> int:
        """``latent_H = image_H / spatial_downsample``.  Default ``32``."""
        return 32

    # convenience -------------------------------------------------------
    def to(self, device: torch.device | str) -> "LatentEncoder":
        """Move internal model(s) to *device* (no-op by default)."""
        return self


# -----------------------------------------------------------------------
#  DCSAE encoder
# -----------------------------------------------------------------------

class DCSAEEncoder(LatentEncoder):
    """Wraps the full dc-sae encode pipeline.

    Reuses ``build_dc_sae`` / ``load_sae_checkpoint`` / ``encode_batch``
    logic from ``scripts/generate_dc_sae_latent.py``.

    Args:
        sae_config_path: YAML config for the dc-sae model.
        sae_ckpt_path:   Checkpoint ``.pth`` file.
        latent_stats_path: Optional ``.pt`` file for per-channel normalisation.
        image_size:      Input resolution (only affects pos-embed compat).
        device:          CUDA device string or ``torch.device``.
        precision:       ``"bf16"`` or ``"fp32"``.
        encoder_name:    Override for ``self.name``; auto-derived if ``None``.
    """

    def __init__(
        self,
        sae_config_path: str,
        sae_ckpt_path: str,
        latent_stats_path: Optional[str] = None,
        image_size: int = 512,
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        encoder_name: Optional[str] = None,
    ):
        self._device = torch.device(device)
        self._precision = precision
        self._image_size = image_size

        # --- import project helpers (they rely on sys.path) ---
        _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        _sae_dir = os.path.join(_repo, "dc-sae")
        if _sae_dir not in sys.path:
            sys.path.insert(0, _sae_dir)
        if _repo not in sys.path:
            sys.path.insert(0, _repo)

        from config_utils import load_config
        from model import DCSAE
        from models.rae.utils.vae_utils import LatentNormalizer, load_latent_stats

        # build model
        sae_cfg = load_config(sae_config_path)
        self._sae_model: DCSAE = self._build(sae_cfg)
        self._load_checkpoint(sae_ckpt_path)

        # optional per-channel normalisation
        self._normalizer: Optional[LatentNormalizer] = None
        if latent_stats_path:
            stats = load_latent_stats(latent_stats_path, device=self._device, verbose=False)
            self._normalizer = LatentNormalizer(stats, per_channel=True)

        # derive name
        if encoder_name is not None:
            self._name = encoder_name
        else:
            enc_type = sae_cfg.encoder.type
            hf_dim = sae_cfg.model.hf_dim
            self._name = f"dc_sae_{enc_type}_hf{hf_dim}"

        # cache channels
        self._latent_channels = self._sae_model.semantic_channels + self._sae_model.hf_dim

    # -- internal helpers -----------------------------------------------

    def _build(self, sae_cfg):
        from train_sae import build_model
        model = build_model(sae_cfg, self._device)
        model.eval()
        model.requires_grad_(False)
        return model

    def _load_checkpoint(self, ckpt_path):
        from checkpoint_utils import load_sae_checkpoint
        load_sae_checkpoint(self._sae_model, ckpt_path)

    # -- public API ------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._sae_model.eval()
        amp_dtype = torch.bfloat16 if self._precision == "bf16" else torch.float32
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self._precision == "bf16"):
            enc = self._sae_model._infer_latent(x)
            z = enc.latent
            enc_cond = self._sae_model.encode(z, x_img=x, force_drop_hf=False)
        B, N, C = enc_cond.shape
        H = W = int(math.sqrt(N))
        latent = enc_cond.float().transpose(1, 2).contiguous().view(B, C, H, W)
        if self._normalizer is not None:
            latent = self._normalizer.normalize(latent)
        return latent

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def name(self) -> str:
        return self._name

    def to(self, device):
        self._device = torch.device(device)
        self._sae_model = self._sae_model.to(self._device)
        return self


# -----------------------------------------------------------------------
#  DC-AE encoder
# -----------------------------------------------------------------------

def build_encoder(encoder_type: str, **kwargs) -> LatentEncoder:
    if encoder_type != 'dc_sae':
        raise ValueError('This release includes only the dc_sae WDS encoder')
    return DCSAEEncoder(**kwargs)
