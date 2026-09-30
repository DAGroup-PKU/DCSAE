"""
Shared components: output types, LoRA modules, loss functions, and utilities.
"""

import contextlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from typing import NamedTuple


# =========================
# Outputs
# =========================

class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]

class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]

class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor


# =========================
# LoRA Modules
# =========================

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 32, alpha: int = 32, dropout: float = 0.0, enabled: bool = True):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = enabled

        self.lora_down = nn.Linear(base.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.enabled and self.r > 0:
            out = out + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return out


class LoRAConv2d(nn.Module):
    """
    LoRA for Conv2d, best for patch-embed conv:
      down: Conv2d(in->r, kernel=k, stride=s)
      up:   Conv2d(r->out, kernel=1)
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        return getattr(self.base, name)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def __init__(self, base: nn.Conv2d, r: int = 32, alpha: int = 32, dropout: float = 0.0, enabled: bool = True):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.enabled = enabled

        self.lora_down = nn.Conv2d(
            base.in_channels, r,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.lora_up = nn.Conv2d(r, base.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.enabled and self.r > 0:
            out = out + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return out


def _set_lora_enabled(module: nn.Module, enabled: bool):
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.enabled = enabled


@contextlib.contextmanager
def lora_disabled(module: nn.Module):
    prev = []
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            prev.append((m, m.enabled))
            m.enabled = False
    try:
        yield
    finally:
        for m, e in prev:
            m.enabled = e


# =========================
# Feature alignment losses (VF loss)
# =========================

def vf_marginal_cos_loss(z_map: torch.Tensor, f_map: torch.Tensor, m1: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """
    z_map,f_map: [B,C,H,W]
    Lmcos = mean ReLU(1 - m1 - cos(z,f))
    """
    z = F.normalize(z_map, dim=1, eps=eps)
    f = F.normalize(f_map, dim=1, eps=eps)
    cos = (z * f).sum(dim=1)  # [B,H,W]
    return F.relu(1.0 - m1 - cos).mean()


def vf_mdms_loss(
    z_map: torch.Tensor,
    f_map: torch.Tensor,
    m2: float = 0.1,
    max_tokens: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pair-wise distance-matrix similarity with random token sampling.
    z_map,f_map: [B,C,H,W]
    """
    B, C, H, W = z_map.shape
    N = H * W
    K = min(N, max_tokens)
    if K <= 1:
        return torch.zeros((), device=z_map.device, dtype=z_map.dtype)

    z = z_map.flatten(2).transpose(1, 2).contiguous()
    f = f_map.flatten(2).transpose(1, 2).contiguous()

    idx = torch.randperm(N, device=z.device)[:K]
    z = z[:, idx, :]
    f = f[:, idx, :]

    z = F.normalize(z, dim=-1, eps=eps)
    f = F.normalize(f, dim=-1, eps=eps)

    sim_z = torch.bmm(z, z.transpose(1, 2))
    sim_f = torch.bmm(f, f.transpose(1, 2))

    diff = (sim_z - sim_f).abs()
    return F.relu(diff - m2).mean()


# =========================
# Utilities
# =========================

def mask_channels(tensor, mask_ratio=0.1, channel_dim=1):
    output = tensor.clone()
    num_channels = tensor.shape[channel_dim]
    num_masked = int(num_channels * mask_ratio)
    if num_masked == 0:
        return output, None
    mask_indices = torch.randperm(num_channels, device=tensor.device)[:num_masked]
    idx = [slice(None)] * tensor.ndim
    idx[channel_dim] = mask_indices
    output[idx] = 0
    return output, mask_indices


def _grad_norm(loss, params, eps=1e-12):
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    norms = []
    for g in grads:
        if g is not None:
            norms.append(g.detach().norm())
    if len(norms) == 0:
        return torch.tensor(0.0, device=loss.device)
    return torch.norm(torch.stack(norms))
