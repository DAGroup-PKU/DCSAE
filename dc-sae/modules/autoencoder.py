"""AutoencoderKL: legacy combined encoder + decoder model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from .common import (
    CausalAutoencoderOutput,
    CausalEncoderOutput,
    CausalDecoderOutput,
    _grad_norm,
    mask_channels,
    vf_marginal_cos_loss,
    vf_mdms_loss,
)
from .encoders import Encoder2D, SigLIP2Encoder2D, DINOv2Encoder2D, Qwen3ViTEncoder2D
from .decoders import Decoder2D, ViTXLDecoder


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        # Encoder config
        encoder_type: str = "dinov3",
        dinov3_model_dir: str = "",
        siglip2_model_name: str = "google/siglip2-base-patch16-256",
        dinov2_model_name: str = "facebook/dinov2-with-registers-base",
        qwen3_vit_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",

        image_size: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,

        latent_channels: int = 1280,
        target_latent_channels: Optional[int] = None,

        spatial_downsample_factor: int = 16,

        # Decoder config
        decoder_type: str = "cnn_decoder",
        dec_block_out_channels: Tuple[int, ...] = (1280, 1024, 512, 256, 128),
        dec_layers_per_block: int = 2,
        decoder_dropout: float = 0.0,
        gradient_checkpointing: bool = False,

        # ViT decoder config
        vit_decoder_hidden_size: int = 1024,
        vit_decoder_num_layers: int = 24,
        vit_decoder_num_heads: int = 16,
        vit_decoder_intermediate_size: int = 4096,

        # VAE behavior
        variational: bool = True,
        kl_weight: float = 1e-6,

        # regularization
        noise_tau: float = 0.0,
        random_masking_channel_ratio: float = 0.0,

        # LoRA
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,

        # VF loss hyper
        vf_margin_cos: float = 0.1,
        vf_margin_dms: float = 0.3,
        vf_max_tokens: int = 32,
        vf_hyper: float = 1.0,
        vf_use_adaptive_weight: bool = True,
        vf_weight_clamp: float = 1e4,
        training_mode: str = "enc_dec",
        denormalize_decoder_output: bool = True,

        skip_to_moments: bool = False,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.variational = variational
        self.kl_weight = kl_weight
        self.noise_tau = noise_tau
        self.random_masking_channel_ratio = random_masking_channel_ratio

        self.vf_margin_cos = vf_margin_cos
        self.vf_margin_dms = vf_margin_dms
        self.vf_max_tokens = vf_max_tokens
        self.vf_hyper = vf_hyper
        self.vf_use_adaptive_weight = vf_use_adaptive_weight
        self.vf_weight_clamp = vf_weight_clamp

        self.original_latent_channels = latent_channels
        self.target_latent_channels = target_latent_channels if target_latent_channels is not None else latent_channels
        self.use_channel_downsample = (self.target_latent_channels != latent_channels)
        self.denormalize_decoder_output = denormalize_decoder_output

        # Build encoder
        encoder_patch_size = None
        decoder_patch_size = patch_size

        if encoder_type == "dinov3" or encoder_type == "dinov3_vitl":
            self.encoder = Encoder2D(
                dinov3_model_dir=dinov3_model_dir,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, enable_lora=enable_lora,
            )
            latent_channels = self.encoder.hidden_size
            encoder_patch_size = self.encoder.patch_size
        elif encoder_type == "siglip2":
            self.encoder = SigLIP2Encoder2D(
                model_name=siglip2_model_name,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, enable_lora=enable_lora,
            )
            latent_channels = self.encoder.hidden_size
            encoder_patch_size = self.encoder.patch_size
        elif encoder_type == "dinov2":
            self.encoder = DINOv2Encoder2D(
                model_name=dinov2_model_name,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, enable_lora=enable_lora,
                normalize=True,
            )
            latent_channels = self.encoder.hidden_size
            encoder_patch_size = self.encoder.patch_size
            decoder_patch_size = patch_size
        elif encoder_type == "qwen3_vit":
            self.encoder = Qwen3ViTEncoder2D(
                model_name=qwen3_vit_model_name,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, enable_lora=enable_lora,
            )
            latent_channels = self.encoder.hidden_size
            encoder_patch_size = self.encoder.patch_size
            decoder_patch_size = patch_size
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Supported: "
                f"'dinov3', 'dinov3_vitl', 'siglip2', 'dinov2', 'qwen3_vit'"
            )

        self.original_latent_channels = latent_channels
        if self.target_latent_channels is None:
            self.target_latent_channels = latent_channels
        self.use_channel_downsample = (self.target_latent_channels != latent_channels)

        base_downsample_factor = patch_size
        assert spatial_downsample_factor % base_downsample_factor == 0, \
            f"spatial_downsample_factor must be {base_downsample_factor} * 2^k for {encoder_type}"
        extra_factor = spatial_downsample_factor // base_downsample_factor
        assert extra_factor & (extra_factor - 1) == 0, f"only allow {base_downsample_factor} * 2^k"
        extra_steps = int(math.log2(extra_factor)) if extra_factor > 1 else 0

        self.latent_downsample_layers = nn.ModuleList([
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(extra_steps)
        ])

        self.channel_downsample_conv = None
        effective_c = latent_channels
        if self.use_channel_downsample:
            self.channel_downsample_conv = nn.Conv2d(latent_channels, self.target_latent_channels, kernel_size=1, stride=1, padding=0)
            effective_c = self.target_latent_channels

        self.skip_to_moments = skip_to_moments
        if skip_to_moments:
            print("skip_to_moments is True")
            self.to_moments = None
        else:
            print("skip_to_moments is False")
            self.to_moments = nn.Conv2d(effective_c, 2 * effective_c, kernel_size=1, stride=1, padding=0)

        # Build decoder
        if decoder_type == "cnn_decoder":
            print("CNN decoder is used")
            self.decoder = Decoder2D(
                in_channels=effective_c,
                out_channels=out_channels,
                block_out_channels=dec_block_out_channels,
                layers_per_block=dec_layers_per_block,
                gradient_checkpointing=gradient_checkpointing,
            )
        elif decoder_type == "vit_decoder":
            print("ViT decoder is used")
            self.decoder = ViTXLDecoder(
                encoder_hidden_size=effective_c,
                decoder_hidden_size=vit_decoder_hidden_size,
                decoder_num_layers=vit_decoder_num_layers,
                decoder_num_heads=vit_decoder_num_heads,
                decoder_intermediate_size=vit_decoder_intermediate_size,
                image_size=image_size,
                patch_size=decoder_patch_size,
                out_channels=out_channels,
                dropout=decoder_dropout,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Supported: 'cnn_decoder', 'vit_decoder'")

    def _noising(self, tensor: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (tensor.shape[0],) + (1,) * (tensor.dim() - 1),
            device=tensor.device, dtype=tensor.dtype,
        )
        return tensor + noise_sigma * torch.randn_like(tensor)

    def _extract_patch_map(self, pred) -> torch.Tensor:
        if self.encoder_type == "dinov3" or self.encoder_type == "dinov3_vitl":
            cls_len = 1
            reg_len = self.encoder.num_register_tokens
            tokens = pred.last_hidden_state[:, cls_len + reg_len:, :]
        elif self.encoder_type == "siglip2":
            tokens = pred.last_hidden_state
        elif self.encoder_type == "dinov2":
            cls_len = 1
            reg_len = self.encoder.num_register_tokens
            tokens = pred.last_hidden_state[:, cls_len + reg_len:, :]
        elif self.encoder_type == "qwen3_vit":
            tokens = pred.last_hidden_state
            if getattr(self.encoder, "has_cls_token", True) and tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        B, N, C = tokens.shape
        S = int(math.sqrt(N))
        assert S * S == N, f"patch token count N={N} is not square; check image size/patch size."
        feat = tokens.transpose(1, 2).contiguous().view(B, C, S, S)
        return feat

    def get_vf_ref_param(self):
        backbone = self.encoder.get_backbone()
        for n, p in backbone.named_parameters():
            if ("lora_up" in n) or ("lora_down" in n):
                if p.requires_grad:
                    return p
        if self.encoder_type == "dinov3" or self.encoder_type == "dinov3_vitl":
            return backbone.embeddings.patch_embeddings.weight
        elif self.encoder_type == "siglip2":
            return backbone.embeddings.patch_embedding.weight
        elif self.encoder_type == "dinov2":
            return backbone.embeddings.patch_embeddings.projection.weight
        elif self.encoder_type == "qwen3_vit":
            if hasattr(backbone, "embeddings") and hasattr(backbone.embeddings, "patch_embeddings"):
                pe = backbone.embeddings.patch_embeddings
                if hasattr(pe, "projection"):
                    return pe.projection.weight
                if hasattr(pe, "weight"):
                    return pe.weight
            return None
        return None

    def encode_features(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        if use_lora:
            pred = self.encoder(x)
        else:
            with torch.no_grad():
                with self.encoder.lora_disabled():
                    pred = self.encoder(x)
        feat = self._extract_patch_map(pred)
        return feat

    def compute_vf_loss(self, z_feat: torch.Tensor, f_feat: torch.Tensor) -> torch.Tensor:
        lmcos = vf_marginal_cos_loss(z_feat, f_feat, m1=self.vf_margin_cos)
        lmdms = vf_mdms_loss(z_feat, f_feat, m2=self.vf_margin_dms, max_tokens=self.vf_max_tokens)
        return lmcos + lmdms

    def adaptive_weight(self, loss_rec, loss_vf, params, eps=1e-6):
        n_rec = _grad_norm(loss_rec, params)
        n_vf = _grad_norm(loss_vf, params)
        if (n_rec == 0) or (n_vf == 0):
            return torch.tensor(1.0, device=loss_rec.device, dtype=loss_rec.dtype)
        w = (n_rec / (n_vf + eps)).clamp(0.0, self.vf_weight_clamp).detach()
        return w

    def encode(self, x: torch.Tensor, sample_posterior: bool | None = None) -> CausalEncoderOutput:
        feat = self.encode_features(x, use_lora=True)
        for conv in self.latent_downsample_layers:
            feat = conv(feat)

        if self.channel_downsample_conv is not None:
            feat = self.channel_downsample_conv(feat)

        if self.to_moments is None:
            if self.random_masking_channel_ratio > 0.0:
                feat, _ = mask_channels(feat, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)
            if self.training and self.noise_tau > 0:
                feat = self._noising(feat)
            return CausalEncoderOutput(latent=feat, posterior=None)

        moments = self.to_moments(feat)
        posterior = DiagonalGaussianDistribution(moments, deterministic=False)

        if sample_posterior is None:
            sample_posterior = self.training

        z = posterior.sample() if sample_posterior else posterior.mode()

        if self.random_masking_channel_ratio > 0.0:
            z, _ = mask_channels(z, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)
        if self.training and self.noise_tau > 0:
            z = self._noising(z)

        return CausalEncoderOutput(latent=z, posterior=posterior)

    def _denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.denormalize_decoder_output:
            return tensor
        device = tensor.device
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
        return (tensor * imagenet_std + imagenet_mean)

    def decode(self, z: torch.Tensor) -> CausalDecoderOutput:
        x = self.decoder(z)
        x = self._denormalize_output(x)
        return CausalDecoderOutput(x)

    def forward(self, x: torch.Tensor) -> CausalAutoencoderOutput:
        enc = self.encode(x)
        dec = self.decode(enc.latent)
        return CausalAutoencoderOutput(dec.sample, enc.latent, enc.posterior)
