"""Qwen3-ViT encoder (from Qwen3-VL checkpoints)."""

import contextlib
from types import SimpleNamespace

import torch
from torch import nn

try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3_VIT = True
except ImportError:
    HAS_QWEN3_VIT = False
    print("Warning: transformers Qwen3VLForConditionalGeneration not found. Qwen3-ViT encoder will not be available.")


# =========================
# Normalization for Qwen3-ViT input
# =========================

class Qwen3ViTScalingLayer(nn.Module):
    """
    Qwen3-ViT input normalization.
    Default pass-through for [-1, 1] range.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([0.0, 0.0, 0.0])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([1.0, 1.0, 1.0])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


# =========================
# Encoder (Qwen3-ViT)
# =========================

class Qwen3ViTEncoder2D(nn.Module):
    """
    Qwen3-ViT encoder (from Qwen3-VL checkpoints).

    Output dim: 4096 (after spatial merger)
    Token count: (H/16/2) * (W/16/2) = 64 (for 256x256 images)

    Only keeps the visual module; LLM part is deleted to save memory (~15GB).
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,
        freeze_encoder: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if not HAS_QWEN3_VIT:
            raise ImportError(
                "Qwen3-ViT requires transformers Qwen3VLForConditionalGeneration. "
                "Please install: pip install transformers"
            )

        self.encoder_type = "qwen3_vit"
        self.model_name = model_name

        print(f"Loading Qwen3-VL from {model_name}...")
        try:
            full_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, AttributeError):
            full_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                local_files_only=False,
            )

        self.visual = full_model.visual
        del full_model.model
        del full_model.lm_head
        del full_model
        torch.cuda.empty_cache()

        if enable_lora and lora_rank > 0:
            print("Warning: LoRA is not wired for Qwen3-ViT yet; running without LoRA adapters.")

        if freeze_encoder:
            for param in self.visual.parameters():
                param.requires_grad = False

        self.normalization_layer = Qwen3ViTScalingLayer()

        vision_cfg = self.visual.config if hasattr(self.visual, 'config') else None

        self.patch_size = getattr(vision_cfg, 'patch_size', 16)
        self.temporal_patch_size = getattr(vision_cfg, 'temporal_patch_size', 2)
        self.spatial_merge_size = getattr(vision_cfg, 'spatial_merge_size', 2)
        self.hidden_size = getattr(vision_cfg, 'out_hidden_size', 4096)
        self.vit_hidden_size = getattr(vision_cfg, 'hidden_size', 1152)

        self.num_register_tokens = 0
        self.has_cls_token = False

        print(f"Qwen3-ViT Encoder initialized:")
        print(f"  - patch_size: {self.patch_size}")
        print(f"  - spatial_merge_size: {self.spatial_merge_size}")
        print(f"  - hidden_size (output): {self.hidden_size}")

    def _preprocess_images(self, images: torch.Tensor) -> tuple:
        """
        Convert [B, C, H, W] image tensor to Qwen3-VL format.

        Returns:
            pixel_values: [total_patches, C * temporal_patch_size * patch_size * patch_size]
            image_grid_thw: [B, 3] - (temporal, h_patches, w_patches)
        """
        B, C, H, W = images.shape
        device = images.device

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        patches = images.view(B, C, h_patches, self.patch_size, w_patches, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B * num_patches, C, self.patch_size, self.patch_size)

        patches = patches.unsqueeze(2).repeat(1, 1, self.temporal_patch_size, 1, 1)

        pixel_values = patches.view(B * num_patches, -1)

        image_grid_thw = torch.tensor(
            [[1, h_patches, w_patches]] * B,
            device=device,
            dtype=torch.long
        )

        return pixel_values, image_grid_thw

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
            where N = (H/16/2) * (W/16/2) = 64 (for 256x256 images)
        """
        B, C, H, W = x.shape
        device = x.device

        x = self.normalization_layer(x)

        pixel_values, image_grid_thw = self._preprocess_images(x)
        pixel_values = pixel_values.to(dtype=self.visual.dtype, device=device)

        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.visual.parameters())):
            image_embeds, _ = self.visual(pixel_values, grid_thw=image_grid_thw)

        tokens_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2] // (self.spatial_merge_size ** 2)).tolist()

        image_embeds_list = torch.split(image_embeds, tokens_per_image)

        if len(set(tokens_per_image)) == 1:
            embeddings = torch.stack(image_embeds_list, dim=0)
        else:
            max_tokens = max(tokens_per_image)
            embeddings = torch.zeros(B, max_tokens, self.hidden_size, device=device, dtype=image_embeds.dtype)
            for i, emb in enumerate(image_embeds_list):
                embeddings[i, :emb.shape[0], :] = emb

        return SimpleNamespace(last_hidden_state=embeddings)

    def get_backbone(self):
        return self.visual


# =========================
# Qwen3-ViT with AvgPool merge (no learned merger)
# =========================

class _AvgPoolMerger(nn.Module):
    """Replace Qwen3 learned PatchMerger with simple 2×2 spatial avg pool.

    Input:  flat sequence of pre-norm vit tokens [total_tokens, vit_hidden_size]
            (same interface as Qwen3VLVisionPatchMerger)
    Output: [total_tokens / spatial_merge_size^2, vit_hidden_size]
    """

    def __init__(self, vit_hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.vit_hidden_size = vit_hidden_size
        self.spatial_merge_size = spatial_merge_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [total_tokens, vit_hidden_size]  (flat across batch)
        # We can't reshape to spatial here because tokens are flat across images.
        # Use unfold-style avg: group every spatial_merge_size^2 consecutive tokens.
        k = self.spatial_merge_size ** 2  # 4
        N = x.shape[0]
        assert N % k == 0, f"Token count {N} not divisible by merge factor {k}"
        x = x.view(N // k, k, self.vit_hidden_size)
        return x.mean(dim=1)  # [N/4, vit_hidden_size]


class Qwen3ViTAvgPoolEncoder2D(Qwen3ViTEncoder2D):
    """Qwen3-ViT encoder with avg-pool spatial merge instead of learned merger.

    Replaces Qwen3's PatchMerger (concat 2×2 + MLP → 4096) with simple
    2×2 avg pool, keeping the original vit_hidden_size (1152).

    Output: [B, 64, 1152]  (for 256×256, same token count as original but dim=1152)
    Designed to be used with DeMerger for spatial expansion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Replace the learned merger with avg pool
        self.visual.merger = _AvgPoolMerger(
            vit_hidden_size=self.vit_hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )
        # Output dim is vit_hidden_size (1152), not out_hidden_size (4096)
        self.hidden_size = self.vit_hidden_size
        self.encoder_type = "qwen3_vit_avg_pool"

        print(f"  - AvgPool merger: spatial_merge_size={self.spatial_merge_size}")
        print(f"  - hidden_size (output): {self.hidden_size} (vit_hidden_size, no projection)")


# =========================
# Qwen3-ViT with learnable Transformer + channel-shuffle merger
# =========================

class _TransformerShuffleMerger(nn.Module):
    """Learnable replacement for Qwen3's PatchMerger.

    Flow on flat pre-merge tokens [total_tokens, vit_hidden_size]:
      1. Two Transformer encoder layers at vit_hidden_size.
      2. Group every spatial_merge_size^2 consecutive tokens.
      3. Channel-shuffle/interleave then concat -> [N/k, k * vit_hidden_size].
      4. MLP (LayerNorm -> Linear -> GELU -> Linear) -> vit_hidden_size.
    """

    def __init__(
        self,
        vit_hidden_size: int,
        spatial_merge_size: int = 2,
        num_transformer_layers: int = 2,
        nhead: int = 8,
    ):
        super().__init__()
        self.vit_hidden_size = vit_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.num_sub_tokens = spatial_merge_size ** 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_hidden_size,
            nhead=nhead,
            dim_feedforward=vit_hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        merged_dim = vit_hidden_size * self.num_sub_tokens
        self.norm = nn.LayerNorm(merged_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_dim, vit_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.num_sub_tokens
        N, C = x.shape
        assert N % k == 0, f"Token count {N} not divisible by merge factor {k}"

        h = self.transformer(x.unsqueeze(0)).squeeze(0)

        grouped = h.view(N // k, k, C)
        shuffled = grouped.transpose(1, 2).contiguous().view(N // k, k * C)

        out = self.linear_fc2(self.act_fn(self.linear_fc1(self.norm(shuffled))))
        return out


class Qwen3ViTTransformerMergerEncoder2D(Qwen3ViTEncoder2D):
    """Qwen3-ViT encoder with learnable Transformer + channel-shuffle merger.

    Replaces Qwen3's PatchMerger with two Transformer layers followed by
    2x2 channel-shuffle concat + MLP projection back to vit_hidden_size.

    Output: [B, 64, 1152] (for 256x256), same token count and dim as the
    avg-pool variant. Designed to remain trainable even when the rest of
    the Qwen3 backbone is frozen.
    """

    def __init__(
        self,
        merger_num_transformer_layers: int = 2,
        merger_nhead: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.visual.merger = _TransformerShuffleMerger(
            vit_hidden_size=self.vit_hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            num_transformer_layers=merger_num_transformer_layers,
            nhead=merger_nhead,
        ).to(dtype=self.visual.dtype)
        self.trainable_merger = self.visual.merger
        self.hidden_size = self.vit_hidden_size
        self.encoder_type = "qwen3_vit_transformer_merger"

        print(f"  - Transformer+Shuffle merger: layers={merger_num_transformer_layers}, "
              f"nhead={merger_nhead}, spatial_merge_size={self.spatial_merge_size}")
        print(f"  - hidden_size (output): {self.hidden_size} (vit_hidden_size, no projection)")
