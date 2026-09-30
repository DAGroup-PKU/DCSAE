from math import sqrt
from re import L
from regex import B
import torch
import torch.nn as nn

from transformers import SwinModel
import torch
from torch import nn
from .lightningDiT import PatchEmbed, Mlp, NormAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed
import torch.nn.functional as F
from typing import *
from .lightningDiT import LightningDiTBlock
from .DDT import DDTFinalLayer


class widthDDT(nn.Module):
    """
    Multi-width Lightning + DDT hybrid head.
    Structure: encoder (multi-width) → projector → decoder (multi-width) → DDTFinalLayer
    Supports e.g. hidden_size = [1280, 768, 1280, 2048]
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: Union[int, list[int]] = 1,
        in_channels: int = 1280,
        hidden_size: List[int] = [1280, 768, 1280, 2048],
        depth: List[int] = [8, 12, 8, 2],
        num_heads: List[int] = [20, 12, 20, 32],
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        assert len(hidden_size) == len(depth)
        assert len(num_heads) == len(hidden_size)

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_pos_embed = use_pos_embed

        # === patch embedders ===
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.s_patch_size, self.x_patch_size = patch_size
        s_ch_per_token = in_channels * self.s_patch_size * self.s_patch_size
        x_ch_per_token = in_channels * self.x_patch_size * self.x_patch_size
        self.num_encoder_blocks = sum(depth[:-1]) + (len(hidden_size) - 2)  # 包含中间Linear
        self.num_blocks = sum(depth) + (len(hidden_size) - 1)

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, s_ch_per_token, hidden_size[0], bias=True)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, x_ch_per_token, hidden_size[-1], bias=True)

        # === Embedders ===
        self.t_embedder = GaussianFourierEmbedding(hidden_size[0])
        self.y_embedder = LabelEmbedder(num_classes, hidden_size[0], class_dropout_prob)

        # === Positional embedding ===
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size[0]), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.x_pos_embed = None

        # === Rotary encoding ===
        if use_rope:
            enc_half_head = hidden_size[0] // num_heads[0] // 2
            dec_half_head = hidden_size[-1] // num_heads[-1] // 2
            seq_len = int(sqrt(self.s_embedder.num_patches))
            self.enc_rope = VisionRotaryEmbeddingFast(dim=enc_half_head, pt_seq_len=seq_len)
            self.dec_rope = VisionRotaryEmbeddingFast(dim=dec_half_head, pt_seq_len=seq_len)
        else:
            self.enc_rope = self.dec_rope = None

        # === build multi-stage blocks ===
        self.blocks = nn.ModuleList()
        for i, (dim, n, heads) in enumerate(zip(hidden_size, depth, num_heads)):
            for _ in range(n):
                self.blocks.append(
                    LightningDiTBlock(
                        hidden_size=dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        use_qknorm=use_qknorm,
                        use_swiglu=use_swiglu,
                        use_rmsnorm=use_rmsnorm,
                        wo_shift=wo_shift,
                    )
                )
            if i < len(hidden_size) - 1:
                self.blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

        self.c_projectors = nn.ModuleList(
            [nn.Identity()] +
            [nn.Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size) - 1)]
        )
        # === encoder→decoder projector ===
        self.s_projector = nn.Linear(hidden_size[-2], hidden_size[-1])

        # === final layer ===
        self.final_layer = DDTFinalLayer(hidden_size[-1], 1, x_ch_per_token, use_rmsnorm=use_rmsnorm)

        self.initialize_weights()

    # -------------------------------------------------
    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # -------------------------------------------------
    def unpatchify(self, x):
        c = self.in_channels * self.x_patch_size * self.x_patch_size
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    # -------------------------------------------------
    def forward(self, x, t, y, s=None, mask=None):
        # ===== 1. prepare condition =====
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = F.silu(t + y)  # condition (initially 1280-dim)

        # ===== 2. encoder (multi-width DiT) =====
        if s is None:
            s = self.s_embedder(x)
            if self.use_pos_embed:
                s = s + self.pos_embed

            stage_idx = 0
            for i in range(self.num_encoder_blocks):
                blk = self.blocks[i]
                if isinstance(blk, nn.Linear):
                    # width transition
                    s = blk(s)
                    stage_idx += 1
                    # project c to next stage width (but skip if beyond encoder)
                    if stage_idx < len(self.c_projectors):
                        c = self.c_projectors[stage_idx](c)
                else:
                    # normal DiT block
                    s = blk(s, c, feat_rope=self.enc_rope)

            # after encoder, broadcast timestep into token space
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = F.silu(t + s)

        # ===== 3. encoder → decoder transition =====
        s = self.s_projector(s)  # e.g., 1280 → 2048

        # ===== 4. decoder (DDT) =====
        x = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed

        decoder_start = self.num_encoder_blocks
        if isinstance(self.blocks[decoder_start], nn.Linear):
            decoder_start += 1
        for i in range(decoder_start, self.num_blocks):
            blk = self.blocks[i]
            if isinstance(blk, nn.Linear):
                x = blk(x)
            else:
                x = blk(x, s, feat_rope=self.dec_rope)

        # ===== 5. final output =====
        x = self.final_layer(x, s)
        x = self.unpatchify(x)
        return x