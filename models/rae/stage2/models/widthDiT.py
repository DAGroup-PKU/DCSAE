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
from .lightningDiT import LightningDiTBlock, LightningFinalLayer

class widthDiT(nn.Module):
    """
    A multi-width LightningDiT head that supports variable hidden sizes
    (e.g., [1280, 768, 1280]) and per-stage depths and head numbers.
    """
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 1280,
        hidden_size: List[int] = [1280, 768, 1280],
        depth: List[int] = [8, 12, 8],
        num_heads: List[int] = [20, 12, 20],
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

        # === Embedders ===
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size[0], bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size[0])
        self.y_embedder = LabelEmbedder(num_classes, hidden_size[0], class_dropout_prob)

        # === Positional embeddings ===
        num_patches = self.x_embedder.num_patches
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size[0]), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # === ROPE ===
        if use_rope:
            half_head_dim = hidden_size[0] // num_heads[0] // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        # === build transformer stages ===
        self.blocks = nn.ModuleList()
        for i, (dim, n_layers, heads) in enumerate(zip(hidden_size, depth, num_heads)):
            for _ in range(n_layers):
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
            [nn.Linear(self.hidden_size[i-1], self.hidden_size[i]) for i in range(1, len(self.hidden_size))]
        )

        # === Final layer ===
        self.final_layer = LightningFinalLayer(hidden_size[-1], patch_size, in_channels, use_rmsnorm=use_rmsnorm)

        # === Init weights ===
        self.initialize_weights()

    # -------------------------------------------------
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # patch_embed init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        # label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        # timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # final layer zero init
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # -------------------------------------------------
    def unpatchify(self, x):
        c = self.in_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        return imgs

    # -------------------------------------------------
    def forward(self, x, t, y):
        x = self.x_embedder(x)
        if self.use_pos_embed:
            x = x + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = F.silu(t + y)
        stage_idx = 0
        for blk in self.blocks:
            if isinstance(blk, nn.Linear):
                x = blk(x)
                stage_idx += 1
                c = self.c_projectors[stage_idx](c)
            else:
                x = blk(x, c, feat_rope=self.feat_rope)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x