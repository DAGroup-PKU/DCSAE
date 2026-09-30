# ==========================
# PatchGAN Discriminator (added)
# ==========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator (pix2pix-style).
    Input:  [B,3,H,W] in [-1, 1]
    Output: [B,1,h,w] logits
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=4, norm="in"):
        super().__init__()
        kw = 4
        padw = 1

        def make_norm(c):
            if norm == "in":
                return nn.InstanceNorm2d(c, affine=True)
            elif norm == "bn":
                return nn.BatchNorm2d(c)
            elif norm == "gn":
                # 32 groups is common; fallback to min(32,c)
                g = min(32, c)
                return nn.GroupNorm(g, c)
            else:
                raise ValueError(f"Unknown norm: {norm}")

        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            c_in = ndf * nf_mult_prev
            c_out = ndf * nf_mult
            sequence += [
                nn.Conv2d(c_in, c_out, kernel_size=kw, stride=2, padding=padw, bias=False),
                make_norm(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # one more layer, stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        c_in = ndf * nf_mult_prev
        c_out = ndf * nf_mult
        sequence += [
            nn.Conv2d(c_in, c_out, kernel_size=kw, stride=1, padding=padw, bias=False),
            make_norm(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # output patch logits
        sequence += [nn.Conv2d(c_out, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def d_hinge_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return loss_real + loss_fake


def g_hinge_loss(logits_fake):
    return -torch.mean(logits_fake)
