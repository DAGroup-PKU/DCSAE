"""CNN-based decoder: progressive upsampling with ResNet blocks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.nonlinearity = nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(self.nonlinearity(self.norm2(h))))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Upsample2D(nn.Module):
    """A 2D upsampling layer with convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels, out_channels=out_channels, dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsampler = Upsample2D(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.upsampler(hidden_states)
        return hidden_states


class FinalBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


class Decoder2D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder
        that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3): The number of input channels.
        out_channels (`int`, *optional*, defaults to 3): The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
                            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to switch on gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            prev_output_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)

        self.final_block = FinalBlock2D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, sample, use_reentrant=False,
                )
            sample = torch.utils.checkpoint.checkpoint(
                self.final_block, sample, use_reentrant=False,
            )
        else:
            for up_block in self.up_blocks:
                sample = up_block(sample)
            sample = self.final_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
