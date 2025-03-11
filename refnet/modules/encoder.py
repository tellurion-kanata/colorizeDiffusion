import torch.nn as nn
import torch.nn.functional as F

from refnet.modules.unet import (
    TimestepEmbedSequential,
    conv_nd,
    zero_module,
    ResBlock,
    Downsample
)
from torch.utils.checkpoint import checkpoint


def make_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels or in_channels
    return TimestepEmbedSequential(zero_module(conv_nd(2, in_channels, out_channels, 1, padding=0)))

def activate_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels or in_channels
    return TimestepEmbedSequential(
        nn.SiLU(),
        zero_module(conv_nd(2, in_channels, out_channels, 1, padding=0))
    )

def sequential_downsample(in_channels, out_channels):
    return nn.Sequential(
        conv_nd(2, in_channels, 16, 3, padding=1),
        nn.SiLU(),
        conv_nd(2, 16, 16, 3, padding=1),
        nn.SiLU(),
        conv_nd(2, 16, 32, 3, padding=1, stride=2),
        nn.SiLU(),
        conv_nd(2, 32, 32, 3, padding=1),
        nn.SiLU(),
        conv_nd(2, 32, 96, 3, padding=1, stride=2),
        nn.SiLU(),
        conv_nd(2, 96, 96, 3, padding=1),
        nn.SiLU(),
        conv_nd(2, 96, 256, 3, padding=1, stride=2),
        nn.SiLU(),
        zero_module(conv_nd(2, 256, out_channels, 3, padding=1))
    )


class SimpleEncoder(nn.Module):
    def __init__(self, c_channels, model_channels):
        super().__init__()
        self.model = sequential_downsample(c_channels, model_channels)

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class MultiEncoder(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults, checkpoint=True):
        super().__init__()
        output_chs = [model_channels * mult for mult in ch_mults]
        self.model = TimestepEmbedSequential(
            conv_nd(2, in_ch, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 256, output_chs[0], 3, padding=1)
        )
        self.zero_layer = make_zero_conv(output_chs[0])
        self.output_blocks = nn.ModuleList()
        self.zero_blocks = nn.ModuleList()

        block_num = len(ch_mults)
        prev_ch = output_chs[0]
        for i in range(block_num):
            self.output_blocks.append(TimestepEmbedSequential(
                nn.SiLU(),
                conv_nd(2, prev_ch, output_chs[i], 3, padding=1, stride=2 if i != block_num-1 else 1),
                nn.SiLU(),
                conv_nd(2, output_chs[i], output_chs[i], 3, padding=1)
            ))
            self.zero_blocks.append(make_zero_conv(output_chs[i]))
            prev_ch = output_chs[i]

        self.checkpoint = checkpoint

    def forward(self, x, emb=None, context=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, emb, context, use_reentrant=False)
        return self._forward(x, emb, context)

    def _forward(self, x, emb=None, context=None):
        x = self.model(x, emb, context)
        hints = [self.zero_layer(x, emb, context)]
        for layer, zero_layer in zip(self.output_blocks, self.zero_blocks):
            x = layer(x, emb, context)
            hints.append(zero_layer(x, emb, context))
        return hints


class TimestepEmbedEncoder(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults, dropout=0.):
        super().__init__()
        output_chs = [model_channels * mult for mult in ch_mults]
        time_embed_dim = model_channels * 4

        self.model = TimestepEmbedSequential(
            conv_nd(2, in_ch, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 256, output_chs[0], 3, padding=1)
        )
        self.zero_layer = make_zero_conv(output_chs[0])
        self.output_blocks = nn.ModuleList()
        self.zero_blocks = nn.ModuleList()

        block_num = len(ch_mults)
        prev_ch = output_chs[0]
        for i in range(block_num):
            self.output_blocks.append(TimestepEmbedSequential(
                nn.SiLU(),
                Downsample(prev_ch, True, out_channels=output_chs[i]),
                nn.SiLU(),
                ResBlock(output_chs[i], time_embed_dim, dropout)
            ))
            self.zero_blocks.append(make_zero_conv(output_chs[i]))
            prev_ch = output_chs[i]

    def forward(self, x, emb=None, context=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, emb, context, use_reentrant=False)
        return self._forward(x, emb, context)

    def _forward(self, x, emb=None, context=None):
        x = self.model(x, emb, context)
        hints = [self.zero_layer(x, emb, context)]
        for layer, zero_layer in zip(self.output_blocks, self.zero_blocks):
            x = layer(x, emb, context)
            hints.append(zero_layer(x, emb, context))
        return hints


class Downsampler(nn.Module):
    """
        Prepare for scribble generation
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")