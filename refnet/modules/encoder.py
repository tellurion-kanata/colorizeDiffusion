import torch.nn as nn

from sgm.util import exists
from ldm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential,
    conv_nd,
    zero_module,
)


def make_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels if exists(out_channels) else in_channels
    return TimestepEmbedSequential(zero_module(conv_nd(2, in_channels, out_channels, 1, padding=0)))

def activate_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels if exists(out_channels) else in_channels
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


class MultiEncoder(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults):
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

    def forward(self, x, emb=None, context=None):
        x = self.model(x, emb, context)
        hints = [self.zero_layer(x, emb, context)]
        for layer, zero_layer in zip(self.output_blocks, self.zero_blocks):
            x = layer(x, emb, context)
            hints.append(zero_layer(x, emb, context))
        return hints


class FullEncoder(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults):
        super().__init__()
        self.block_num = len(ch_mults)
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
            zero_module(conv_nd(2, 256, output_chs[0], 3, padding=1))
        )
        prev_ch = output_chs[0]
        self.enc_zeros = nn.ModuleList()
        self.dec_zeros = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        self.enc_zeros.append(make_zero_conv(output_chs[0]))
        self.dec_zeros.append(make_zero_conv(output_chs[0]))
        for i in range(self.block_num):
            self.output_blocks.append(TimestepEmbedSequential(
                nn.SiLU(),
                conv_nd(2, prev_ch, prev_ch, 3, padding=1, stride=2 if i != self.block_num-1 else 1),
                nn.SiLU(),
                conv_nd(2, prev_ch, output_chs[i], 3, padding=1)
            ))
            prev_ch = output_chs[i]

            self.enc_zeros.append(make_zero_conv(output_chs[i]))
            if i != self.block_num - 1:
                self.dec_zeros.append(make_zero_conv(output_chs[i], output_chs[i+1]))


    def forward(self, x, emb=None, context=None):
        x = self.model(x, emb, context)

        enc_iter, dec_iter = iter(self.enc_zeros), iter(self.dec_zeros)
        hints = [next(enc_iter)(x, emb, context)]
        dhints = [next(dec_iter)(x, emb, context)]

        for idx, layer in enumerate(self.output_blocks):
            x = layer(x, emb, context)
            hints.append(next(enc_iter)(x, emb, context))
            if idx != self.block_num - 1:
                dhints.append(next(dec_iter)(x, emb, context))
        dhints.reverse()
        hints.extend(dhints)
        return hints



class FullEncoder2(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults):
        super().__init__()
        self.block_num = len(ch_mults)
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
            zero_module(conv_nd(2, 256, output_chs[0], 3, padding=1))
        )
        prev_ch = output_chs[0]
        self.enc_zeros = nn.ModuleList()
        self.dec_zeros = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        self.enc_zeros.append(make_zero_conv(output_chs[0]))
        self.dec_zeros.append(make_zero_conv(output_chs[0]))
        for i in range(self.block_num):
            self.output_blocks.append(TimestepEmbedSequential(
                nn.SiLU(),
                conv_nd(2, prev_ch, output_chs[i], 3, padding=1, stride=2 if i != self.block_num-1 else 1),
                nn.SiLU(),
                conv_nd(2, output_chs[i], output_chs[i], 3, padding=1)
            ))
            prev_ch = output_chs[i]

            self.enc_zeros.append(make_zero_conv(output_chs[i]))
            if i != self.block_num - 1:
                self.dec_zeros.append(make_zero_conv(output_chs[i], output_chs[i+1]))

    def forward(self, x, emb=None, context=None):
        x = self.model(x, emb, context)

        enc_iter, dec_iter = iter(self.enc_zeros), iter(self.dec_zeros)
        hints = [next(enc_iter)(x, emb, context)]
        dhints = [next(dec_iter)(x, emb, context)]

        for idx, layer in enumerate(self.output_blocks):
            x = layer(x, emb, context)
            hints.append(next(enc_iter)(x, emb, context))
            if idx != self.block_num - 1:
                dhints.append(next(dec_iter)(x, emb, context))
        dhints.reverse()
        hints.extend(dhints)
        return hints