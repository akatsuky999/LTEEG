"""Convolutional encoder/decoder of the U-shaped network.

The encoder halves the time axis at every level (``MaxPool1d(2, ceil_mode=True)``)
and keeps each level's activation as a skip connection; the decoder upsamples by 2
(nearest), crops to the matching skip's length and adds it.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int, filters: Sequence[int], kernel_sizes: Sequence[int]):
        super().__init__()
        convs = []
        for cin, cout, k in zip([input_channels] + list(filters[:-1]), filters, kernel_sizes):
            convs.append(nn.Conv1d(cin, cout, k, padding=k // 2))
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for conv in self.convs:
            x = F.elu(conv(x))
            skips.append(x)
            x = self.pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, input_channels: int, filters: Sequence[int], kernel_sizes: Sequence[int]):
        super().__init__()
        convs = []
        for cin, cout, k in zip([input_channels] + list(filters[:-1]), filters, kernel_sizes):
            convs.append(nn.Conv1d(cin, cout, k, padding=k // 2))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            skip = skips[-(i + 1)]
            x = F.interpolate(x, scale_factor=2, mode="nearest")[..., : skip.shape[-1]]
            x = F.elu(conv(x)) + skip
        return x
