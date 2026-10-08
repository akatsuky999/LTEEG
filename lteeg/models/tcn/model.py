"""Lightweight dilated temporal convolutional network.

A strided convolutional stem reduces the sampling rate by ``stem_stride``, a stack of
residual dilated convolutions (dilation 1, 2, 4, ...) builds a long receptive field,
and a 1x1 head emits logits at the reduced rate (``output_stride = stem_stride``);
the framework interpolates them to one value per sample. Small and fast: useful as
a second baseline and for smoke tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ResidualDilatedBlock


class DilatedTCN(nn.Module):
    def __init__(self, in_channels: int, in_samples: int, num_outputs: int = 1, hidden: int = 64,
                 levels: int = 8, kernel_size: int = 3, stem_stride: int = 8, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.output_stride = stem_stride
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=2 * stem_stride + 1, stride=stem_stride, padding=stem_stride),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualDilatedBlock(hidden, kernel_size, 2 ** i, dropout)
                                      for i in range(levels)])
        self.head = nn.Conv1d(hidden, num_outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
