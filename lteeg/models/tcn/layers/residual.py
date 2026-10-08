"""Residual dilated convolution block of the TCN."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualDilatedBlock(nn.Module):
    """``x + Conv1x1(Dropout(GELU(BN(DilatedConv(x)))))`` with 'same' padding (odd kernels)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
