"""Building blocks of the template model. Put every block your model needs here (or in
more files of this folder) and keep ``model.py`` a readable assembly of them."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Residual ``Conv1d -> BatchNorm -> GELU`` block with 'same' padding."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))
