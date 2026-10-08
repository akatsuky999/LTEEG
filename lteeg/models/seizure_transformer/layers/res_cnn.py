"""Residual CNN blocks applied at the bottleneck before the Transformer."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1d(nn.Module):
    """Drops whole feature channels (the same mask for every time step)."""

    def __init__(self, drop_rate: float):
        super().__init__()
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x.unsqueeze(-1)).squeeze(-1)


class ResCNNBlock(nn.Module):
    def __init__(self, filters: int, ker: int, drop_rate: float):
        super().__init__()
        # kernel 2 is padded on the right only (TensorFlow 'same'), kernel 3 symmetric
        self.manual_padding = ker == 2
        padding = 0 if self.manual_padding else ker // 2
        self.dropout = SpatialDropout1d(drop_rate)
        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)
        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.relu(self.norm1(x)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        y = self.conv1(y)
        y = self.dropout(F.relu(self.norm2(y)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        y = self.conv2(y)
        return x + y


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int], filters: int, drop_rate: float):
        super().__init__()
        self.members = nn.ModuleList([ResCNNBlock(filters, k, drop_rate) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.members:
            x = m(x)
        return x
