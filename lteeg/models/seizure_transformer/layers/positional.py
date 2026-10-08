"""Sinusoidal positional encoding (Vaswani et al., 2017) for the bottleneck tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence of {x.size(1)} tokens exceeds positional table ({self.pe.size(1)})")
        return self.dropout(x + self.pe[:, : x.size(1)].to(x.dtype))
