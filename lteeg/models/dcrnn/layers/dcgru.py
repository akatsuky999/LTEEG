"""Diffusion-convolutional GRU (Li et al., 2018), as used by Tang et al. (2022).

Tensors are kept as ``(B, N, F)`` (batch, nodes, features) instead of the reference's
flattened ``(B, N * F)``; the arithmetic, the order of the diffusion terms and the
parameter layout (``weight``: ``(F * M, out)``, ``biases``: ``(out,)``) are the
reference ones, so its weights load unchanged.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    """``y = [x, T_1(S) x, ..., T_K(S) x  for each support S] W + b`` on ``x = [input, state]``.

    ``T_k`` follows the Chebyshev-style recursion of the reference implementation:
    ``x_1 = S x_0``, ``x_k = 2 S x_{k-1} - x_{k-2}``.

    Note (kept for fidelity): as in the reference code (and the original DCRNN), the
    recursion variables are not reset between supports, so with two supports and
    ``max_diffusion_step >= 2`` the second support's terms start from ``S_1 x`` instead
    of ``x``.
    """

    def __init__(self, num_supports: int, input_dim: int, hid_dim: int, max_diffusion_step: int,
                 output_dim: int, bias_start: float = 0.0):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        self.num_matrices = num_supports * max_diffusion_step + 1
        self.input_size = input_dim + hid_dim
        self.weight = nn.Parameter(torch.empty(self.input_size * self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.empty(output_dim))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.constant_(self.biases, bias_start)

    def forward(self, supports: List[torch.Tensor], inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``inputs`` ``(B, N, D)``, ``state`` ``(B, N, H)``, each support ``(N, N)`` or
        ``(B, N, N)`` -> ``(B, N, output_dim)``."""
        x0 = torch.cat([inputs, state], dim=-1)
        terms = [x0]
        if self.max_diffusion_step > 0:
            for support in supports:
                x1 = support @ x0
                terms.append(x1)
                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2 * (support @ x1) - x0
                    terms.append(x2)
                    x1, x0 = x2, x1
        b, n, f = terms[0].shape
        x = torch.stack(terms, dim=-1).reshape(b, n, f * len(terms))  # feature-major, as the reference
        return x @ self.weight + self.biases


class DCGRUCell(nn.Module):
    """GRU cell whose gates are diffusion graph convolutions."""

    def __init__(self, input_dim: int, num_units: int, max_diffusion_step: int, num_supports: int,
                 activation: str = "tanh"):
        super().__init__()
        if activation not in ("tanh", "relu"):
            raise ValueError(f"activation must be 'tanh' or 'relu', got {activation!r}")
        self.num_units = num_units
        self.activation = torch.tanh if activation == "tanh" else torch.relu
        self.dconv_gate = DiffusionGraphConv(num_supports, input_dim, num_units, max_diffusion_step, 2 * num_units)
        self.dconv_candidate = DiffusionGraphConv(num_supports, input_dim, num_units, max_diffusion_step,
                                                  num_units)

    def forward(self, supports: List[torch.Tensor], inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """One step: ``inputs`` ``(B, N, D)``, ``state`` ``(B, N, H)`` -> new state ``(B, N, H)``."""
        r, u = torch.sigmoid(self.dconv_gate(supports, inputs, state)).split(self.num_units, dim=-1)
        c = self.activation(self.dconv_candidate(supports, inputs, r * state))
        return u * state + (1.0 - u) * c


class DCRNNEncoder(nn.Module):
    """Stack of DCGRU layers run over the time steps (layer by layer, zero initial state).

    Returns the top layer's hidden state at every step: ``(B, S, N, H)``.
    """

    def __init__(self, input_dim: int, hid_dim: int, num_rnn_layers: int, max_diffusion_step: int,
                 num_supports: int, activation: str = "tanh"):
        super().__init__()
        self.hid_dim = hid_dim
        dims = [input_dim] + [hid_dim] * (num_rnn_layers - 1)
        self.encoding_cells = nn.ModuleList(
            [DCGRUCell(d, hid_dim, max_diffusion_step, num_supports, activation) for d in dims])

    def forward(self, inputs: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        b, s, n, _ = inputs.shape
        current = inputs
        for cell in self.encoding_cells:
            state = inputs.new_zeros(b, n, self.hid_dim)
            outputs = []
            for t in range(s):
                state = cell(supports, current[:, t], state)
                outputs.append(state)
            current = torch.stack(outputs, dim=1)
        return current
