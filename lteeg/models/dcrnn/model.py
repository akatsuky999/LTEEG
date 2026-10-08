"""DCRNN for point-level seizure detection.

Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018) in the EEG
setting of Tang et al., "Self-Supervised Graph Neural Networks for Improved
Electroencephalographic Seizure Analysis" (ICLR 2022); reference code
https://github.com/tsy935/eeg-gnn-ssl (``DCRNNModel_classification``).

For a window ``x`` of shape ``(B, N, T)``:

1. **time steps**: cut into ``S = ceil(T / step)`` segments of ``step = step_sec * fs``
   samples (the last one zero-padded if needed); each channel's segment becomes a
   node feature vector (log-amplitude FFT by default) -> ``(B, S, N, D)``;
2. **graph**: a correlation graph computed per window (default) or a fixed
   adjacency, turned into diffusion supports;
3. **feature normalisation** per channel (``feature_norm``);
4. **DCGRU encoder** over the ``S`` steps -> hidden states ``(B, S, N, H)``;
5. **head**: ``Linear(ReLU(Dropout(h)))`` on every node, max over nodes -> logits
   ``(B, num_outputs, S)``.

Point-level adaptation: the reference classifies a 60-s clip with this head applied
to the top layer's hidden state at the *last* step. Here the head is applied at
*every* step, giving one logit per step (per second by default); at the last step it
is exactly the reference's clip logit. The framework interpolates the step logits to
one value per sample. See README.md for parameters, deviations and verification.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    FEATURE_KINDS,
    DCRNNEncoder,
    correlation_adjacency,
    diffusion_supports,
    feature_dim,
    num_supports,
    step_features,
)

GRAPHS = ("correlation", "static")
FEATURE_NORMS = ("batch", "none")


class DCRNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_samples: int,
        num_outputs: int = 1,
        fs: float = 256.0,
        step_sec: float = 1.0,
        features: str = "fft",
        graph: str = "correlation",
        top_k: int = 3,
        adjacency: Optional[Sequence[Sequence[float]]] = None,
        filter_type: Optional[str] = None,
        num_rnn_layers: int = 2,
        rnn_units: int = 64,
        max_diffusion_step: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        feature_norm: str = "batch",
    ):
        super().__init__()
        if in_channels < 2:
            raise ValueError("DCRNN needs at least 2 channels (graph nodes)")
        step = int(round(step_sec * fs))
        if step < 2 or abs(step - step_sec * fs) > 1e-6 * step:
            raise ValueError(f"step_sec * fs = {step_sec * fs:g} must be a whole number of samples >= 2")
        if in_samples < step:
            raise ValueError(f"window of {in_samples} samples is shorter than one time step ({step} samples)")
        if features not in FEATURE_KINDS:
            raise ValueError(f"features must be one of {FEATURE_KINDS}, got {features!r}")
        if graph not in GRAPHS:
            raise ValueError(f"graph must be one of {GRAPHS}, got {graph!r}")
        if feature_norm not in FEATURE_NORMS:
            raise ValueError(f"feature_norm must be one of {FEATURE_NORMS}, got {feature_norm!r}")
        if num_rnn_layers < 1 or rnn_units < 1 or max_diffusion_step < 0:
            raise ValueError("num_rnn_layers and rnn_units must be >= 1, max_diffusion_step >= 0")
        if graph == "correlation":
            if adjacency is not None:
                raise ValueError("adjacency is only used with graph='static'")
            if not 1 <= top_k <= in_channels - 1:
                raise ValueError(f"top_k must be in [1, {in_channels - 1}], got {top_k}")
            filter_type = filter_type or "dual_random_walk"  # Tang et al.: directed per-window graph
        else:
            if adjacency is None:
                raise ValueError("graph='static' needs `adjacency`, an N x N matrix of non-negative weights")
            filter_type = filter_type or "laplacian"  # Tang et al.: undirected distance graph
        n_supports = num_supports(filter_type)

        self.in_channels, self.in_samples, self.num_outputs = in_channels, in_samples, num_outputs
        self.step = self.output_stride = step
        self.features, self.graph, self.top_k, self.filter_type = features, graph, top_k, filter_type

        # Tang et al. standardise features with per-channel training-set mean/std; a non-affine
        # BatchNorm over the same per-channel statistics plays that role (momentum=None: the running
        # averages are cumulative, i.e. converge to the training-set statistics).
        self.feature_norm = (nn.BatchNorm1d(in_channels, affine=False, momentum=None)
                             if feature_norm == "batch" else nn.Identity())
        self.encoder = DCRNNEncoder(feature_dim(step, features), rnn_units, num_rnn_layers, max_diffusion_step,
                                    n_supports, activation)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_units, num_outputs)

        if graph == "static":
            adj = torch.as_tensor(adjacency, dtype=torch.float32)
            if adj.shape != (in_channels, in_channels):
                raise ValueError(f"adjacency must be {in_channels} x {in_channels}, got {tuple(adj.shape)}")
            if not torch.isfinite(adj).all() or (adj < 0).any():
                raise ValueError("adjacency must contain finite, non-negative weights")
            if not (adj * (1 - torch.eye(in_channels))).any():
                raise ValueError("adjacency has no edge between distinct channels")
            self.register_buffer("static_supports", torch.stack(diffusion_supports(adj, filter_type)),
                                 persistent=False)
        else:
            self.register_buffer("static_supports", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, T), got {tuple(x.shape)}")
        b, n, t = x.shape
        if t < self.step:
            raise ValueError(f"input of {t} samples is shorter than one time step ({self.step} samples)")
        s = -(-t // self.step)
        pad = s * self.step - t

        # Features and graph in float32 even under autocast (FFT, correlation ranking).
        with torch.autocast(device_type=x.device.type, enabled=False):
            signal = F.pad(x.float(), (0, pad)) if pad else x.float()
            feats = step_features(signal, self.step, self.features)  # (B, S, N, D)
            if self.static_supports is None:
                with torch.no_grad():  # the graph is derived from the data, not learned
                    supports = diffusion_supports(correlation_adjacency(feats, self.top_k), self.filter_type)
            else:
                supports = list(self.static_supports)
            d = feats.shape[-1]
            per_channel = feats.transpose(1, 2).reshape(b, n, s * d)
            feats = self.feature_norm(per_channel).reshape(b, n, s, d).transpose(1, 2)

        hidden = self.encoder(feats, supports)  # (B, S, N, H)
        logits = self.fc(F.relu(self.dropout(hidden))).max(dim=2).values.transpose(1, 2)  # (B, K, S)
        if pad:  # exact alignment: interpolate over the padded span, then crop to T
            logits = F.interpolate(logits, size=s * self.step, mode="linear", align_corners=False)[..., :t]
        return logits
