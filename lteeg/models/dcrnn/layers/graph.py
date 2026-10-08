"""Graph construction and diffusion supports (Li et al., 2018; Tang et al., 2022).

Adjacency
---------
* :func:`correlation_adjacency` -- Tang et al.'s per-sample "correlation graph": the
  absolute normalised zero-lag cross-correlation between the channels' feature
  sequences, self-loops of weight 1, and for every node only its ``top_k`` strongest
  neighbours (a directed graph). Same result as ``comp_xcorr(mode='valid')`` +
  ``keep_topk(directed=True)`` of the reference code, computed for a whole batch.
* a fixed adjacency matrix given by the user (e.g. a distance graph).

Supports (``filter_type``)
--------------------------
* ``laplacian``: scaled Laplacian ``2 L / lambda_max - I`` of the symmetrised graph,
  ``L = I - D^-1/2 A D^-1/2`` (ChebNet); one support.
* ``random_walk``: ``(D^-1 A)^T``; one support.
* ``dual_random_walk``: ``(D^-1 A)^T`` and ``(D_in^-1 A^T)^T`` (forward and backward
  diffusion of a directed graph); two supports.

All functions accept a single ``(N, N)`` matrix or a batch ``(B, N, N)``.
"""

from __future__ import annotations

from typing import List

import torch

FILTER_TYPES = ("laplacian", "random_walk", "dual_random_walk")


def num_supports(filter_type: str) -> int:
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"unknown filter_type {filter_type!r}; expected one of {FILTER_TYPES}")
    return 2 if filter_type == "dual_random_walk" else 1


def correlation_adjacency(features: torch.Tensor, top_k: int) -> torch.Tensor:
    """``(B, S, N, D)`` node features -> ``(B, N, N)`` sparsified |correlation| adjacency.

    Row ``i`` keeps the self-loop and the ``top_k`` largest off-diagonal entries. The
    correlation is computed in float64 so that the neighbour ranking does not depend
    on the device or on float32 rounding.
    """
    b, s, n, d = features.shape
    if not 1 <= top_k <= n - 1:
        raise ValueError(f"top_k must be in [1, {n - 1}] for {n} nodes, got {top_k}")
    z = features.transpose(1, 2).reshape(b, n, s * d).double()
    gram = z @ z.transpose(1, 2)
    energy = torch.diagonal(gram, dim1=1, dim2=2)
    scale = torch.sqrt(energy.unsqueeze(2) * energy.unsqueeze(1))
    corr = torch.where(scale > 0, gram / torch.where(scale > 0, scale, torch.ones_like(scale)), gram).abs()
    eye = torch.eye(n, dtype=torch.bool, device=features.device).expand(b, n, n)
    corr = torch.where(eye, torch.ones_like(corr), corr)
    neighbours = corr.masked_fill(eye, 0.0).topk(top_k, dim=-1).indices
    keep = eye.clone()
    keep.scatter_(-1, neighbours, True)
    return (corr * keep).to(features.dtype)


def random_walk_matrix(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalised transition matrix ``D^-1 A`` (rows with zero degree stay zero)."""
    deg = adj.sum(-1, keepdim=True)
    return adj / torch.where(deg > 0, deg, torch.ones_like(deg))


def scaled_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """``2 L / lambda_max - I`` with ``L`` the normalised Laplacian of ``max(A, A^T)``.

    ``lambda_max`` is the largest eigenvalue of ``L`` (the reference computes it with
    ``eigsh(which='LM')``); a graph without edges (``lambda_max = 0``) falls back to the
    theoretical bound 2.
    """
    a = torch.maximum(adj, adj.transpose(-1, -2))
    deg = a.sum(-1)
    safe_deg = torch.where(deg > 0, deg, torch.ones_like(deg))
    d_inv_sqrt = torch.where(deg > 0, safe_deg.rsqrt(), torch.zeros_like(deg))
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    lap = eye - d_inv_sqrt.unsqueeze(-1) * a * d_inv_sqrt.unsqueeze(-2)
    lam = torch.linalg.eigvalsh(lap.double())[..., -1].to(a.dtype)
    lam = torch.where(lam > 1e-6, lam, torch.full_like(lam, 2.0))
    return 2.0 * lap / lam[..., None, None] - eye


def diffusion_supports(adj: torch.Tensor, filter_type: str) -> List[torch.Tensor]:
    """Supports used by the diffusion convolution, each shaped like ``adj``."""
    num_supports(filter_type)  # validates the name
    if filter_type == "laplacian":
        return [scaled_laplacian(adj)]
    forward = random_walk_matrix(adj).transpose(-1, -2)
    if filter_type == "random_walk":
        return [forward]
    backward = random_walk_matrix(adj.transpose(-1, -2)).transpose(-1, -2)
    return [forward, backward]
