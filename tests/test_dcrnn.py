"""DCRNN: features, graph, supports and the point-level adaptation.

The numpy helpers below restate the reference definitions (tsy935/eeg-gnn-ssl:
computeFFT, comp_xcorr(mode='valid'), keep_topk(directed=True),
calculate_random_walk_matrix, calculate_scaled_laplacian) so the port is pinned to
them without depending on the reference repository.
"""

import numpy as np
import pytest
import torch

from lteeg.models.dcrnn import DCRNN
from lteeg.models.dcrnn.layers import (
    correlation_adjacency,
    diffusion_supports,
    random_walk_matrix,
    scaled_laplacian,
    step_features,
)


def _ref_fft(seg: np.ndarray, n: int) -> np.ndarray:
    amp = np.abs(np.fft.fft(seg, n=n, axis=-1)[..., : n // 2])
    amp[amp == 0.0] = 1e-8
    return np.log(amp)


def _ref_adjacency(clip: np.ndarray, top_k: int) -> np.ndarray:
    """clip: (S, N, D)."""
    n = clip.shape[1]
    flat = np.transpose(clip, (1, 0, 2)).reshape(n, -1)
    adj = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = np.sum(flat[i] * flat[j])
            e = np.sum(flat[i] ** 2) * np.sum(flat[j] ** 2)
            adj[i, j] = adj[j, i] = c / np.sqrt(e) if e > 0 else c
    adj = np.abs(adj)
    no_self = adj.copy()
    np.fill_diagonal(no_self, 0)
    mask = np.eye(n, dtype=bool)
    for i, row in enumerate((-no_self).argsort(axis=-1)[:, :top_k]):
        mask[i, row] = True
    return adj * mask


def _ref_random_walk(adj: np.ndarray) -> np.ndarray:
    d = adj.sum(1)
    d_inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    return d_inv[:, None] * adj


def _ref_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    a = np.maximum(adj, adj.T)
    d = a.sum(1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    lap = np.eye(len(a)) - d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    lam = np.linalg.eigvalsh(lap)[-1]
    return 2 * lap / lam - np.eye(len(a))


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 8 * 64))
    x[:, :3] += np.sin(np.arange(8 * 64) / 5.0)  # correlated block -> non-trivial graph
    return x


def test_fft_features_match_reference(signal):
    feats = step_features(torch.from_numpy(signal), 64, "fft").numpy()
    assert feats.shape == (2, 8, 6, 32)
    ref = np.stack([np.stack([_ref_fft(signal[b][:, t * 64:(t + 1) * 64], 64) for t in range(8)]) for b in range(2)])
    np.testing.assert_allclose(feats, ref, atol=1e-10)
    zero = step_features(torch.zeros(1, 2, 64, dtype=torch.float64), 64, "fft")
    assert torch.allclose(zero, torch.full_like(zero, np.log(1e-8)))


def test_raw_features_are_segments(signal):
    raw = step_features(torch.from_numpy(signal), 64, "raw")
    assert torch.equal(raw[1, 3, 2], torch.from_numpy(signal[1, 2, 192:256]))


def test_correlation_adjacency_matches_reference(signal):
    feats = step_features(torch.from_numpy(signal), 64, "fft")
    adj = correlation_adjacency(feats, top_k=2).numpy()
    for b in range(2):
        np.testing.assert_allclose(adj[b], _ref_adjacency(feats[b].numpy(), 2), atol=1e-12)
        assert ((adj[b] > 0).sum(axis=1) == 3).all()  # self-loop + top_k neighbours per row
        assert np.allclose(np.diag(adj[b]), 1.0)


def test_supports_match_reference(signal):
    adj = correlation_adjacency(step_features(torch.from_numpy(signal), 64, "fft"), top_k=2)
    fwd, bwd = diffusion_supports(adj, "dual_random_walk")
    for b in range(2):
        a = adj[b].numpy()
        np.testing.assert_allclose(fwd[b].numpy(), _ref_random_walk(a).T, atol=1e-12)
        np.testing.assert_allclose(bwd[b].numpy(), _ref_random_walk(a.T).T, atol=1e-12)
        np.testing.assert_allclose(scaled_laplacian(adj[b]).numpy(), _ref_scaled_laplacian(a), atol=1e-10)
    assert torch.allclose(random_walk_matrix(adj).sum(-1), torch.ones(2, 6, dtype=adj.dtype))
    (single,) = diffusion_supports(adj, "random_walk")
    assert torch.equal(single, fwd)


def test_scaled_laplacian_spectrum_and_empty_graph():
    a = torch.rand(5, 5, dtype=torch.float64)
    eig = torch.linalg.eigvalsh(scaled_laplacian(a))
    assert eig.min() >= -1 - 1e-9 and abs(eig.max() - 1) < 1e-9
    assert torch.equal(scaled_laplacian(torch.eye(3)), -torch.eye(3))  # no edges: lambda_max -> 2


def test_one_logit_per_time_step_and_exact_alignment():
    m = DCRNN(in_channels=4, in_samples=640, fs=64.0, rnn_units=8, num_rnn_layers=1, top_k=2).eval()
    x = torch.randn(2, 4, 640)
    with torch.no_grad():
        steps = m(x)
        assert steps.shape == (2, 1, 10) and m.output_stride == 64
        ragged = m(x[..., :600])  # not a multiple of the step: the model aligns to T itself
    assert ragged.shape == (2, 1, 600)


def test_reference_parameter_layout():
    m = DCRNN(in_channels=19, in_samples=200 * 60, fs=200.0, feature_norm="none")
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    # eeg-gnn-ssl DCRNNModel_classification, 19 nodes, input_dim 100, 2 layers x 64 units, dual random walk
    assert shapes == {
        "encoder.encoding_cells.0.dconv_gate.weight": ((100 + 64) * 5, 128),
        "encoder.encoding_cells.0.dconv_gate.biases": (128,),
        "encoder.encoding_cells.0.dconv_candidate.weight": ((100 + 64) * 5, 64),
        "encoder.encoding_cells.0.dconv_candidate.biases": (64,),
        "encoder.encoding_cells.1.dconv_gate.weight": ((64 + 64) * 5, 128),
        "encoder.encoding_cells.1.dconv_gate.biases": (128,),
        "encoder.encoding_cells.1.dconv_candidate.weight": ((64 + 64) * 5, 64),
        "encoder.encoding_cells.1.dconv_candidate.biases": (64,),
        "fc.weight": (1, 64),
        "fc.bias": (1,),
    }


def test_static_graph():
    adj = np.ones((4, 4)) - np.eye(4)
    m = DCRNN(in_channels=4, in_samples=256, fs=64.0, graph="static", adjacency=adj.tolist(), rnn_units=4,
              num_rnn_layers=1)
    assert m.filter_type == "laplacian" and m.static_supports.shape == (1, 4, 4)
    assert "static_supports" not in m.state_dict()  # derived from the config, not saved
    with torch.no_grad():
        assert m.eval()(torch.randn(1, 4, 256)).shape == (1, 1, 4)


@pytest.mark.parametrize("kwargs, message", [
    ({"step_sec": 0.3}, "whole number of samples"),
    ({"top_k": 4}, "top_k"),
    ({"graph": "distance"}, "graph must be"),
    ({"features": "wavelet"}, "features must be"),
    ({"graph": "static"}, "needs `adjacency`"),
    ({"graph": "static", "adjacency": [[1.0] * 3] * 3}, "4 x 4"),
    ({"graph": "static", "adjacency": np.eye(4).tolist()}, "no edge"),
    ({"adjacency": np.ones((4, 4)).tolist()}, "only used with graph='static'"),
    ({"filter_type": "chebyshev"}, "filter_type"),
    ({"activation": "gelu"}, "activation"),
])
def test_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DCRNN(in_channels=4, in_samples=256, fs=64.0, **kwargs)


def test_feature_norm_learns_training_statistics():
    m = DCRNN(in_channels=4, in_samples=256, fs=64.0, rnn_units=4, num_rnn_layers=1, top_k=2)
    m.train()(torch.randn(3, 4, 256) * 5)
    assert int(m.feature_norm.num_batches_tracked) == 1 and m.feature_norm.running_mean.abs().sum() > 0


def test_bf16_autocast_cpu():
    m = DCRNN(in_channels=4, in_samples=256, fs=64.0, rnn_units=4, num_rnn_layers=1, top_k=2).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(torch.randn(2, 4, 256))
    assert torch.isfinite(out.float()).all()
