"""Node features: the signal is cut into consecutive segments ("time steps") and each
channel's segment becomes one feature vector (Tang et al., 2022, Sec. 3.1).

``fft``: log-amplitude of the FFT of the segment, positive frequencies only
(``floor(step / 2)`` bins, DC included, Nyquist excluded), with zero amplitudes
replaced by 1e-8 before the log -- exactly ``computeFFT`` of the reference code.
``raw``: the segment's samples themselves.
"""

from __future__ import annotations

import torch

FEATURE_KINDS = ("fft", "raw")


def feature_dim(step: int, kind: str) -> int:
    if kind == "fft":
        return step // 2
    if kind == "raw":
        return step
    raise ValueError(f"unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")


def step_features(x: torch.Tensor, step: int, kind: str) -> torch.Tensor:
    """``(B, N, S * step)`` signal -> ``(B, S, N, D)`` node features for ``S`` time steps."""
    b, n, t = x.shape
    if t % step:
        raise ValueError(f"signal length {t} is not a multiple of the step size {step}")
    seg = x.reshape(b, n, t // step, step).transpose(1, 2)  # (B, S, N, step)
    if kind == "raw":
        return seg
    if kind != "fft":
        raise ValueError(f"unknown feature kind {kind!r}; expected one of {FEATURE_KINDS}")
    amp = torch.fft.rfft(seg, n=step, dim=-1)[..., : step // 2].abs()
    amp = torch.where(amp == 0, torch.full_like(amp, 1e-8), amp)
    return torch.log(amp)
