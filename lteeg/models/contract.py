"""Executable form of the model contract described in :mod:`lteeg.models.base`.

:func:`check_contract` builds a model twice from a factory and verifies, on random
input, everything the training loop and the long-range inference rely on. It raises
:class:`ModelContractError` naming the first violated rule, or returns a
:class:`ContractReport` with size and speed figures.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F

from ..task import align_logits
from .base import ModelOutput, as_output, count_parameters


class ModelContractError(AssertionError):
    """A model violates the framework contract (see lteeg/models/base.py)."""


@dataclass
class ContractReport:
    model: str
    parameters: int
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, ...]
    fwd_bwd_sec: float
    initial_loss: float
    peak_mem_gb: Optional[float] = None

    @property
    def output_stride(self) -> float:
        return self.input_shape[-1] / self.output_shape[-1]

    def __str__(self) -> str:
        lines = [f"{self.model}: {self.parameters / 1e6:.2f}M trainable parameters",
                 f"input {self.input_shape} -> logits {self.output_shape} (stride {self.output_stride:g})",
                 f"train-mode forward+backward {self.fwd_bwd_sec:.2f}s, initial loss {self.initial_loss:.4f}"]
        if self.peak_mem_gb is not None:
            lines.append(f"peak GPU memory (forward+backward, no optimizer state): {self.peak_mem_gb:.1f} GB")
        lines.append("contract OK: shapes, finite values, gradients, untouched input, reproducible eval, "
                     "independent batch elements, state-dict round trip")
        return "\n".join(lines)


def _fail(msg: str) -> None:
    raise ModelContractError(msg)


def _check_logits(t: torch.Tensor, what: str, batch: int, num_outputs: int, length: int) -> None:
    if not isinstance(t, torch.Tensor):
        _fail(f"{what} must be a tensor, got {type(t).__name__}")
    if t.ndim != 3 or t.shape[0] != batch or t.shape[1] != num_outputs:
        _fail(f"{what} must have shape (B={batch}, num_outputs={num_outputs}, T'), got {tuple(t.shape)}")
    if not 1 <= t.shape[2] <= length:
        _fail(f"{what} has {t.shape[2]} time steps; expected 1 <= T' <= T = {length}")
    if not t.is_floating_point():
        _fail(f"{what} must be floating point, got {t.dtype}")
    if not torch.isfinite(t).all():
        _fail(f"{what} contains non-finite values")


@contextlib.contextmanager
def _full_fp32(device: torch.device) -> Iterator[None]:
    """Disable TF32 on CUDA while checking: TF32 convolutions/matmuls (on by default for
    cuDNN on Ampere+) round to ~1e-3 and would make exact-comparison checks flaky."""
    if device.type != "cuda":
        yield
        return
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_contract(
    factory: Callable[[], torch.nn.Module],
    in_channels: int,
    in_samples: int,
    num_outputs: int,
    *,
    batch_size: int = 2,
    device: str = "cpu",
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    name: Optional[str] = None,
) -> ContractReport:
    """Verify the model contract.

    ``factory()`` must return a freshly constructed model (it is called twice, the
    second time for the state-dict round trip). The model is checked in train mode
    (forward + backward) and eval mode (reproducibility, batch independence). The
    tolerances apply to comparisons of eval-mode logits computed in different ways.
    """
    dev = torch.device(device)
    with _full_fp32(dev):
        return _check(factory, in_channels, in_samples, num_outputs, batch_size, dev, seed, rtol, atol, name)


def _check(factory: Callable[[], torch.nn.Module], in_channels: int, in_samples: int, num_outputs: int,
           batch_size: int, dev: torch.device, seed: int, rtol: float, atol: float,
           name: Optional[str]) -> ContractReport:
    torch.manual_seed(seed)
    model = factory()
    if not isinstance(model, torch.nn.Module):
        _fail(f"the model factory returned {type(model).__name__}, not a torch.nn.Module")
    model = model.to(dev)
    label = name or type(model).__name__
    wants_meta = bool(getattr(model, "wants_meta", False))
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, in_channels, in_samples, generator=gen).to(dev)

    def meta_for(xb: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = xb.shape[0]
        return {"patient": torch.zeros(b, dtype=torch.long, device=dev),
                "rec": torch.zeros(b, dtype=torch.long, device=dev),
                "start": torch.zeros(b, dtype=torch.long, device=dev)}

    def run(m: torch.nn.Module, xb: torch.Tensor) -> ModelOutput:
        return as_output(m(xb, meta_for(xb)) if wants_meta else m(xb))

    # ---- train mode: forward + backward --------------------------------------------
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    x_before = x.clone()
    t0 = time.time()
    out = run(model, x)
    if not torch.equal(x, x_before):
        _fail("forward modified its input in place")
    _check_logits(out.logits, "logits", batch_size, num_outputs, in_samples)
    for i, aux in enumerate(out.aux_logits):
        _check_logits(aux, f"aux_logits[{i}]", batch_size, num_outputs, in_samples)
    for key, value in out.losses.items():
        if not (isinstance(value, torch.Tensor) and value.ndim == 0 and torch.isfinite(value)):
            _fail(f"losses[{key!r}] must be a finite scalar tensor")
    if num_outputs == 1:
        target = (torch.rand(batch_size, in_samples, generator=gen) > 0.5).float().to(dev)
    else:
        target = torch.randint(0, num_outputs, (batch_size, in_samples), generator=gen).to(dev)

    def crit(logits: torch.Tensor) -> torch.Tensor:
        aligned = align_logits(logits.float(), in_samples)
        if num_outputs == 1:
            return F.binary_cross_entropy_with_logits(aligned[:, 0], target)
        return F.cross_entropy(aligned, target)

    loss = crit(out.logits) + sum(crit(a) for a in out.aux_logits) + sum(out.losses.values())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if trainable and not loss.requires_grad:
        _fail("no parameter received a gradient: the output is detached from the parameters")
    if loss.requires_grad:
        loss.backward()
    _sync(dev)
    fwd_bwd = time.time() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == "cuda" else None
    if trainable and all(p.grad is None for _, p in trainable):
        _fail("no parameter received a gradient: the output is detached from the parameters")
    bad = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        _fail(f"non-finite gradients in {bad[:5]}")
    model.zero_grad(set_to_none=True)

    # ---- eval mode: reproducible, batch elements independent --------------------------
    model.eval()
    with torch.no_grad():
        y1 = run(model, x).logits
        y2 = run(model, x).logits
        if not torch.allclose(y1, y2, rtol=rtol, atol=atol):
            _fail(f"eval-mode forward is not reproducible (max diff {(y1 - y2).abs().max().item():.3g})")
        k = min(batch_size, 4)
        if k >= 2:
            single = torch.cat([run(model, x[i:i + 1]).logits for i in range(k)])
            if not torch.allclose(single, y1[:k], rtol=rtol, atol=atol):
                diff = (single - y1[:k]).abs().max().item()
                _fail(f"batch elements interact in eval mode (max diff {diff:.3g} between a batch and its "
                      f"elements run one by one): check reshapes/permutes over the batch dimension")

        # ---- state-dict round trip ---------------------------------------------------
        clone = factory().to(dev)
        try:
            clone.load_state_dict(model.state_dict(), strict=True)
        except RuntimeError as e:
            raise ModelContractError(f"state dict does not load strictly into a fresh instance: {e}") from e
        clone.eval()
        y3 = run(clone, x).logits
        if not torch.allclose(y3, y1, rtol=rtol, atol=atol):
            _fail("a fresh instance loaded with the state dict computes different outputs "
                  "(state that influences forward is not registered as a parameter or persistent buffer)")

    return ContractReport(model=label, parameters=count_parameters(model),
                          input_shape=(batch_size, in_channels, in_samples), output_shape=tuple(y1.shape),
                          fwd_bwd_sec=fwd_bwd, initial_loss=float(loss.detach()), peak_mem_gb=peak)
