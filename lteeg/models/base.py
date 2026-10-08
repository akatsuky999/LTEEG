"""The contract every model in LTEEG satisfies. Read this before writing a model.

Where a model lives
-------------------
One package per model, the folder name is the model name::

    lteeg/models/<name>/
    ├── __init__.py   exports ``Model`` (the network class); optional ``SMOKE_PARAMS``
    ├── model.py      the network: assembles the blocks and defines ``forward``
    ├── layers/       the building blocks used only by this model
    └── README.md     source paper/code, parameters, deviations from the reference

``model.name: <name>`` in the config selects ``lteeg.models.<name>.Model``; only that
package is imported. A model kept outside the framework is selected with
``model.name: "package.module:Class"``. ``lteeg/models/_template/`` is a ready-to-copy
skeleton.

Constructor
-----------
::

    Model(in_channels: int, in_samples: int, num_outputs: int, **params)

* ``in_channels`` = ``len(data.channels)``; ``in_samples`` = training window length in
  samples; ``num_outputs`` = 1 (binary task) or the number of classes (multiclass).
* ``params`` = ``model.params`` from the config; a misspelled name is an error.
* Optional context: the framework also passes ``fs`` (sampling rate in Hz) and
  ``channel_names`` (``data.channels``) when -- and only when -- the constructor
  declares a parameter with that exact name. Context values cannot be set in
  ``model.params``.

Forward
-------
``forward(x)`` receives ``x`` of shape ``(B, C, T)`` (float32; bf16/fp16 autocast may be
active) and returns either

* a tensor of logits ``(B, num_outputs, T')`` with ``1 <= T' <= T``, or
* a :class:`ModelOutput`: the main logits plus optional auxiliary logits (deep
  supervision / multi-stage refinement, each scored with the main loss) and
  ready-to-add auxiliary loss terms (e.g. a reconstruction loss).

``T' < T`` is allowed (token- or segment-level models): the framework interpolates the
logits linearly to one value per sample, treating the ``T'`` values as the centres of
``T'`` equal cells covering the input. ``in_samples`` is the training window length;
length-agnostic models may ignore it. ``forward`` must not modify ``x`` in place.

Optional members
----------------
* ``wants_meta = True`` -> ``forward(x, meta)`` receives a dict with ``patient``,
  ``rec`` and ``start`` tensors (patient-conditioned or adaptive models; ``-1`` when
  unknown, e.g. at inference).
* ``convert_state_dict(state) -> state`` (static or class method): applied by
  :func:`lteeg.models.load_weights` to external weights, e.g. checkpoints saved by the
  model's original implementation.
* ``output_stride``: informative ``T / T'``.

Checked automatically
---------------------
``tests/test_model_contract.py`` runs :func:`lteeg.models.contract.check_contract` on
every model package (with its ``SMOKE_PARAMS``): output type, shape and dtype, finite
values, gradients reaching the parameters, input left untouched, reproducible eval
outputs, no interaction between batch elements in eval mode, and a strict state-dict
round trip. ``python -m lteeg check-model`` runs the same checks with the real config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_logits: List[torch.Tensor] = field(default_factory=list)
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)


def as_output(out: Union[torch.Tensor, ModelOutput, dict]) -> ModelOutput:
    if isinstance(out, ModelOutput):
        return out
    if isinstance(out, torch.Tensor):
        return ModelOutput(out)
    if isinstance(out, dict):
        return ModelOutput(out["logits"], list(out.get("aux_logits", [])), dict(out.get("losses", {})))
    raise TypeError(f"Model returned unsupported type {type(out).__name__}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
