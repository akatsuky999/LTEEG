"""Template for a new model. Copy the whole ``_template`` folder to start::

    cp -r lteeg/models/_template lteeg/models/my_net      # the folder name is the model name

then

1. rename ``TemplateNet`` and implement ``__init__`` / ``forward`` below, keeping the
   first three constructor arguments (``in_channels, in_samples, num_outputs``);
   add ``fs`` and/or ``channel_names`` to the signature if the model needs them;
2. put the building blocks in ``layers/`` and export them from ``layers/__init__.py``;
3. in ``__init__.py`` set ``Model = <your class>`` and small ``SMOKE_PARAMS``;
4. fill in ``README.md`` (source, parameters, deviations from the reference);
5. run ``python -m pytest tests/test_model_contract.py`` (picks the new folder up
   automatically) and ``python -m lteeg check-model --set model.name=my_net``;
6. train with ``python -m lteeg train --set model.name=my_net "model.params={...}"``.

The contract (shapes, optional members) is documented in ``lteeg/models/base.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvBlock


class TemplateNet(nn.Module):
    """Strided convolutional stem -> ``depth`` residual blocks -> 1x1 head.

    Emits one logit vector every ``stride`` samples (``T' = ceil(T / stride)``); the
    framework interpolates the logits back to one value per sample.
    """

    def __init__(self, in_channels: int, in_samples: int, num_outputs: int = 1,
                 hidden: int = 32, depth: int = 2, stride: int = 4):
        super().__init__()
        if stride < 1 or hidden < 1 or depth < 0:
            raise ValueError("stride and hidden must be >= 1, depth >= 0")
        self.output_stride = stride
        self.stem = nn.Conv1d(in_channels, hidden, kernel_size=2 * stride + 1, stride=stride, padding=stride)
        self.blocks = nn.Sequential(*[ConvBlock(hidden) for _ in range(depth)])
        self.head = nn.Conv1d(hidden, num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, num_outputs, T')
        return self.head(self.blocks(self.stem(x)))
