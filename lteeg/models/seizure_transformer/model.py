"""SeizureTransformer (Wu, Zhao & Yener, 2025; arXiv:2504.00336).

U-shaped sequence-to-sequence network, winner of the 2025 SzCORE seizure detection
challenge: a 1-D convolutional encoder (5 levels, max-pool /2), a residual CNN
stack and a Transformer encoder at the bottleneck (T/32 tokens), and a decoder with
nearest upsampling and additive skip connections that emits one score per input
sample.

Port of the original ``time_step_level/model.py`` with these deliberate changes
(none alters the function computed for the original configuration):

* returns **logits** ``(B, num_outputs, T)`` instead of ``sigmoid`` probabilities, so
  the framework can use numerically stable ``BCEWithLogits`` (also required under
  mixed precision) and multi-class heads;
* the unused ``self.transformer_encoder_layer`` attribute is removed
  (``nn.TransformerEncoder`` deep-copies the layer it is given, so the original
  carried ~3.2M dead parameters that the optimizer still had to track);
* odd lengths are handled with ``MaxPool1d(ceil_mode=True)`` (identical to the
  original right-padding with -1e10) and the decoder crops to each skip's length,
  so the network accepts any input length >= 32, not only ``in_samples``;
* the positional-encoding table is sized from the input length (the original fixed
  6000 tokens, i.e. at most 192 000 samples) and is a non-persistent buffer;
* the Transformer runs ``batch_first`` (same parameters, faster kernels).

Parameter names are unchanged, so weights trained with the original code load via
:func:`load_original_state_dict` (applied automatically by
:func:`lteeg.models.load_weights` through :meth:`SeizureTransformer.convert_state_dict`).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn

from .layers import Decoder, Encoder, PositionalEncoding, ResCNNStack


def load_original_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a state dict saved by the original repository (``torch.save(model.state_dict())``)
    for :class:`SeizureTransformer`: drops the dead ``transformer_encoder_layer.*`` weights,
    the positional table and any ``module.`` prefix from DataParallel. A state dict of
    this port passes through unchanged.

    Note the original model applied ``sigmoid`` inside ``forward``; the converted model
    returns logits, so apply ``sigmoid`` (the framework does) to get identical outputs.
    """
    out = {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith("transformer_encoder_layer.") or k == "position_encoding.pe":
            continue
        out[k] = v
    return out


class SeizureTransformer(nn.Module):
    output_stride = 1

    def __init__(
        self,
        in_channels: int = 18,
        in_samples: int = 15360,
        num_outputs: int = 1,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        drop_rate: float = 0.1,
        transformer_dropout: float = 0.1,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        kernel_sizes: Sequence[int] = (11, 9, 7, 7, 5, 5, 3),
        res_cnn_kernels: Sequence[int] = (3, 3, 3, 3, 2, 3, 2),
        norm_first: bool = False,
    ):
        super().__init__()
        filters, kernel_sizes = list(filters), list(kernel_sizes)
        if len(kernel_sizes) < len(filters):
            raise ValueError("need at least one kernel size per encoder level")
        self.in_channels, self.in_samples, self.num_outputs = in_channels, in_samples, num_outputs
        self.min_samples = 2 ** len(filters)
        d_model = filters[-1]
        self.encoder = Encoder(in_channels, filters, kernel_sizes)
        self.res_cnn_stack = ResCNNStack(res_cnn_kernels, d_model, drop_rate)
        max_tokens = max(6000, math.ceil(in_samples / 2 ** len(filters)) * 4)
        self.position_encoding = PositionalEncoding(d_model, dropout=transformer_dropout, max_len=max_tokens)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                                           dropout=transformer_dropout, batch_first=True, norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                         enable_nested_tensor=False)
        # same zip-truncation as the original: decoder kernels are the reversed list's head
        self.decoder_d = Decoder(d_model, filters[::-1], kernel_sizes[::-1])
        self.conv_d = nn.Conv1d(filters[0], num_outputs, kernel_size=11, padding=5)

    # weights saved by the original repository load directly (see lteeg.models.load_weights)
    convert_state_dict = staticmethod(load_original_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, T), got {tuple(x.shape)}")
        if x.shape[-1] < self.min_samples:
            raise ValueError(f"input length {x.shape[-1]} < minimum {self.min_samples}")
        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)
        h = self.position_encoding(res_x.transpose(1, 2))
        h = self.transformer_encoder(h).transpose(1, 2)
        x = h + res_x
        x = self.decoder_d(x, skips)
        return self.conv_d(x)
